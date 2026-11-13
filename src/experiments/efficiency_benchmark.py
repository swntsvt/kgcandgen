"""Efficiency benchmark for candidate-generation methods.

Each (dataset, entity type, method) measurement runs in a fresh spawned process so that
its peak resident memory is not mixed with other methods or with RDF loading. The parent
loads labels once and sends them to the child; the child then measures:

- ``build_seconds``: ``fit`` on the target labels, including the method's own preprocessing
- ``index_bytes``: size of the pickled fitted retriever (a uniform, serialised index size)
- query latency after ``warmup`` untimed queries, over ``repeats`` passes of the query set:
  median, p95 and mean in milliseconds, plus queries per second
- ``peak_rss_delta_mb``: peak RSS of the child minus its RSS before ``fit`` (labels loaded)

BLAS/OpenMP thread pools are pinned to one thread so methods are compared single-threaded.
"""

from __future__ import annotations

import csv
import logging
import multiprocessing
import os
import pickle
import platform
import random
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config_loader import DatasetConfig, load_runtime_config
from src.experiments.experiment_runner import (
    _build_labels,
    _extract_owl_class_entities,
    _load_store_with_fallback,
)
from src.rdf_utils.alignment_parser import filter_alignment_targets, load_alignment_targets
from src.rdf_utils.entity_partitioning import (
    EntityType,
    extract_typed_entity_partitions,
    partition_alignment_mappings_by_entity_type,
)
from src.run_metadata import build_timestamped_results_path

logger = logging.getLogger(__name__)

TIERS: tuple[str, ...] = ("development", "kg", "class")
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

EFFICIENCY_CSV_COLUMNS: tuple[str, ...] = (
    "tier",
    "track",
    "dataset",
    "entity_type",
    "method",
    "hyperparameters",
    "candidate_policy",
    "target_pool_size",
    "target_limit",
    "query_count",
    "warmup_queries",
    "repeats",
    "k",
    "build_seconds",
    "index_bytes",
    "latency_median_ms",
    "latency_p95_ms",
    "latency_mean_ms",
    "queries_per_second",
    "mean_candidate_count",
    "rss_before_fit_mb",
    "peak_rss_mb",
    "peak_rss_delta_mb",
    "threads",
    "python",
    "numpy",
    "scikit_learn",
    "bm25s",
    "platform",
)


@dataclass(frozen=True)
class BenchmarkInputs:
    """Labels for one (tier, dataset, entity type) unit."""

    tier: str
    track: str
    dataset: str
    entity_type: str
    target_ids: list[str]
    target_labels: list[str]
    query_labels: list[str]
    full_target_pool_size: int


@dataclass(frozen=True)
class MethodSpec:
    method: str
    hyperparameters: dict[str, object]
    runtime_options: dict[str, object] | None = None


def _max_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def make_retriever(
    method: str,
    hyperparameters: dict[str, object],
    candidate_policy: str,
    runtime_options: dict[str, object] | None = None,
):
    """Construct an unfitted retriever exposing ``fit(ids, labels)`` and ``retrieve(text, k)``."""
    if method.startswith("dense_"):
        from src.retrieval.dense_retriever import DenseRetriever

        options = runtime_options or {}
        return DenseRetriever(
            model_name=str(hyperparameters["model"]),
            revision=str(hyperparameters["revision"]),
            device=str(options.get("device", "cpu")),
            batch_size=int(options.get("batch_size", 256)),  # type: ignore[arg-type]
            candidate_policy=candidate_policy,  # type: ignore[arg-type]
        )
    if method == "tfidf":
        from src.retrieval.tfidf_retriever import TfidfRetriever

        ngram = hyperparameters["ngram_range"]
        return TfidfRetriever(
            ngram_range=(int(ngram[0]), int(ngram[1])),  # type: ignore[index]
            min_df=hyperparameters["min_df"],  # type: ignore[arg-type]
            max_df=hyperparameters["max_df"],  # type: ignore[arg-type]
            sublinear_tf=bool(hyperparameters["sublinear_tf"]),
            candidate_policy=candidate_policy,  # type: ignore[arg-type]
        )
    if method == "bm25":
        from src.retrieval.bm25_retriever import Bm25Retriever

        return Bm25Retriever(
            k1=float(hyperparameters["k1"]),  # type: ignore[arg-type]
            b=float(hyperparameters["b"]),  # type: ignore[arg-type]
            candidate_policy=candidate_policy,  # type: ignore[arg-type]
        )
    if method == "exact_match":
        from src.retrieval.exact_match_retriever import ExactMatchRetriever

        return ExactMatchRetriever()
    if method == "char_ngram":
        from src.retrieval.char_ngram_retriever import CharNgramRetriever

        ngram = hyperparameters.get("ngram_range", [3, 5])
        return CharNgramRetriever(
            analyzer=str(hyperparameters.get("analyzer", "char_wb")),  # type: ignore[arg-type]
            ngram_range=(int(ngram[0]), int(ngram[1])),  # type: ignore[index]
            candidate_policy=candidate_policy,  # type: ignore[arg-type]
        )
    raise ValueError(f"Unsupported method for efficiency benchmark: {method!r}")


def measure_method(
    *,
    method: str,
    hyperparameters: dict[str, object],
    candidate_policy: str,
    target_ids: list[str],
    target_labels: list[str],
    query_labels: list[str],
    k: int,
    warmup: int,
    repeats: int,
    runtime_options: dict[str, object] | None = None,
) -> dict[str, float]:
    """Fit and time one method in the current process. Run it in a fresh process for RSS.

    For dense methods the model is loaded before timing starts, so ``build_seconds`` is
    the encoding of the target labels and the RSS baseline includes the model weights.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    if not query_labels:
        raise ValueError("query_labels must not be empty.")
    if method.startswith("dense_"):
        from src.retrieval.dense_retriever import _load_model

        options = runtime_options or {}
        _load_model(
            str(hyperparameters["model"]),
            str(hyperparameters["revision"]),
            str(options.get("device", "cpu")),
        )
    rss_before_fit = _max_rss_mb()
    retriever = make_retriever(method, hyperparameters, candidate_policy, runtime_options)

    start = time.perf_counter()
    retriever.fit(target_ids, target_labels)
    build_seconds = time.perf_counter() - start
    index_bytes = len(pickle.dumps(retriever, protocol=pickle.HIGHEST_PROTOCOL))

    for query in query_labels[:warmup]:
        retriever.retrieve(query, k)

    latencies: list[float] = []
    candidate_total = 0
    for _ in range(repeats):
        for query in query_labels:
            query_start = time.perf_counter()
            candidates = retriever.retrieve(query, k)
            latencies.append(time.perf_counter() - query_start)
            candidate_total += len(candidates)

    latency_ms = np.asarray(latencies) * 1000.0
    peak_rss = _max_rss_mb()
    return {
        "build_seconds": build_seconds,
        "index_bytes": float(index_bytes),
        "latency_median_ms": float(np.median(latency_ms)),
        "latency_p95_ms": float(np.percentile(latency_ms, 95)),
        "latency_mean_ms": float(latency_ms.mean()),
        "queries_per_second": float(len(latencies) / (latency_ms.sum() / 1000.0))
        if latency_ms.sum() > 0
        else float("inf"),
        "mean_candidate_count": candidate_total / len(latencies),
        "rss_before_fit_mb": rss_before_fit,
        "peak_rss_mb": peak_rss,
        "peak_rss_delta_mb": peak_rss - rss_before_fit,
    }


def _measure_in_child(kwargs: dict[str, object]) -> dict[str, float]:
    return measure_method(**kwargs)  # type: ignore[arg-type]


def measure_method_isolated(**kwargs: object) -> dict[str, float]:
    """Run ``measure_method`` in a fresh spawned process with single-threaded BLAS."""
    previous = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
    try:
        for name in THREAD_ENV_VARS:
            os.environ[name] = "1"
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            return executor.submit(_measure_in_child, dict(kwargs)).result()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _subsample_targets(
    target_ids: list[str],
    target_labels: list[str],
    keep: set[str],
    target_limit: int | None,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Keep all ``keep`` targets plus a seeded sample of the rest, up to ``target_limit``."""
    if target_limit is None or target_limit >= len(target_ids):
        return target_ids, target_labels
    kept = [i for i, target in enumerate(target_ids) if target in keep]
    others = [i for i, target in enumerate(target_ids) if target not in keep]
    room = max(0, target_limit - len(kept))
    chosen = sorted(kept + random.Random(seed).sample(others, min(room, len(others))))
    return [target_ids[i] for i in chosen], [target_labels[i] for i in chosen]


def load_benchmark_inputs(
    tier: str,
    dataset: DatasetConfig,
    *,
    alignment_policy: str,
    entity_types: Iterable[str] = ("class", "predicate", "instance"),
    target_limit: int | None = None,
    max_queries: int | None = None,
    seed: int = 13,
) -> list[BenchmarkInputs]:
    """Load target labels and query labels for each evaluated unit of a dataset."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {list(TIERS)}; got {tier!r}.")
    source_store = _load_store_with_fallback(dataset.source_rdf)
    target_store = _load_store_with_fallback(dataset.target_rdf)
    gold_raw = load_alignment_targets(dataset.alignment_rdf, alignment_policy)  # type: ignore[arg-type]

    units: list[tuple[str, list[str], dict[str, tuple[str, ...]]]] = []
    if tier == "kg":
        source_partitions = extract_typed_entity_partitions(source_store, graph_name="source")
        target_partitions = extract_typed_entity_partitions(target_store, graph_name="target")
        gold_partitions = partition_alignment_mappings_by_entity_type(
            gold_raw, source_partitions, target_partitions, alignment_name=dataset.name
        )
        by_type = {
            "class": (list(target_partitions.classes), gold_partitions.classes),
            "predicate": (list(target_partitions.predicates), gold_partitions.predicates),
            "instance": (list(target_partitions.instances), gold_partitions.instances),
        }
        for entity_type in entity_types:
            targets, gold = by_type[EntityType(entity_type).value]
            units.append((entity_type, targets, gold))
    else:
        sources = _extract_owl_class_entities(source_store)
        targets = _extract_owl_class_entities(target_store)
        units.append(("class", targets, filter_alignment_targets(gold_raw, set(sources), set(targets))))

    inputs: list[BenchmarkInputs] = []
    for entity_type, targets, gold in units:
        if not targets or not gold:
            logger.warning("Skipping %s/%s: no targets or no gold", dataset.name, entity_type)
            continue
        sources = sorted(gold)
        if max_queries is not None and len(sources) > max_queries:
            sources = sorted(random.Random(seed).sample(sources, max_queries))
        keep = {target for source in sources for target in gold[source]}
        labels = _build_labels(target_store, targets)
        sampled_ids, sampled_labels = _subsample_targets(targets, labels, keep, target_limit, seed)
        inputs.append(
            BenchmarkInputs(
                tier=tier,
                track=dataset.track,
                dataset=dataset.name,
                entity_type=entity_type,
                target_ids=sampled_ids,
                target_labels=sampled_labels,
                query_labels=_build_labels(source_store, sources),
                full_target_pool_size=len(targets),
            )
        )
    return inputs


def _library_versions() -> dict[str, str]:
    import bm25s
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "bm25s": getattr(bm25s, "__version__", "unknown"),
        "platform": platform.platform(),
    }


def method_specs_from_selected_settings(
    selected_settings_path: str | Path | None,
) -> list[MethodSpec]:
    """Return the frozen tunable settings plus the fixed baselines, in registry order.

    ``None`` resolves to the latest ``heldout_selected_settings.json``, as the held-out runners do.
    """
    from src.experiments.heldout_kg_runner import (
        _build_heldout_method_runs,
        _load_selected_settings,
        _resolve_selected_settings_json,
    )

    resolved = _resolve_selected_settings_json(selected_settings_path)
    runs = _build_heldout_method_runs(_load_selected_settings(resolved))
    return [MethodSpec(run.method_name, dict(run.hyperparameters)) for run in runs]


def run_efficiency_benchmark(
    *,
    config_path: str | Path = "config/runtime.yaml",
    selected_settings_path: str | Path | None = None,
    tiers: Iterable[str] = TIERS,
    datasets: Iterable[str] | None = None,
    entity_types: Iterable[str] = ("class", "predicate", "instance"),
    methods: Iterable[str] | None = None,
    k: int | None = None,
    warmup: int = 20,
    repeats: int = 3,
    max_queries: int | None = None,
    target_limit: int | None = None,
    seed: int = 13,
    output_csv_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Benchmark every requested method on every requested unit and write one CSV."""
    runtime = load_runtime_config(config_path=config_path)
    k_value = k if k is not None else max(runtime.experiments.evaluation_ks)
    candidate_policy = runtime.experiments.candidate_policy
    specs = method_specs_from_selected_settings(selected_settings_path)
    if methods is not None:
        wanted = set(methods)
        specs = [spec for spec in specs if spec.method in wanted]
    groups = {
        "development": runtime.development_datasets,
        "kg": runtime.heldout_datasets,
        "class": runtime.heldout_secondary_datasets,
    }
    wanted_datasets = set(datasets) if datasets is not None else None
    versions = _library_versions()
    output_path = (
        Path(output_csv_path)
        if output_csv_path is not None
        else build_timestamped_results_path("efficiency")
    )

    rows: list[dict[str, object]] = []
    for tier in tiers:
        for name, dataset in groups[tier].items():
            if wanted_datasets is not None and name not in wanted_datasets:
                continue
            for unit in load_benchmark_inputs(
                tier,
                dataset,
                alignment_policy=runtime.experiments.alignment_policy,
                entity_types=entity_types,
                target_limit=target_limit,
                max_queries=max_queries,
                seed=seed,
            ):
                tier_specs = [
                    *specs,
                    *(
                        MethodSpec(
                            entry.name,
                            entry.hyperparameters(),
                            {"device": entry.device, "batch_size": entry.batch_size},
                        )
                        for entry in runtime.experiments.dense_models_for(tier)
                        if methods is None or entry.name in set(methods)
                    ),
                ]
                for spec in tier_specs:
                    logger.info(
                        "Benchmarking %s/%s/%s method=%s targets=%d queries=%d",
                        tier,
                        unit.dataset,
                        unit.entity_type,
                        spec.method,
                        len(unit.target_ids),
                        len(unit.query_labels),
                    )
                    measured = measure_method_isolated(
                        method=spec.method,
                        hyperparameters=spec.hyperparameters,
                        candidate_policy=candidate_policy,
                        target_ids=unit.target_ids,
                        target_labels=unit.target_labels,
                        query_labels=unit.query_labels,
                        k=k_value,
                        warmup=warmup,
                        repeats=repeats,
                        runtime_options=spec.runtime_options,
                    )
                    rows.append(
                        {
                            "tier": tier,
                            "track": unit.track,
                            "dataset": unit.dataset,
                            "entity_type": unit.entity_type,
                            "method": spec.method,
                            "hyperparameters": _json(spec.hyperparameters),
                            "candidate_policy": candidate_policy,
                            "target_pool_size": len(unit.target_ids),
                            "target_limit": "" if target_limit is None else target_limit,
                            "query_count": len(unit.query_labels),
                            "warmup_queries": min(warmup, len(unit.query_labels)),
                            "repeats": repeats,
                            "k": k_value,
                            **measured,
                            "threads": 1,
                            **versions,
                        }
                    )
                    _write_rows(rows, output_path)
    logger.info("Wrote %d efficiency row(s) to %s", len(rows), output_path)
    return rows


def _json(value: dict[str, object]) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_rows(rows: list[dict[str, object]], output_path: Path) -> None:
    """Rewrite the CSV after every measurement so a long run keeps partial results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EFFICIENCY_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
