"""Scaling curve on the Marvel instance pool: cost and recall as the target pool grows.

    python scripts/scaling_benchmark.py --config-path config/runtime.paper.yaml \
        --selected-settings-json <heldout_selected_settings.json> \
        --output-csv-path results/scaling/scaling.csv

The query set is fixed: every Marvel (marvelcinematicuniverse -> marvel) instance query.
The target pool varies:

- nested subsamples of the Marvel instance pool (all gold targets plus a seeded prefix of a
  fixed permutation of the other targets), so each smaller pool is contained in the larger;
- the full Marvel pool (532,781 targets);
- a pooled set: the Marvel pool plus the instance target pools of the other four KG Track
  pairs (about 709k targets), i.e. targets beyond the largest benchmark pool.

Each (pool, method) measurement runs in a fresh spawned process with single-threaded BLAS,
using the same protocol as ``src.experiments.efficiency_benchmark`` (warm-up, repeats,
peak RSS). In addition the first timed pass records the rank of each gold target within the
top ``k``, giving pair-level Recall@1/10/50 and MRR@50 at that pool size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing
import os
import pickle
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_runtime_config  # noqa: E402
from src.experiments.efficiency_benchmark import (  # noqa: E402
    THREAD_ENV_VARS,
    MethodSpec,
    _library_versions,
    _max_rss_mb,
    make_retriever,
    method_specs_from_selected_settings,
)
from src.experiments.experiment_runner import _build_labels, _load_store_with_fallback  # noqa: E402
from src.rdf_utils.alignment_parser import load_alignment_targets  # noqa: E402
from src.rdf_utils.entity_partitioning import (  # noqa: E402
    extract_typed_entity_partitions,
    partition_alignment_mappings_by_entity_type,
)

logger = logging.getLogger("scaling")

QUERY_DATASET = "marvelcinematicuniverse-marvel"
SUBSAMPLE_SIZES = (10_000, 25_000, 50_000, 100_000, 200_000, 350_000)
COLUMNS = (
    "pool", "target_pool_size", "distractor_source", "method", "hyperparameters", "query_count",
    "gold_pairs", "k", "warmup_queries", "repeats", "build_seconds", "index_bytes",
    "latency_median_ms", "latency_p95_ms", "latency_mean_ms", "queries_per_second",
    "mean_candidate_count", "rss_before_fit_mb", "peak_rss_mb", "peak_rss_delta_mb",
    "recall_at_1", "recall_at_10", "recall_at_50", "mrr", "threads", "seed",
    "python", "numpy", "scikit_learn", "bm25s", "platform",
)


def measure(kwargs: dict) -> dict[str, float]:
    """Fit, time and score one method on one pool. Runs in a fresh process."""
    method, hp, options = kwargs["method"], kwargs["hyperparameters"], kwargs["runtime_options"]
    queries, gold, k = kwargs["query_labels"], kwargs["gold"], kwargs["k"]
    if method.startswith(("dense_", "hybrid_")):
        from src.retrieval.dense_retriever import _load_model

        opts = options or {}
        _load_model(str(hp["model"]), str(hp["revision"]), str(opts.get("device", "cpu")))
    rss_before = _max_rss_mb()
    retriever = make_retriever(method, hp, kwargs["candidate_policy"], options)
    start = time.perf_counter()
    retriever.fit(kwargs["target_ids"], kwargs["target_labels"])
    build_seconds = time.perf_counter() - start
    index_bytes = len(pickle.dumps(retriever, protocol=pickle.HIGHEST_PROTOCOL))
    for query in queries[: kwargs["warmup"]]:
        retriever.retrieve(query, k)

    latencies: list[float] = []
    candidate_total = 0
    ranks: list[int] = []  # one per gold pair; 0 = not in top k
    for repeat in range(kwargs["repeats"]):
        for query, targets in zip(queries, gold):
            t0 = time.perf_counter()
            candidates = retriever.retrieve(query, k)
            latencies.append(time.perf_counter() - t0)
            candidate_total += len(candidates)
            if repeat == 0:
                position = {cid: i + 1 for i, (cid, _) in enumerate(candidates)}
                ranks.extend(position.get(t, 0) for t in targets)
    ms = np.asarray(latencies) * 1000.0
    r = np.asarray(ranks)
    hit = lambda n: float(((r >= 1) & (r <= n)).mean())  # noqa: E731
    peak = _max_rss_mb()
    return {
        "build_seconds": build_seconds,
        "index_bytes": float(index_bytes),
        "latency_median_ms": float(np.median(ms)),
        "latency_p95_ms": float(np.percentile(ms, 95)),
        "latency_mean_ms": float(ms.mean()),
        "queries_per_second": float(len(ms) / (ms.sum() / 1000.0)),
        "mean_candidate_count": candidate_total / len(ms),
        "rss_before_fit_mb": rss_before,
        "peak_rss_mb": peak,
        "peak_rss_delta_mb": peak - rss_before,
        "recall_at_1": hit(1),
        "recall_at_10": hit(10),
        "recall_at_50": hit(50),
        "mrr": float(np.where(r >= 1, 1.0 / np.maximum(r, 1), 0.0).mean()),
        "gold_pairs": len(r),
    }


def measure_isolated(kwargs: dict) -> dict[str, float]:
    previous = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
    try:
        for name in THREAD_ENV_VARS:
            os.environ[name] = "1"
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as ex:
            return ex.submit(measure, kwargs).result()
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def load_pools(runtime, seed: int, sizes: tuple[int, ...]):
    datasets = runtime.heldout_datasets
    query_ds = datasets[QUERY_DATASET]
    source_store = _load_store_with_fallback(query_ds.source_rdf)
    target_store = _load_store_with_fallback(query_ds.target_rdf)
    source_parts = extract_typed_entity_partitions(source_store, graph_name="source")
    target_parts = extract_typed_entity_partitions(target_store, graph_name="target")
    gold_raw = load_alignment_targets(query_ds.alignment_rdf, runtime.experiments.alignment_policy)
    gold_all = partition_alignment_mappings_by_entity_type(
        gold_raw, source_parts, target_parts, alignment_name=QUERY_DATASET
    ).instances
    sources = sorted(gold_all)
    gold = [tuple(gold_all[s]) for s in sources]
    query_labels = _build_labels(source_store, sources)

    marvel_ids = list(target_parts.instances)
    marvel_labels = _build_labels(target_store, marvel_ids)
    keep = {t for targets in gold for t in targets}
    gold_idx = [i for i, t in enumerate(marvel_ids) if t in keep]
    others = [i for i, t in enumerate(marvel_ids) if t not in keep]
    random.Random(seed).shuffle(others)

    pools: list[tuple[str, str, list[str], list[str]]] = []
    for size in sizes:
        if size >= len(marvel_ids):
            continue
        chosen = sorted(gold_idx + others[: size - len(gold_idx)])
        pools.append((f"marvel_{size}", "marvel",
                      [marvel_ids[i] for i in chosen], [marvel_labels[i] for i in chosen]))
    pools.append(("marvel_full", "marvel", marvel_ids, marvel_labels))

    pooled_ids, pooled_labels = list(marvel_ids), list(marvel_labels)
    seen = set(pooled_ids)
    for name, ds in datasets.items():
        if name == QUERY_DATASET:
            continue
        store = _load_store_with_fallback(ds.target_rdf)
        ids = [t for t in extract_typed_entity_partitions(store, graph_name="target").instances if t not in seen]
        seen.update(ids)
        pooled_ids += ids
        pooled_labels += _build_labels(store, ids)
        logger.info("Pooled %d instance targets from %s", len(ids), name)
    pools.append(("pooled_kg", "marvel+4 KG pairs", pooled_ids, pooled_labels))
    return query_labels, gold, pools


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    return commit, dirty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-path", default="config/runtime.paper.yaml")
    parser.add_argument("--selected-settings-json", required=True)
    parser.add_argument("--output-csv-path", required=True, type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SUBSAMPLE_SIZES))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--max-queries", type=int, default=None, help="Smoke tests only.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    runtime = load_runtime_config(config_path=args.config_path)
    k = max(runtime.experiments.evaluation_ks)
    specs = method_specs_from_selected_settings(args.selected_settings_json)
    specs += [MethodSpec(e.name, e.hyperparameters(), {"device": e.device, "batch_size": e.batch_size})
              for e in runtime.experiments.fixed_baselines_for("kg")]
    if args.methods:
        specs = [s for s in specs if s.method in set(args.methods)]
    query_labels, gold, pools = load_pools(runtime, args.seed, tuple(args.sizes))
    if args.max_queries:
        query_labels, gold = query_labels[: args.max_queries], gold[: args.max_queries]
    versions = _library_versions()
    rows: list[dict] = []
    args.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    for pool, source, ids, labels in pools:
        for spec in specs:
            logger.info("pool=%s targets=%d method=%s queries=%d", pool, len(ids), spec.method, len(query_labels))
            measured = measure_isolated(dict(
                method=spec.method, hyperparameters=spec.hyperparameters, runtime_options=spec.runtime_options,
                candidate_policy=runtime.experiments.candidate_policy, target_ids=ids, target_labels=labels,
                query_labels=query_labels, gold=gold, k=k, warmup=args.warmup, repeats=args.repeats))
            rows.append({
                "pool": pool, "target_pool_size": len(ids), "distractor_source": source, "method": spec.method,
                "hyperparameters": json.dumps(spec.hyperparameters, sort_keys=True, separators=(",", ":")),
                "query_count": len(query_labels), "k": k, "warmup_queries": min(args.warmup, len(query_labels)),
                "repeats": args.repeats, **measured, "threads": 1, "seed": args.seed, **versions})
            logger.info("  build=%.1fs median=%.2fms R@50=%.3f MRR=%.3f", measured["build_seconds"],
                        measured["latency_median_ms"], measured["recall_at_50"], measured["mrr"])
            with args.output_csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
                writer.writeheader()
                writer.writerows(rows)
    commit, dirty = git_state()
    meta = {
        "script": "scripts/scaling_benchmark.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "commit": commit, "working_tree_dirty": dirty, "config": args.config_path,
        "selected_settings_json": args.selected_settings_json, "query_dataset": QUERY_DATASET,
        "sizes": args.sizes, "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
        "max_queries": args.max_queries, "environment": {k: os.environ.get(k) for k in
                                                         ("KGCANDGEN_DATA_ROOT", "PYTHONHASHSEED", "HF_HUB_OFFLINE")},
        "output_sha256": hashlib.sha256(args.output_csv_path.read_bytes()).hexdigest(), **versions,
    }
    args.output_csv_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %d rows to %s", len(rows), args.output_csv_path)


if __name__ == "__main__":
    main()
