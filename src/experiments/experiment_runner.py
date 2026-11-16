"""Experiment runner for candidate generation retrieval studies."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Protocol, TypedDict, cast

from pyoxigraph import NamedNode, RdfFormat, Store
from tqdm import tqdm

from src.config_loader import (
    Bm25GridEntry,
    CharNgramGridEntry,
    FixedBaselineEntry,
    TfidfGridEntry,
    load_runtime_config,
)
from src.evaluation.metrics import compute_retrieval_metrics, mean_candidate_count
from src.experiments.query_records import (
    QueryRecord,
    build_query_records,
    query_output_csv_path,
    write_query_csv,
)
from src.method_registry import development_method_names, fixed_method_hyperparameters
from src.preprocessing.text_preprocessor import preprocess_text, validate_nltk_assets
from src.rdf_utils.alignment_parser import (
    filter_alignment_targets,
    load_alignment_targets,
    multi_target_source_count,
)
from src.rdf_utils.label_extractor import extract_entity_label
from src.retrieval.baselines import build_fixed_baseline, is_fixed_baseline
from src.retrieval.bm25_retriever import Bm25Retriever
from src.retrieval.char_ngram_retriever import CharNgramRetriever
from src.retrieval.exact_match_retriever import ExactMatchRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.run_metadata import build_timestamped_results_path, get_git_short_sha

RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
OWL_CLASS = NamedNode("http://www.w3.org/2002/07/owl#Class")
FIXED_CSV_COLUMNS: tuple[str, ...] = (
    "track",
    "version",
    "dataset",
    "method",
    "hyperparameters",
    "gold_count",
    "candidate_size",
    "dataset_prep_seconds",
    "mrr",
    "runtime_seconds",
)
# Appended after the submitted-version columns so existing readers keep working.
EXTRA_CSV_COLUMNS: tuple[str, ...] = (
    "query_count",
    "multi_target_query_count",
    "mean_candidate_count",
)
TRAILING_CSV_COLUMNS: tuple[str, ...] = (
    "any_gold_mrr",
    "rank_cutoff",
    "candidate_policy",
    "alignment_policy",
    "zero_candidate_query_share",
)
CHAR_NGRAM_SENSITIVITY_METHOD = "char_ngram_sensitivity"
ERROR_CSV_COLUMNS: tuple[str, ...] = ("scope", "dataset", "method", "hyperparameters", "error")

logger = logging.getLogger(__name__)


class RetrieverProtocol(Protocol):
    def fit(self, entity_ids: list[str], labels: list[str]) -> None: ...
    def retrieve(self, query_text: str, k: int) -> list[tuple[str, float]]: ...


class ExperimentResultRecord(TypedDict):
    dataset_name: str
    track: str
    version: str
    model: str
    hyperparameters: dict[str, object]
    num_source_entities: int
    num_target_entities: int
    num_gold_pairs: int
    gold_count: int
    candidate_size: int
    dataset_prep_seconds: float
    recalls: dict[int, float]
    mrr: float
    runtime_seconds: float
    query_count: int
    multi_target_query_count: int
    mean_candidate_count: float
    any_gold_recalls: dict[int, float]
    any_gold_mrr: float
    rank_cutoff: int
    candidate_policy: str
    alignment_policy: str
    zero_candidate_query_share: float


class ExperimentErrorRecord(TypedDict):
    scope: str
    dataset_name: str
    model: str
    hyperparameters: dict[str, object]
    error: str


@dataclass(frozen=True)
class ModelRunSpec:
    model_name: str
    hyperparameters: dict[str, object]
    retriever: RetrieverProtocol


def _load_store_with_fallback(path: Path) -> Store:
    store = Store()
    try:
        store.load(path=str(path), format=RdfFormat.RDF_XML, lenient=False)
    except SyntaxError as exc:
        logger.warning(
            "Strict RDF/XML parse failed for %s; retrying in lenient mode: %s",
            path,
            exc,
        )
        store = Store()
        store.load(path=str(path), format=RdfFormat.RDF_XML, lenient=True)
    return store


def _extract_owl_class_entities(store: Store) -> list[str]:
    entities: set[str] = set()
    for quad in store.quads_for_pattern(None, RDF_TYPE, OWL_CLASS, None):
        if isinstance(quad.subject, NamedNode):
            entities.add(quad.subject.value)
    return sorted(entities)


def _build_labels(store: Store, entity_ids: list[str]) -> list[str]:
    return [extract_entity_label(store, uri) for uri in entity_ids]


def _preprocess_labels(labels: list[str]) -> tuple[list[list[str]], list[str]]:
    tokenized = [preprocess_text(label) for label in labels]
    joined = [" ".join(tokens) for tokens in tokenized]
    return tokenized, joined


def _build_csv_columns(evaluation_ks: list[int]) -> list[str]:
    return [
        *FIXED_CSV_COLUMNS[:8],
        *(f"recall_at_{k}" for k in evaluation_ks),
        *FIXED_CSV_COLUMNS[8:],
        *EXTRA_CSV_COLUMNS,
        *(f"any_gold_recall_at_{k}" for k in evaluation_ks),
        *TRAILING_CSV_COLUMNS,
    ]


def _persist_results_to_csv(
    results: list[ExperimentResultRecord], output_csv_path: str | Path, evaluation_ks: list[int]
) -> None:
    csv_columns = _build_csv_columns(evaluation_ks)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        for record in results:
            row: dict[str, object] = {
                "track": record["track"],
                "version": record["version"],
                "dataset": record["dataset_name"],
                "method": record["model"],
                "hyperparameters": json.dumps(
                    record["hyperparameters"], sort_keys=True, separators=(",", ":")
                ),
                "gold_count": record["gold_count"],
                "candidate_size": record["candidate_size"],
                "dataset_prep_seconds": record["dataset_prep_seconds"],
                "mrr": record["mrr"],
                "runtime_seconds": record["runtime_seconds"],
                "query_count": record["query_count"],
                "multi_target_query_count": record["multi_target_query_count"],
                "mean_candidate_count": record["mean_candidate_count"],
                "any_gold_mrr": record["any_gold_mrr"],
                "rank_cutoff": record["rank_cutoff"],
                "candidate_policy": record["candidate_policy"],
                "alignment_policy": record["alignment_policy"],
                "zero_candidate_query_share": record["zero_candidate_query_share"],
            }
            for k in evaluation_ks:
                row[f"recall_at_{k}"] = record["recalls"][k]
                row[f"any_gold_recall_at_{k}"] = record["any_gold_recalls"][k]
            writer.writerow(row)


def _persist_errors_to_csv(errors: list[ExperimentErrorRecord], output_csv_path: Path) -> None:
    """Write every failed dataset or model run, so failed grid cells are explicit."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(ERROR_CSV_COLUMNS))
        writer.writeheader()
        for record in errors:
            writer.writerow(
                {
                    "scope": record["scope"],
                    "dataset": record["dataset_name"],
                    "method": record["model"],
                    "hyperparameters": json.dumps(
                        record["hyperparameters"], sort_keys=True, separators=(",", ":")
                    ),
                    "error": record["error"],
                }
            )


def _get_git_short_sha() -> str:
    """Backward-compatible wrapper around shared git metadata helper."""
    return get_git_short_sha(repo_root=Path(__file__).resolve().parents[2])


def _default_output_csv_path() -> Path:
    return build_timestamped_results_path("result")


def _build_tfidf_model_run(
    params: TfidfGridEntry, candidate_policy: str = "nonzero"
) -> ModelRunSpec:
    return ModelRunSpec(
        model_name="tfidf",
        hyperparameters={
            "ngram_range": params.ngram_range,
            "min_df": params.min_df,
            "max_df": params.max_df,
            "sublinear_tf": params.sublinear_tf,
        },
        retriever=TfidfRetriever(
            ngram_range=params.ngram_range,
            min_df=params.min_df,
            max_df=params.max_df,
            sublinear_tf=params.sublinear_tf,
            candidate_policy=candidate_policy,
        ),
    )


def _build_bm25_model_run(
    params: Bm25GridEntry, candidate_policy: str = "nonzero"
) -> ModelRunSpec:
    return ModelRunSpec(
        model_name="bm25",
        hyperparameters={"k1": params.k1, "b": params.b},
        retriever=Bm25Retriever(k1=params.k1, b=params.b, candidate_policy=candidate_policy),
    )


def _build_exact_match_model_run() -> ModelRunSpec:
    return ModelRunSpec(
        model_name="exact_match",
        hyperparameters=fixed_method_hyperparameters("exact_match"),
        retriever=ExactMatchRetriever(),
    )


def _build_char_ngram_model_run(candidate_policy: str = "nonzero") -> ModelRunSpec:
    return ModelRunSpec(
        model_name="char_ngram",
        hyperparameters=fixed_method_hyperparameters("char_ngram"),
        retriever=CharNgramRetriever(candidate_policy=candidate_policy),
    )


def _build_char_ngram_sensitivity_run(
    params: CharNgramGridEntry, candidate_policy: str = "nonzero"
) -> ModelRunSpec:
    return ModelRunSpec(
        model_name=CHAR_NGRAM_SENSITIVITY_METHOD,
        hyperparameters={
            "normalization": fixed_method_hyperparameters("char_ngram")["normalization"],
            "analyzer": params.analyzer,
            "ngram_range": list(params.ngram_range),
            "min_df": params.min_df,
            "sublinear_tf": params.sublinear_tf,
        },
        retriever=CharNgramRetriever(
            analyzer=params.analyzer,  # type: ignore[arg-type]
            ngram_range=params.ngram_range,
            min_df=params.min_df,
            sublinear_tf=params.sublinear_tf,
            candidate_policy=candidate_policy,  # type: ignore[arg-type]
        ),
    )


def _build_dense_model_run(entry: FixedBaselineEntry, candidate_policy: str = "nonzero") -> ModelRunSpec:
    """A pinned dense or hybrid baseline (one configuration, never tuned)."""
    return ModelRunSpec(
        model_name=entry.name,
        hyperparameters=entry.hyperparameters(),
        retriever=build_fixed_baseline(entry, candidate_policy),
    )


def _build_model_runs(
    tfidf_grid: list[TfidfGridEntry],
    bm25_grid: list[Bm25GridEntry],
    candidate_policy: str = "nonzero",
    char_ngram_grid: tuple[CharNgramGridEntry, ...] = (),
    dense_models: tuple[FixedBaselineEntry, ...] | list[FixedBaselineEntry] = (),
) -> list[ModelRunSpec]:
    model_runs: list[ModelRunSpec] = []
    for method_name in development_method_names():
        if method_name == "tfidf":
            for params in tfidf_grid:
                model_runs.append(_build_tfidf_model_run(params, candidate_policy))
            continue
        if method_name == "bm25":
            for params in bm25_grid:
                model_runs.append(_build_bm25_model_run(params, candidate_policy))
            continue
        if method_name == "exact_match":
            model_runs.append(_build_exact_match_model_run())
            continue
        if method_name == "char_ngram":
            model_runs.append(_build_char_ngram_model_run(candidate_policy))
            continue
        raise ValueError(f"Unsupported registered development method: {method_name}")
    # Sensitivity rows use their own method name so the zero-configuration char_ngram
    # default stays a single row per dataset in every downstream analysis.
    for params in char_ngram_grid:
        model_runs.append(_build_char_ngram_sensitivity_run(params, candidate_policy))
    for entry in dense_models:
        model_runs.append(_build_dense_model_run(entry, candidate_policy))
    return model_runs


def _format_recall_for_log(recalls: dict[int, float], evaluation_ks: list[int]) -> tuple[str, float]:
    recall_k = 10 if 10 in recalls else evaluation_ks[0]
    return f"recall@{recall_k}", recalls[recall_k]


def run_experiments(
    config_path: str | Path = "config/runtime.yaml",
    output_csv_path: str | Path | None = None,
    show_progress: bool | None = None,
) -> list[ExperimentResultRecord]:
    """Run retrieval experiments over datasets and hyperparameter grids."""
    validate_nltk_assets()
    progress_enabled = sys.stderr.isatty() if show_progress is None else show_progress
    resolved_output_csv_path = (
        Path(output_csv_path) if output_csv_path is not None else _default_output_csv_path()
    )

    runtime_config = load_runtime_config(config_path=config_path)
    datasets = runtime_config.development_datasets
    evaluation_ks = runtime_config.experiments.evaluation_ks
    k_max = max(evaluation_ks)
    candidate_policy = runtime_config.experiments.candidate_policy
    alignment_policy = runtime_config.experiments.alignment_policy
    model_runs = _build_model_runs(
        runtime_config.experiments.tfidf_grid,
        runtime_config.experiments.bm25_grid,
        candidate_policy,
        runtime_config.experiments.char_ngram_grid,
        runtime_config.experiments.fixed_baselines_for("development"),
    )

    logger.info(
        "Starting experiment run with config: %s (output_csv_path=%s, progress=%s)",
        config_path,
        resolved_output_csv_path,
        progress_enabled,
    )
    logger.info(
        "Loaded %d dataset(s) from config with evaluation_ks=%s, tfidf_grid=%d, bm25_grid=%d",
        len(datasets),
        evaluation_ks,
        len(runtime_config.experiments.tfidf_grid),
        len(runtime_config.experiments.bm25_grid),
    )

    results: list[ExperimentResultRecord] = []
    query_results: list[QueryRecord] = []
    errors: list[ExperimentErrorRecord] = []
    datasets_processed = 0
    successful_model_runs = 0
    failed_model_runs = 0
    dataset_failures = 0

    # Preserve YAML-defined dataset order for deterministic, config-driven execution.
    dataset_names = list(datasets.keys())
    dataset_iterator = tqdm(
        dataset_names,
        desc="Datasets",
        disable=not progress_enabled,
    )
    for dataset_name in dataset_iterator:
        dataset = datasets[dataset_name]
        datasets_processed += 1
        try:
            dataset_prep_start = time.perf_counter()
            logger.info(
                "Processing dataset '%s' (track=%s, version=%s)",
                dataset_name,
                dataset.track,
                dataset.version,
            )
            source_store = _load_store_with_fallback(dataset.source_rdf)
            target_store = _load_store_with_fallback(dataset.target_rdf)

            source_entities = _extract_owl_class_entities(source_store)
            target_entities = _extract_owl_class_entities(target_store)

            if not source_entities or not target_entities:
                logger.warning(
                    "Skipping dataset %s: no owl:Class entities found (source=%d, target=%d)",
                    dataset_name,
                    len(source_entities),
                    len(target_entities),
                )
                continue

            logger.info(
                "Extracted owl:Class entities (source=%d, target=%d)",
                len(source_entities),
                len(target_entities),
            )

            gold_raw = load_alignment_targets(dataset.alignment_rdf, alignment_policy)
            gold_filtered = filter_alignment_targets(
                gold_raw, set(source_entities), set(target_entities)
            )

            if not gold_filtered:
                logger.warning(
                    "Skipping dataset %s: no filtered gold mappings remain", dataset_name
                )
                continue

            logger.info(
                "Filtered gold mappings for dataset '%s': %d pair(s)",
                dataset_name,
                len(gold_filtered),
            )
            logger.info(
                "Dataset '%s' evaluation setup: source_entities=%d target_entities=%d eval_sources=%d",
                dataset_name,
                len(source_entities),
                len(target_entities),
                len(gold_filtered),
            )
            target_labels = _build_labels(target_store, target_entities)
            eval_sources = sorted(gold_filtered.keys())
            source_labels = _build_labels(source_store, eval_sources)
            source_label_map = dict(zip(eval_sources, source_labels, strict=True))
            target_label_map = dict(zip(target_entities, target_labels, strict=True))

            target_tokens, target_labels_preprocessed = _preprocess_labels(target_labels)
            source_tokens, source_labels_preprocessed = _preprocess_labels(source_labels)
            source_label_preprocessed_map = dict(
                zip(eval_sources, source_labels_preprocessed, strict=True)
            )
            source_token_map = dict(zip(eval_sources, source_tokens, strict=True))
            candidate_size = min(k_max, len(target_entities))
            gold_count = sum(len(targets) for targets in gold_filtered.values())
            multi_target_query_count = multi_target_source_count(gold_filtered)
            dataset_prep_seconds = time.perf_counter() - dataset_prep_start

            logger.info(
                "Dataset '%s' shared preprocessing completed in %.4fs (gold_count=%d, candidate_size=%d)",
                dataset_name,
                dataset_prep_seconds,
                gold_count,
                candidate_size,
            )

            model_iterator = tqdm(
                model_runs,
                desc=f"Models:{dataset_name}",
                disable=not progress_enabled,
            )
            for run in model_iterator:
                try:
                    run_start = time.perf_counter()
                    logger.info(
                        "Running model=%s with hyperparameters=%s on dataset='%s'",
                        run.model_name,
                        run.hyperparameters,
                        dataset_name,
                    )
                    if run.model_name == "tfidf":
                        tfidf_retriever = cast(TfidfRetriever, run.retriever)
                        tfidf_retriever.fit_preprocessed(
                            target_entities, target_labels_preprocessed
                        )
                    elif run.model_name == "bm25":
                        bm25_retriever = cast(Bm25Retriever, run.retriever)
                        bm25_retriever.fit_tokenized(target_entities, target_tokens)
                    elif run.model_name == "exact_match":
                        exact_match_retriever = cast(ExactMatchRetriever, run.retriever)
                        exact_match_retriever.fit(target_entities, target_labels)
                    elif run.model_name in ("char_ngram", CHAR_NGRAM_SENSITIVITY_METHOD):
                        char_ngram_retriever = cast(CharNgramRetriever, run.retriever)
                        char_ngram_retriever.fit(target_entities, target_labels)
                    elif is_fixed_baseline(run.model_name):
                        run.retriever.fit(target_entities, target_labels)
                    else:
                        raise ValueError(f"Unsupported model type: {run.model_name}")

                    predictions: dict[str, list[tuple[str, float]]] = {}
                    source_iterator = tqdm(
                        eval_sources,
                        desc=f"Sources:{dataset_name}:{run.model_name}",
                        disable=not progress_enabled,
                        leave=False,
                    )
                    if is_fixed_baseline(run.model_name):
                        # Batch-encode all queries; ranking is identical to per-query calls.
                        ranked_lists = run.retriever.retrieve_many(  # type: ignore[attr-defined]
                            [source_label_map[source_id] for source_id in eval_sources], k=k_max
                        )
                        predictions = dict(zip(eval_sources, ranked_lists, strict=True))
                    else:
                        for source_id in source_iterator:
                            if run.model_name == "tfidf":
                                tfidf_retriever = cast(TfidfRetriever, run.retriever)
                                predictions[source_id] = tfidf_retriever.retrieve_preprocessed(
                                    source_label_preprocessed_map[source_id], k=k_max
                                )
                            elif run.model_name == "bm25":
                                bm25_retriever = cast(Bm25Retriever, run.retriever)
                                predictions[source_id] = bm25_retriever.retrieve_tokenized(
                                    source_token_map[source_id], k=k_max
                                )
                            elif run.model_name == "exact_match":
                                exact_match_retriever = cast(ExactMatchRetriever, run.retriever)
                                predictions[source_id] = exact_match_retriever.retrieve(
                                    source_label_map[source_id], k=k_max
                                )
                            else:
                                char_ngram_retriever = cast(CharNgramRetriever, run.retriever)
                                predictions[source_id] = char_ngram_retriever.retrieve(
                                    source_label_map[source_id], k=k_max
                                )

                    metrics = compute_retrieval_metrics(
                        predictions, gold_filtered, evaluation_ks
                    )
                    recalls = {k: metrics.recalls[k] for k in evaluation_ks}
                    mrr = metrics.mrr
                    result_record: ExperimentResultRecord = {
                        "dataset_name": dataset.name,
                        "track": dataset.track,
                        "version": dataset.version,
                        "model": run.model_name,
                        "hyperparameters": run.hyperparameters,
                        "num_source_entities": len(source_entities),
                        "num_target_entities": len(target_entities),
                        "num_gold_pairs": gold_count,
                        "gold_count": gold_count,
                        "candidate_size": candidate_size,
                        "dataset_prep_seconds": dataset_prep_seconds,
                        "recalls": recalls,
                        "mrr": mrr,
                        "runtime_seconds": time.perf_counter() - run_start,
                        "query_count": metrics.query_count,
                        "multi_target_query_count": multi_target_query_count,
                        "mean_candidate_count": mean_candidate_count(
                            predictions, eval_sources
                        ),
                        "any_gold_recalls": {
                            k: metrics.any_gold_recalls[k] for k in evaluation_ks
                        },
                        "any_gold_mrr": metrics.any_gold_mrr,
                        "rank_cutoff": k_max,
                        "candidate_policy": candidate_policy,
                        "alignment_policy": alignment_policy,
                        "zero_candidate_query_share": sum(
                            1 for source in eval_sources if not predictions.get(source)
                        )
                        / len(eval_sources),
                    }
                    results.append(result_record)
                    query_results.extend(
                        build_query_records(
                            track=dataset.track,
                            version=dataset.version,
                            dataset_name=dataset.name,
                            entity_type="class",
                            method_name=run.model_name,
                            hyperparameters=run.hyperparameters,
                            eval_sources=eval_sources,
                            gold=gold_filtered,
                            predictions=predictions,
                            source_label_map=source_label_map,
                            target_label_map=target_label_map,
                            k_max=k_max,
                        )
                    )
                    successful_model_runs += 1
                    recall_label, recall_value = _format_recall_for_log(recalls, evaluation_ks)
                    logger.info(
                        "Completed run model=%s dataset='%s' mrr=%.4f %s=%.4f runtime=%.4fs",
                        run.model_name,
                        dataset_name,
                        result_record["mrr"],
                        recall_label,
                        recall_value,
                        result_record["runtime_seconds"],
                    )
                except Exception as exc:
                    failed_model_runs += 1
                    errors.append(
                        ExperimentErrorRecord(
                            scope="model_run",
                            dataset_name=dataset_name,
                            model=run.model_name,
                            hyperparameters=run.hyperparameters,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    logger.exception(
                        "Model run failed for dataset='%s', model=%s, hyperparameters=%s",
                        dataset_name,
                        run.model_name,
                        run.hyperparameters,
                    )
                    continue
            logger.info("Finished dataset '%s' model runs", dataset_name)
        except Exception as exc:
            dataset_failures += 1
            errors.append(
                ExperimentErrorRecord(
                    scope="dataset",
                    dataset_name=dataset_name,
                    model="",
                    hyperparameters={},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            logger.exception("Dataset processing failed for dataset='%s'", dataset_name)
            logger.info("Finished dataset '%s' with dataset-level failure", dataset_name)
            continue

    try:
        _persist_results_to_csv(
            results,
            output_csv_path=resolved_output_csv_path,
            evaluation_ks=evaluation_ks,
        )
        logger.info(
            "Persisted %d result record(s) to %s",
            len(results),
            resolved_output_csv_path,
        )
        query_csv_path = query_output_csv_path(resolved_output_csv_path)
        write_query_csv(query_results, query_csv_path)
        logger.info(
            "Persisted %d query-level record(s) to %s", len(query_results), query_csv_path
        )
    except Exception as exc:
        errors.append(
            ExperimentErrorRecord(
                scope="result_persistence",
                dataset_name="",
                model="",
                hyperparameters={},
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        logger.exception(
            "Failed to persist result records to CSV at %s; returning in-memory results",
            resolved_output_csv_path,
        )

    errors_csv_path = resolved_output_csv_path.with_name(
        f"{resolved_output_csv_path.stem}_errors.csv"
    )
    try:
        _persist_errors_to_csv(errors, errors_csv_path)
    except OSError:
        logger.exception("Failed to persist error records to %s", errors_csv_path)

    if errors:
        logger.warning("Experiment run completed with %d error record(s)", len(errors))
        for index, record in enumerate(errors, start=1):
            logger.warning("error_record_%d=%s", index, record)
    if not results and errors:
        logger.warning(
            "Best-effort execution completed with zero successful runs; all runs failed."
        )
    logger.info(
        "Run summary: datasets_processed=%d dataset_failures=%d successful_model_runs=%d failed_model_runs=%d result_rows=%d error_records=%d",
        datasets_processed,
        dataset_failures,
        successful_model_runs,
        failed_model_runs,
        len(results),
        len(errors),
    )
    logger.info("Experiment run finished with %d result record(s)", len(results))
    return results
