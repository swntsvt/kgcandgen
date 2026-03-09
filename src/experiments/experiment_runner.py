"""Experiment runner for candidate generation retrieval studies."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from typing import Protocol, TypedDict

from pyoxigraph import NamedNode, RdfFormat, Store

from src.config_loader import load_datasets_config
from src.evaluation.metrics import compute_recall_at_ks_and_mrr
from src.rdf_utils.alignment_parser import load_alignment_mappings
from src.rdf_utils.label_extractor import extract_entity_label
from src.retrieval.bm25_retriever import Bm25Retriever
from src.retrieval.tfidf_retriever import TfidfRetriever

RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
OWL_CLASS = NamedNode("http://www.w3.org/2002/07/owl#Class")
EVAL_KS: tuple[int, ...] = (1, 5, 10, 20, 50)
CSV_COLUMNS: tuple[str, ...] = (
    "track",
    "version",
    "dataset",
    "method",
    "hyperparameters",
    "candidate_size",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "recall_at_50",
    "mrr",
    "runtime_seconds",
)

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
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    recall_at_50: float
    mrr: float
    runtime_seconds: float


class ExperimentErrorRecord(TypedDict):
    scope: str
    dataset_name: str
    model: str
    hyperparameters: dict[str, object]
    error: str


class TfidfHyperparameters(TypedDict):
    ngram_range: tuple[int, int]
    min_df: int | float
    max_df: int | float
    sublinear_tf: bool


class Bm25Hyperparameters(TypedDict):
    k1: float
    b: float


@dataclass(frozen=True)
class ModelRunSpec:
    model_name: str
    hyperparameters: dict[str, object]
    retriever: RetrieverProtocol


TFIDF_GRID: tuple[TfidfHyperparameters, ...] = (
    {"ngram_range": (1, 1), "min_df": 1, "max_df": 1.0, "sublinear_tf": False},
    {"ngram_range": (1, 2), "min_df": 1, "max_df": 1.0, "sublinear_tf": True},
)
BM25_GRID: tuple[Bm25Hyperparameters, ...] = (
    {"k1": 1.5, "b": 0.75},
    {"k1": 1.2, "b": 0.75},
)


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


def _persist_results_to_csv(
    results: list[ExperimentResultRecord], output_csv_path: str | Path
) -> None:
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in results:
            writer.writerow(
                {
                    "track": record["track"],
                    "version": record["version"],
                    "dataset": record["dataset_name"],
                    "method": record["model"],
                    "hyperparameters": json.dumps(
                        record["hyperparameters"], sort_keys=True, separators=(",", ":")
                    ),
                    "candidate_size": max(EVAL_KS),
                    "recall_at_1": record["recall_at_1"],
                    "recall_at_5": record["recall_at_5"],
                    "recall_at_10": record["recall_at_10"],
                    "recall_at_20": record["recall_at_20"],
                    "recall_at_50": record["recall_at_50"],
                    "mrr": record["mrr"],
                    "runtime_seconds": record["runtime_seconds"],
                }
            )


def _get_git_short_sha() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "nogit"


def _default_output_csv_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = _get_git_short_sha()
    return Path("results") / f"result_{timestamp}_{git_sha}.csv"


def run_experiments(
    config_path: str | Path = "config/datasets.yaml",
    output_csv_path: str | Path | None = None,
) -> list[ExperimentResultRecord]:
    """Run retrieval experiments over datasets and hyperparameter grids."""
    resolved_output_csv_path = (
        Path(output_csv_path) if output_csv_path is not None else _default_output_csv_path()
    )
    logger.info(
        "Starting experiment run with config: %s (output_csv_path=%s)",
        config_path,
        resolved_output_csv_path,
    )
    datasets = load_datasets_config(config_path=config_path)
    logger.info("Loaded %d dataset(s) from config", len(datasets))
    results: list[ExperimentResultRecord] = []
    errors: list[ExperimentErrorRecord] = []
    k_max = max(EVAL_KS)

    for dataset_name in sorted(datasets.keys()):
        dataset = datasets[dataset_name]
        try:
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

            gold_raw = load_alignment_mappings(dataset.alignment_rdf)
            target_set = set(target_entities)
            source_set = set(source_entities)
            gold_filtered = {
                source: target
                for source, target in gold_raw.items()
                if source in source_set and target in target_set
            }

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
            target_labels = _build_labels(target_store, target_entities)
            eval_sources = sorted(gold_filtered.keys())
            source_labels = _build_labels(source_store, eval_sources)
            source_label_map = dict(zip(eval_sources, source_labels, strict=True))
            model_runs: list[ModelRunSpec] = []
            for params in TFIDF_GRID:
                model_runs.append(
                    ModelRunSpec(
                        "tfidf",
                        dict(params),
                        TfidfRetriever(
                            ngram_range=params["ngram_range"],
                            min_df=params["min_df"],
                            max_df=params["max_df"],
                            sublinear_tf=params["sublinear_tf"],
                        ),
                    )
                )
            for params in BM25_GRID:
                model_runs.append(
                    ModelRunSpec(
                        "bm25",
                        dict(params),
                        Bm25Retriever(k1=params["k1"], b=params["b"]),
                    )
                )

            for run in model_runs:
                try:
                    run_start = time.perf_counter()
                    logger.info(
                        "Running model=%s with hyperparameters=%s on dataset='%s'",
                        run.model_name,
                        run.hyperparameters,
                        dataset_name,
                    )
                    run.retriever.fit(target_entities, target_labels)
                    predictions: dict[str, list[tuple[str, float]]] = {}
                    for source_id in eval_sources:
                        predictions[source_id] = run.retriever.retrieve(
                            source_label_map[source_id], k=k_max
                        )

                    recalls, mrr = compute_recall_at_ks_and_mrr(
                        predictions, gold_filtered, EVAL_KS
                    )
                    result_record: ExperimentResultRecord = {
                        "dataset_name": dataset.name,
                        "track": dataset.track,
                        "version": dataset.version,
                        "model": run.model_name,
                        "hyperparameters": run.hyperparameters,
                        "num_source_entities": len(source_entities),
                        "num_target_entities": len(target_entities),
                        "num_gold_pairs": len(gold_filtered),
                        "recall_at_1": recalls[1],
                        "recall_at_5": recalls[5],
                        "recall_at_10": recalls[10],
                        "recall_at_20": recalls[20],
                        "recall_at_50": recalls[50],
                        "mrr": mrr,
                        "runtime_seconds": time.perf_counter() - run_start,
                    }
                    results.append(result_record)
                    logger.info(
                        "Completed run model=%s dataset='%s' mrr=%.4f recall@10=%.4f runtime=%.4fs",
                        run.model_name,
                        dataset_name,
                        result_record["mrr"],
                        result_record["recall_at_10"],
                        result_record["runtime_seconds"],
                    )
                except Exception as exc:
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
        except Exception as exc:
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
            continue

    try:
        _persist_results_to_csv(results, output_csv_path=resolved_output_csv_path)
        logger.info(
            "Persisted %d result record(s) to %s",
            len(results),
            resolved_output_csv_path,
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

    if errors:
        logger.warning("Experiment run completed with %d error record(s)", len(errors))
        for index, record in enumerate(errors, start=1):
            logger.warning("error_record_%d=%s", index, record)
    if not results and errors:
        logger.warning(
            "Best-effort execution completed with zero successful runs; all runs failed."
        )
    logger.info("Experiment run finished with %d result record(s)", len(results))
    return results
