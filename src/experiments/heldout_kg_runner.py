"""Held-out KG experiment runner with typed partitions and frozen settings."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import TypedDict, cast

from tqdm import tqdm

from src.config_loader import load_runtime_config
from src.evaluation.metrics import compute_recall_at_ks_and_mrr
from src.experiments.experiment_runner import _load_store_with_fallback
from src.method_registry import (
    fixed_method_hyperparameters,
    fixed_method_names,
    heldout_selected_method_names,
    ordered_method_names,
)
from src.preprocessing.text_preprocessor import preprocess_text, validate_nltk_assets
from src.rdf_utils.alignment_parser import load_alignment_mappings
from src.rdf_utils.entity_partitioning import (
    EntityType,
    TypedAlignmentPartitions,
    TypedEntityPartitions,
    extract_typed_entity_partitions,
    partition_alignment_mappings_by_entity_type,
)
from src.rdf_utils.label_extractor import extract_entity_label
from src.retrieval.bm25_retriever import Bm25Retriever
from src.retrieval.exact_match_retriever import ExactMatchRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever

logger = logging.getLogger(__name__)

FIXED_CSV_COLUMNS: tuple[str, ...] = (
    "track",
    "version",
    "dataset",
    "entity_type",
    "method",
    "hyperparameters",
    "gold_count",
    "target_pool_size",
    "retained_candidate_size",
    "candidate_reduction_ratio",
    "mrr",
    "runtime_seconds",
)


class HeldoutKgRunnerValidationError(ValueError):
    """Raised when held-out KG execution inputs are invalid."""


class HeldoutKgResultRecord(TypedDict):
    dataset_name: str
    track: str
    version: str
    entity_type: str
    model: str
    hyperparameters: dict[str, object]
    gold_count: int
    target_pool_size: int
    retained_candidate_size: int
    candidate_reduction_ratio: float
    recalls: dict[int, float]
    mrr: float
    runtime_seconds: float


@dataclass(frozen=True)
class HeldoutMethodRunSpec:
    """Resolved held-out method execution configuration."""

    method_name: str
    hyperparameters: dict[str, object]


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
    return Path("results") / f"heldout_result_{timestamp}_{git_sha}.csv"


def _resolve_selected_settings_json(
    selected_settings_path: str | Path | None,
) -> Path:
    if selected_settings_path is not None:
        path = Path(selected_settings_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Selected settings JSON not found: {path}")
        return path

    candidates = sorted(
        Path("results/comparisons").glob("**/heldout_selected_settings.json"),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "No results/comparisons/**/heldout_selected_settings.json files found."
        )
    return candidates[-1].resolve()


def _load_selected_settings(selected_settings_path: Path) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(selected_settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HeldoutKgRunnerValidationError(
            f"Malformed selected settings JSON: {selected_settings_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise HeldoutKgRunnerValidationError(
            f"Selected settings payload must be a JSON object: {selected_settings_path}"
        )
    selected_settings = payload.get("selected_settings")
    if not isinstance(selected_settings, dict):
        raise HeldoutKgRunnerValidationError(
            "Selected settings JSON must contain a 'selected_settings' object."
        )

    required_hyperparameter_keys = {
        "tfidf": ("ngram_range", "min_df", "max_df", "sublinear_tf"),
        "bm25": ("k1", "b"),
    }

    resolved: dict[str, dict[str, object]] = {}
    for method in heldout_selected_method_names():
        entry = selected_settings.get(method)
        if not isinstance(entry, dict):
            raise HeldoutKgRunnerValidationError(
                f"Selected settings JSON missing required method '{method}'."
            )
        hyperparameters = entry.get("hyperparameters")
        if not isinstance(hyperparameters, dict):
            raise HeldoutKgRunnerValidationError(
                f"Selected settings for method '{method}' must include a hyperparameters object."
            )
        missing = [
            key for key in required_hyperparameter_keys[method] if key not in hyperparameters
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise HeldoutKgRunnerValidationError(
                f"Selected settings for method '{method}' missing required hyperparameter keys: {missing_str}."
            )
        resolved[method] = hyperparameters

    logger.info(
        "Loaded frozen held-out settings from %s (tfidf=%s, bm25=%s)",
        selected_settings_path,
        json.dumps(resolved["tfidf"], sort_keys=True, separators=(",", ":")),
        json.dumps(resolved["bm25"], sort_keys=True, separators=(",", ":")),
    )
    return resolved


def _build_heldout_method_runs(
    frozen_settings: dict[str, dict[str, object]],
) -> list[HeldoutMethodRunSpec]:
    method_runs: list[HeldoutMethodRunSpec] = []
    runtime_methods = list(frozen_settings.keys()) + fixed_method_names(heldout=True)
    for method_name in ordered_method_names(runtime_methods):
        if method_name in frozen_settings:
            hyperparameters = frozen_settings[method_name]
        else:
            hyperparameters = fixed_method_hyperparameters(method_name)
        method_runs.append(
            HeldoutMethodRunSpec(
                method_name=method_name,
                hyperparameters=hyperparameters,
            )
        )
    return method_runs


def _coerce_tfidf_ngram_range(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise HeldoutKgRunnerValidationError(
            "Selected settings for method 'tfidf' must include a two-item ngram_range list."
        )
    first, second = value
    if not isinstance(first, int) or isinstance(first, bool):
        raise HeldoutKgRunnerValidationError(
            "Selected settings for method 'tfidf' must include an integer ngram_range[0]."
        )
    if not isinstance(second, int) or isinstance(second, bool):
        raise HeldoutKgRunnerValidationError(
            "Selected settings for method 'tfidf' must include an integer ngram_range[1]."
        )
    return (first, second)


def _predict_for_method(
    *,
    method_name: str,
    hyperparameters: dict[str, object],
    target_entities: list[str],
    target_labels: list[str],
    target_labels_preprocessed: list[str],
    target_tokens: list[list[str]],
    source_label_map: dict[str, str],
    source_label_preprocessed_map: dict[str, str],
    source_token_map: dict[str, list[str]],
    eval_sources: list[str],
    k_max: int,
) -> dict[str, list[tuple[str, float]]]:
    if method_name == "tfidf":
        retriever = TfidfRetriever(
            ngram_range=_coerce_tfidf_ngram_range(hyperparameters["ngram_range"]),
            min_df=cast(int | float, hyperparameters["min_df"]),
            max_df=cast(int | float, hyperparameters["max_df"]),
            sublinear_tf=cast(bool, hyperparameters["sublinear_tf"]),
        )
        retriever.fit_preprocessed(target_entities, target_labels_preprocessed)
        return {
            source_id: retriever.retrieve_preprocessed(
                source_label_preprocessed_map[source_id],
                k=k_max,
            )
            for source_id in eval_sources
        }

    if method_name == "bm25":
        retriever = Bm25Retriever(
            k1=float(cast(int | float, hyperparameters["k1"])),
            b=float(cast(int | float, hyperparameters["b"])),
        )
        retriever.fit_tokenized(target_entities, target_tokens)
        return {
            source_id: retriever.retrieve_tokenized(
                source_token_map[source_id],
                k=k_max,
            )
            for source_id in eval_sources
        }

    if method_name == "exact_match":
        retriever = ExactMatchRetriever()
        retriever.fit(target_entities, target_labels)
        return {
            source_id: retriever.retrieve(
                source_label_map[source_id],
                k=k_max,
            )
            for source_id in eval_sources
        }

    raise HeldoutKgRunnerValidationError(
        f"Unsupported registered held-out method '{method_name}'."
    )


def _build_labels(store, entity_ids: list[str]) -> list[str]:
    return [extract_entity_label(store, uri) for uri in entity_ids]


def _preprocess_labels(labels: list[str]) -> tuple[list[list[str]], list[str]]:
    tokenized = [preprocess_text(label) for label in labels]
    joined = [" ".join(tokens) for tokens in tokenized]
    return tokenized, joined


def _build_csv_columns(evaluation_ks: list[int]) -> list[str]:
    return [
        *FIXED_CSV_COLUMNS[:10],
        *(f"recall_at_{k}" for k in evaluation_ks),
        *FIXED_CSV_COLUMNS[10:],
    ]


def _persist_results_to_csv(
    results: list[HeldoutKgResultRecord],
    output_csv_path: str | Path,
    evaluation_ks: list[int],
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
                "entity_type": record["entity_type"],
                "method": record["model"],
                "hyperparameters": json.dumps(
                    record["hyperparameters"], sort_keys=True, separators=(",", ":")
                ),
                "gold_count": record["gold_count"],
                "target_pool_size": record["target_pool_size"],
                "retained_candidate_size": record["retained_candidate_size"],
                "candidate_reduction_ratio": record["candidate_reduction_ratio"],
                "mrr": record["mrr"],
                "runtime_seconds": record["runtime_seconds"],
            }
            for k in evaluation_ks:
                row[f"recall_at_{k}"] = record["recalls"][k]
            writer.writerow(row)


def _entity_values(partitions: TypedEntityPartitions, entity_type: EntityType) -> tuple[str, ...]:
    if entity_type is EntityType.CLASS:
        return partitions.classes
    if entity_type is EntityType.PREDICATE:
        return partitions.predicates
    return partitions.instances


def _gold_values(
    partitions: TypedAlignmentPartitions,
    entity_type: EntityType,
) -> dict[str, str]:
    if entity_type is EntityType.CLASS:
        return partitions.classes
    if entity_type is EntityType.PREDICATE:
        return partitions.predicates
    return partitions.instances


def _validate_type_partition(
    *,
    dataset_name: str,
    entity_type: EntityType,
    target_entities: tuple[str, ...],
    gold: dict[str, str],
) -> None:
    if not target_entities:
        raise HeldoutKgRunnerValidationError(
            f"Held-out dataset '{dataset_name}' has zero target entities for entity type '{entity_type.value}'."
        )
    if not gold:
        raise HeldoutKgRunnerValidationError(
            f"Held-out dataset '{dataset_name}' has zero gold mappings for entity type '{entity_type.value}'."
        )


def run_heldout_kg_experiments(
    config_path: str | Path = "config/runtime.yaml",
    selected_settings_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    show_progress: bool | None = None,
) -> list[HeldoutKgResultRecord]:
    """Run frozen TF-IDF and BM25 held-out KG evaluation by entity type."""
    validate_nltk_assets()
    progress_enabled = sys.stderr.isatty() if show_progress is None else show_progress
    resolved_output_csv_path = (
        Path(output_csv_path) if output_csv_path is not None else _default_output_csv_path()
    )
    runtime_config = load_runtime_config(config_path=config_path, require_heldout=True)
    selected_settings_json = _resolve_selected_settings_json(selected_settings_path)
    frozen_settings = _load_selected_settings(selected_settings_json)
    method_runs = _build_heldout_method_runs(frozen_settings)
    datasets = runtime_config.heldout_datasets
    if not datasets:
        raise HeldoutKgRunnerValidationError(
            "Held-out KG execution requires at least one heldout_datasets entry."
        )

    evaluation_ks = runtime_config.experiments.evaluation_ks
    k_max = max(evaluation_ks)
    logger.info(
        "Starting held-out KG run with config=%s selected_settings=%s output_csv_path=%s progress=%s",
        config_path,
        selected_settings_json,
        resolved_output_csv_path,
        progress_enabled,
    )

    results: list[HeldoutKgResultRecord] = []
    dataset_names = list(datasets.keys())
    dataset_iterator = tqdm(
        dataset_names,
        desc="HeldoutDatasets",
        disable=not progress_enabled,
    )

    for dataset_name in dataset_iterator:
        dataset = datasets[dataset_name]
        logger.info(
            "Processing held-out dataset '%s' (track=%s, version=%s)",
            dataset_name,
            dataset.track,
            dataset.version,
        )
        source_store = _load_store_with_fallback(dataset.source_rdf)
        target_store = _load_store_with_fallback(dataset.target_rdf)
        gold_raw = load_alignment_mappings(dataset.alignment_rdf)
        source_partitions = extract_typed_entity_partitions(
            source_store,
            graph_name=f"{dataset_name}:source",
        )
        target_partitions = extract_typed_entity_partitions(
            target_store,
            graph_name=f"{dataset_name}:target",
        )
        gold_partitions = partition_alignment_mappings_by_entity_type(
            gold_raw,
            source_partitions,
            target_partitions,
            alignment_name=dataset_name,
        )

        for entity_type in (EntityType.CLASS, EntityType.PREDICATE, EntityType.INSTANCE):
            target_entities = list(_entity_values(target_partitions, entity_type))
            gold = _gold_values(gold_partitions, entity_type)
            _validate_type_partition(
                dataset_name=dataset_name,
                entity_type=entity_type,
                target_entities=tuple(target_entities),
                gold=gold,
            )

            eval_sources = sorted(gold.keys())
            target_labels = _build_labels(target_store, target_entities)
            source_labels = _build_labels(source_store, eval_sources)
            source_label_map = dict(zip(eval_sources, source_labels, strict=True))
            target_tokens, target_labels_preprocessed = _preprocess_labels(target_labels)
            source_tokens, source_labels_preprocessed = _preprocess_labels(source_labels)
            source_label_preprocessed_map = dict(
                zip(eval_sources, source_labels_preprocessed, strict=True)
            )
            source_token_map = dict(zip(eval_sources, source_tokens, strict=True))
            target_pool_size = len(target_entities)
            retained_candidate_size = min(k_max, target_pool_size)
            candidate_reduction_ratio = 1.0 - (
                retained_candidate_size / target_pool_size
            )

            logger.info(
                "Held-out dataset '%s' entity_type=%s setup: target_pool_size=%d gold_count=%d retained_candidate_size=%d reduction_ratio=%.6f",
                dataset_name,
                entity_type.value,
                target_pool_size,
                len(gold),
                retained_candidate_size,
                candidate_reduction_ratio,
            )

            for method_run in method_runs:
                method_name = method_run.method_name
                hyperparameters = method_run.hyperparameters
                run_start = time.perf_counter()
                logger.info(
                    "Running held-out dataset='%s' entity_type=%s method=%s hyperparameters=%s",
                    dataset_name,
                    entity_type.value,
                    method_name,
                    hyperparameters,
                )
                predictions = _predict_for_method(
                    method_name=method_name,
                    hyperparameters=hyperparameters,
                    target_entities=target_entities,
                    target_labels=target_labels,
                    target_labels_preprocessed=target_labels_preprocessed,
                    target_tokens=target_tokens,
                    source_label_map=source_label_map,
                    source_label_preprocessed_map=source_label_preprocessed_map,
                    source_token_map=source_token_map,
                    eval_sources=eval_sources,
                    k_max=k_max,
                )

                raw_recalls, mrr = compute_recall_at_ks_and_mrr(
                    predictions, gold, evaluation_ks
                )
                recalls = {k: raw_recalls[k] for k in evaluation_ks}
                results.append(
                    HeldoutKgResultRecord(
                        dataset_name=dataset.name,
                        track=dataset.track,
                        version=dataset.version,
                        entity_type=entity_type.value,
                        model=method_name,
                        hyperparameters=hyperparameters,
                        gold_count=len(gold),
                        target_pool_size=target_pool_size,
                        retained_candidate_size=retained_candidate_size,
                        candidate_reduction_ratio=candidate_reduction_ratio,
                        recalls=recalls,
                        mrr=mrr,
                        runtime_seconds=time.perf_counter() - run_start,
                    )
                )
                logger.info(
                    "Completed held-out run dataset='%s' entity_type=%s method=%s mrr=%.4f runtime=%.4fs",
                    dataset_name,
                    entity_type.value,
                    method_name,
                    mrr,
                    results[-1]["runtime_seconds"],
                )
        logger.info("Finished held-out dataset '%s'", dataset_name)

    _persist_results_to_csv(
        results,
        output_csv_path=resolved_output_csv_path,
        evaluation_ks=evaluation_ks,
    )
    logger.info(
        "Persisted %d held-out KG result record(s) to %s",
        len(results),
        resolved_output_csv_path,
    )
    logger.info("Held-out KG run finished with %d result record(s)", len(results))
    return results
