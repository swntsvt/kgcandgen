"""Secondary held-out class-alignment runner."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import TypedDict, cast

from pyoxigraph import NamedNode
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
from src.rdf_utils.label_extractor import extract_entity_label
from src.retrieval.bm25_retriever import Bm25Retriever
from src.retrieval.char_ngram_retriever import CharNgramRetriever
from src.retrieval.exact_match_retriever import ExactMatchRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.run_metadata import build_timestamped_results_path, get_git_short_sha

logger = logging.getLogger(__name__)

RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
OWL_CLASS = NamedNode("http://www.w3.org/2002/07/owl#Class")

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


class HeldoutClassRunnerValidationError(ValueError):
    """Raised when held-out class-only execution inputs are invalid."""


class HeldoutClassResultRecord(TypedDict):
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
    method_name: str
    hyperparameters: dict[str, object]


def _get_git_short_sha() -> str:
    """Backward-compatible wrapper around shared git metadata helper."""
    return get_git_short_sha(repo_root=Path(__file__).resolve().parents[2])


def _default_output_csv_path() -> Path:
    return build_timestamped_results_path("heldout_class_result")


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
        raise HeldoutClassRunnerValidationError(
            f"Malformed selected settings JSON: {selected_settings_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise HeldoutClassRunnerValidationError(
            f"Selected settings payload must be a JSON object: {selected_settings_path}"
        )
    selected_settings = payload.get("selected_settings")
    if not isinstance(selected_settings, dict):
        raise HeldoutClassRunnerValidationError(
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
            raise HeldoutClassRunnerValidationError(
                f"Selected settings JSON missing required method '{method}'."
            )
        hyperparameters = entry.get("hyperparameters")
        if not isinstance(hyperparameters, dict):
            raise HeldoutClassRunnerValidationError(
                f"Selected settings for method '{method}' must include a hyperparameters object."
            )
        missing = [
            key for key in required_hyperparameter_keys[method] if key not in hyperparameters
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise HeldoutClassRunnerValidationError(
                f"Selected settings for method '{method}' missing required hyperparameter keys: {missing_str}."
            )
        resolved[method] = hyperparameters
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
        raise HeldoutClassRunnerValidationError(
            "Selected settings for method 'tfidf' must include a two-item ngram_range list."
        )
    first, second = value
    if not isinstance(first, int) or isinstance(first, bool):
        raise HeldoutClassRunnerValidationError(
            "Selected settings for method 'tfidf' must include an integer ngram_range[0]."
        )
    if not isinstance(second, int) or isinstance(second, bool):
        raise HeldoutClassRunnerValidationError(
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

    if method_name == "char_ngram":
        retriever = CharNgramRetriever()
        retriever.fit(target_entities, target_labels)
        return {
            source_id: retriever.retrieve(
                source_label_map[source_id],
                k=k_max,
            )
            for source_id in eval_sources
        }

    raise HeldoutClassRunnerValidationError(
        f"Unsupported registered held-out method '{method_name}'."
    )


def _build_labels(store, entity_ids: list[str]) -> list[str]:
    return [extract_entity_label(store, uri) for uri in entity_ids]


def _extract_owl_class_entities(store) -> list[str]:
    entities: set[str] = set()
    for quad in store.quads_for_pattern(None, RDF_TYPE, OWL_CLASS, None):
        if isinstance(quad.subject, NamedNode):
            entities.add(quad.subject.value)
    return sorted(entities)


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
    results: list[HeldoutClassResultRecord],
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


def run_heldout_class_experiments(
    config_path: str | Path = "config/runtime.yaml",
    selected_settings_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    show_progress: bool | None = None,
) -> list[HeldoutClassResultRecord]:
    """Run secondary held-out class-only evaluation."""
    validate_nltk_assets()
    progress_enabled = sys.stderr.isatty() if show_progress is None else show_progress
    resolved_output_csv_path = (
        Path(output_csv_path) if output_csv_path is not None else _default_output_csv_path()
    )
    runtime_config = load_runtime_config(config_path=config_path)
    selected_settings_json = _resolve_selected_settings_json(selected_settings_path)
    frozen_settings = _load_selected_settings(selected_settings_json)
    method_runs = _build_heldout_method_runs(frozen_settings)
    datasets = runtime_config.heldout_secondary_datasets
    if not datasets:
        raise HeldoutClassRunnerValidationError(
            "Held-out class-only execution requires at least one heldout_secondary_datasets entry."
        )

    evaluation_ks = runtime_config.experiments.evaluation_ks
    k_max = max(evaluation_ks)

    results: list[HeldoutClassResultRecord] = []
    dataset_names = list(datasets.keys())
    dataset_iterator = tqdm(
        dataset_names,
        desc="HeldoutClassDatasets",
        disable=not progress_enabled,
    )
    for dataset_name in dataset_iterator:
        dataset = datasets[dataset_name]
        source_store = _load_store_with_fallback(dataset.source_rdf)
        target_store = _load_store_with_fallback(dataset.target_rdf)
        gold_raw = load_alignment_mappings(dataset.alignment_rdf)
        source_entities = _extract_owl_class_entities(source_store)
        target_entities = _extract_owl_class_entities(target_store)
        class_source_set = set(source_entities)
        class_target_set = set(target_entities)
        gold = {
            source: target
            for source, target in gold_raw.items()
            if source in class_source_set and target in class_target_set
        }
        if not target_entities:
            raise HeldoutClassRunnerValidationError(
                f"Held-out secondary dataset '{dataset_name}' has zero class target entities."
            )
        if not gold:
            raise HeldoutClassRunnerValidationError(
                f"Held-out secondary dataset '{dataset_name}' has zero class gold mappings."
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
        candidate_reduction_ratio = 1.0 - (retained_candidate_size / target_pool_size)

        for method_run in method_runs:
            run_start = time.perf_counter()
            predictions = _predict_for_method(
                method_name=method_run.method_name,
                hyperparameters=method_run.hyperparameters,
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
            raw_recalls, mrr = compute_recall_at_ks_and_mrr(predictions, gold, evaluation_ks)
            recalls = {k: raw_recalls[k] for k in evaluation_ks}
            results.append(
                HeldoutClassResultRecord(
                    dataset_name=dataset.name,
                    track=dataset.track,
                    version=dataset.version,
                    entity_type="class",
                    model=method_run.method_name,
                    hyperparameters=method_run.hyperparameters,
                    gold_count=len(gold),
                    target_pool_size=target_pool_size,
                    retained_candidate_size=retained_candidate_size,
                    candidate_reduction_ratio=candidate_reduction_ratio,
                    recalls=recalls,
                    mrr=mrr,
                    runtime_seconds=time.perf_counter() - run_start,
                )
            )

    _persist_results_to_csv(
        results,
        output_csv_path=resolved_output_csv_path,
        evaluation_ks=evaluation_ks,
    )
    logger.info(
        "Persisted %d held-out class-only result record(s) to %s",
        len(results),
        resolved_output_csv_path,
    )
    return results
