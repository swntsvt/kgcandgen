"""Per-dataset accounting: entity counts, gold pairs, exclusions and label statistics.

One row per evaluated unit (development and class-only tiers: class; KG tier: class,
predicate, instance). Alignment-level columns repeat on each row of the same dataset.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from src.config_loader import DatasetConfig, load_runtime_config
from src.experiments.experiment_runner import (
    _extract_owl_class_entities,
    _load_store_with_fallback,
)
from src.preprocessing.text_preprocessor import preprocess_text
from src.rdf_utils.alignment_parser import (
    alignment_cell_summary,
    filter_alignment_targets,
    load_alignment_targets,
    multi_target_source_count,
)
from src.rdf_utils.entity_partitioning import (
    extract_typed_entity_partitions,
    partition_alignment_mappings_by_entity_type,
)
from src.rdf_utils.label_extractor import extract_entity_label_with_source
from src.run_metadata import build_timestamped_results_path

logger = logging.getLogger(__name__)

_LABEL_STAT_FIELDS: tuple[str, ...] = (
    "count",
    "chars_p10",
    "chars_median",
    "chars_p90",
    "chars_mean",
    "tokens_median",
    "tokens_mean",
    "tokens_p90",
    "single_token_share",
    "zero_token_share",
    "uri_fallback_share",
    "skos_label_share",
    "multi_rdfs_label_share",
)

ACCOUNTING_CSV_COLUMNS: tuple[str, ...] = (
    "tier",
    "track",
    "version",
    "dataset",
    "entity_type",
    "alignment_policy",
    "parser_backend",
    "cell_count",
    "equivalence_cell_count",
    "non_equivalence_cell_count",
    "invalid_iri_cell_count",
    "distinct_equivalence_pairs",
    "alignment_sources",
    "one_to_many_sources",
    "many_to_one_targets",
    "source_entities",
    "target_entities",
    "eval_queries",
    "gold_pairs",
    "multi_target_queries",
    "excluded_pairs",
    "excluded_source_not_in_scope",
    "excluded_target_not_in_scope",
    "excluded_untyped_pairs",
    "excluded_mixed_type_pairs",
    *(f"query_label_{name}" for name in _LABEL_STAT_FIELDS),
    *(f"target_label_{name}" for name in _LABEL_STAT_FIELDS),
)


def label_statistics(store, uris: Sequence[str]) -> dict[str, float]:
    """Character and token statistics for the labels the retrievers actually use."""
    if not uris:
        return {name: float("nan") for name in _LABEL_STAT_FIELDS}
    chars: list[int] = []
    tokens: list[int] = []
    sources: list[str] = []
    multi_rdfs = 0
    for uri in uris:
        label, source, rdfs_count = extract_entity_label_with_source(store, uri)
        chars.append(len(label))
        tokens.append(len(preprocess_text(label)))
        sources.append(source)
        multi_rdfs += rdfs_count > 1
    char_array = np.asarray(chars, dtype=float)
    token_array = np.asarray(tokens, dtype=float)
    n = float(len(uris))
    return {
        "count": n,
        "chars_p10": float(np.percentile(char_array, 10)),
        "chars_median": float(np.median(char_array)),
        "chars_p90": float(np.percentile(char_array, 90)),
        "chars_mean": float(char_array.mean()),
        "tokens_median": float(np.median(token_array)),
        "tokens_mean": float(token_array.mean()),
        "tokens_p90": float(np.percentile(token_array, 90)),
        "single_token_share": float((token_array == 1).sum() / n),
        "zero_token_share": float((token_array == 0).sum() / n),
        "uri_fallback_share": sources.count("uri") / n,
        "skos_label_share": sources.count("skos_pref_label") / n,
        "multi_rdfs_label_share": multi_rdfs / n,
    }


def _pair_count(gold: dict[str, tuple[str, ...]]) -> int:
    return sum(len(targets) for targets in gold.values())


def account_dataset(
    tier: str, dataset: DatasetConfig, *, alignment_policy: str
) -> list[dict[str, object]]:
    """Return accounting rows for every evaluated unit of one dataset."""
    source_store = _load_store_with_fallback(dataset.source_rdf)
    target_store = _load_store_with_fallback(dataset.target_rdf)
    gold_raw = load_alignment_targets(dataset.alignment_rdf, alignment_policy)  # type: ignore[arg-type]
    base = {
        "tier": tier,
        "track": dataset.track,
        "version": dataset.version,
        "dataset": dataset.name,
        "alignment_policy": alignment_policy,
        **alignment_cell_summary(dataset.alignment_rdf),
    }
    raw_pairs = _pair_count(gold_raw)

    units: list[tuple[str, list[str], list[str], dict[str, tuple[str, ...]], dict[str, int]]] = []
    if tier == "kg":
        source_parts = extract_typed_entity_partitions(source_store, graph_name="source")
        target_parts = extract_typed_entity_partitions(target_store, graph_name="target")
        gold_parts = partition_alignment_mappings_by_entity_type(
            gold_raw, source_parts, target_parts, alignment_name=dataset.name
        )
        exclusions = {
            "excluded_untyped_pairs": gold_parts.skipped_untyped_pairs,
            "excluded_mixed_type_pairs": gold_parts.skipped_mixed_type_pairs,
        }
        for entity_type, sources, targets, gold in (
            ("class", source_parts.classes, target_parts.classes, gold_parts.classes),
            ("predicate", source_parts.predicates, target_parts.predicates, gold_parts.predicates),
            ("instance", source_parts.instances, target_parts.instances, gold_parts.instances),
        ):
            units.append((entity_type, list(sources), list(targets), gold, exclusions))
    else:
        sources = _extract_owl_class_entities(source_store)
        targets = _extract_owl_class_entities(target_store)
        source_set, target_set = set(sources), set(targets)
        gold = filter_alignment_targets(gold_raw, source_set, target_set)
        pairs = [(s, t) for s, ts in gold_raw.items() for t in ts]
        exclusions = {
            "excluded_source_not_in_scope": sum(1 for s, _ in pairs if s not in source_set),
            "excluded_target_not_in_scope": sum(
                1 for s, t in pairs if s in source_set and t not in target_set
            ),
        }
        units.append(("class", sources, targets, gold, exclusions))

    total_kept = sum(_pair_count(gold) for _, _, _, gold, _ in units)
    rows: list[dict[str, object]] = []
    for entity_type, sources, targets, gold, exclusions in units:
        logger.info("Accounting %s/%s/%s", tier, dataset.name, entity_type)
        query_stats = label_statistics(source_store, sorted(gold))
        target_stats = label_statistics(target_store, targets)
        row: dict[str, object] = {
            **base,
            "entity_type": entity_type,
            "source_entities": len(sources),
            "target_entities": len(targets),
            "eval_queries": len(gold),
            "gold_pairs": _pair_count(gold),
            "multi_target_queries": multi_target_source_count(gold),
            # Pairs in the alignment that no unit of this dataset evaluates.
            "excluded_pairs": raw_pairs - total_kept,
            "excluded_source_not_in_scope": exclusions.get("excluded_source_not_in_scope", ""),
            "excluded_target_not_in_scope": exclusions.get("excluded_target_not_in_scope", ""),
            "excluded_untyped_pairs": exclusions.get("excluded_untyped_pairs", ""),
            "excluded_mixed_type_pairs": exclusions.get("excluded_mixed_type_pairs", ""),
        }
        row.update({f"query_label_{k}": v for k, v in query_stats.items()})
        row.update({f"target_label_{k}": v for k, v in target_stats.items()})
        rows.append(row)
    return rows


def run_dataset_accounting(
    *,
    config_path: str | Path = "config/runtime.yaml",
    tiers: Iterable[str] = ("development", "kg", "class"),
    output_csv_path: str | Path | None = None,
) -> list[dict[str, object]]:
    runtime = load_runtime_config(config_path=config_path)
    groups = {
        "development": runtime.development_datasets,
        "kg": runtime.heldout_datasets,
        "class": runtime.heldout_secondary_datasets,
    }
    output_path = (
        Path(output_csv_path)
        if output_csv_path is not None
        else build_timestamped_results_path("dataset_accounting")
    )
    rows: list[dict[str, object]] = []
    for tier in tiers:
        for dataset in groups[tier].values():
            rows.extend(
                account_dataset(
                    tier, dataset, alignment_policy=runtime.experiments.alignment_policy
                )
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ACCOUNTING_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d accounting row(s) to %s", len(rows), output_path)
    return rows
