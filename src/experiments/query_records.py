"""Query-level (one row per gold pair) records shared by all evaluation tiers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict

from src.evaluation.metrics import gold_target_ranks

QUERY_CSV_COLUMNS: tuple[str, ...] = (
    "track",
    "version",
    "dataset",
    "entity_type",
    "method",
    "hyperparameters",
    "source_entity",
    "source_label",
    "gold_target",
    "gold_target_label",
    "gold_rank",
    "retrieved_in_top_kmax",
    "retrieval_band",
    "gold_score",
    "kth_score",
    "candidate_count",
)

# Results-file prefix -> query-file prefix, so query files sort next to their results.
_QUERY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("heldout_class_result_", "heldout_class_query_result_"),
    ("heldout_result_", "heldout_query_result_"),
    ("result_", "query_result_"),
)


class QueryRecord(TypedDict):
    track: str
    version: str
    dataset_name: str
    entity_type: str
    model: str
    hyperparameters: dict[str, object]
    source_entity: str
    source_label: str
    gold_target: str
    gold_target_label: str
    gold_rank: int
    retrieved_in_top_kmax: bool
    retrieval_band: str
    gold_score: float | None
    kth_score: float | None
    candidate_count: int


def query_output_csv_path(results_output_csv_path: Path) -> Path:
    """Return the query-level CSV path that accompanies a results CSV."""
    stem = results_output_csv_path.stem
    for results_prefix, query_prefix in _QUERY_PREFIXES:
        if stem.startswith(results_prefix):
            return results_output_csv_path.with_name(
                f"{query_prefix}{stem.removeprefix(results_prefix)}.csv"
            )
    return results_output_csv_path.with_name(f"{stem}_query.csv")


def retrieval_band(gold_rank: int) -> str:
    if gold_rank <= 0:
        return "missed"
    if gold_rank <= 10:
        return "strong"
    if gold_rank <= 50:
        return "weak"
    return "missed"


def build_query_records(
    *,
    track: str,
    version: str,
    dataset_name: str,
    entity_type: str,
    method_name: str,
    hyperparameters: dict[str, object],
    eval_sources: Sequence[str],
    gold: Mapping[str, Sequence[str]],
    predictions: Mapping[str, list[tuple[str, float]]],
    source_label_map: Mapping[str, str],
    target_label_map: Mapping[str, str],
    k_max: int,
) -> list[QueryRecord]:
    """Build one record per (source, gold target) pair for a single method run."""
    records: list[QueryRecord] = []
    for source_id in eval_sources:
        ranked = predictions.get(source_id, [])
        scores_by_target: dict[str, float] = {}
        for target, score in ranked:
            scores_by_target.setdefault(target, float(score))
        kth_score = float(ranked[-1][1]) if ranked else None
        for gold_target, gold_rank in gold_target_ranks(ranked, gold[source_id]).items():
            records.append(
                QueryRecord(
                    track=track,
                    version=version,
                    dataset_name=dataset_name,
                    entity_type=entity_type,
                    model=method_name,
                    hyperparameters=hyperparameters,
                    source_entity=source_id,
                    source_label=source_label_map[source_id],
                    gold_target=gold_target,
                    gold_target_label=target_label_map.get(gold_target, ""),
                    gold_rank=gold_rank,
                    retrieved_in_top_kmax=(0 < gold_rank <= k_max),
                    retrieval_band=retrieval_band(gold_rank),
                    gold_score=scores_by_target[gold_target] if gold_rank > 0 else None,
                    kth_score=kth_score,
                    candidate_count=len(ranked),
                )
            )
    return records


def write_query_csv(records: Iterable[QueryRecord], output_csv_path: str | Path) -> None:
    """Write query records sorted by track, dataset, entity type, method, settings, source."""
    ordered = sorted(
        records,
        key=lambda record: (
            record["track"],
            record["dataset_name"],
            record["entity_type"],
            record["model"],
            json.dumps(record["hyperparameters"], sort_keys=True),
            record["source_entity"],
            record["gold_target"],
        ),
    )
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(QUERY_CSV_COLUMNS))
        writer.writeheader()
        for record in ordered:
            writer.writerow(
                {
                    "track": record["track"],
                    "version": record["version"],
                    "dataset": record["dataset_name"],
                    "entity_type": record["entity_type"],
                    "method": record["model"],
                    "hyperparameters": json.dumps(
                        record["hyperparameters"], sort_keys=True, separators=(",", ":")
                    ),
                    "source_entity": record["source_entity"],
                    "source_label": record["source_label"],
                    "gold_target": record["gold_target"],
                    "gold_target_label": record["gold_target_label"],
                    "gold_rank": record["gold_rank"],
                    "retrieved_in_top_kmax": record["retrieved_in_top_kmax"],
                    "retrieval_band": record["retrieval_band"],
                    "gold_score": "" if record["gold_score"] is None else record["gold_score"],
                    "kth_score": "" if record["kth_score"] is None else record["kth_score"],
                    "candidate_count": record["candidate_count"],
                }
            )
