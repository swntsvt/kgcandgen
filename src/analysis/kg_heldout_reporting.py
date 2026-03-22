"""Reporting utilities for held-out KG evaluation artifacts."""

from __future__ import annotations

import csv
from itertools import combinations
import json
import logging
from pathlib import Path

import pandas as pd

from src.analysis.heldout_inference import (
    paired_sign_flip_p_value,
    paired_bootstrap_confidence_interval,
)
from src.method_registry import PRIMARY_COMPARISON_METHODS, ordered_method_names, supports_primary_comparison

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
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
}
REQUIRED_ENTITY_TYPES = {"class", "predicate", "instance"}
RETRIEVAL_BANDS = ("strong", "weak", "missed")
REQUIRED_QUERY_COLUMNS = {
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
}


class KgHeldoutReportingValidationError(ValueError):
    """Raised when held-out reporting inputs are invalid."""


def _coerce_float(value: object) -> float:
    if isinstance(value, str):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    # Handle numpy-style scalar objects (for example np.int64 / np.float64).
    if hasattr(value, "item"):
        try:
            scalar_value = value.item()  # type: ignore[call-arg]
        except (TypeError, ValueError, AttributeError):
            scalar_value = None
        if isinstance(scalar_value, (int, float, str)):
            return float(scalar_value)
    raise TypeError(f"Unsupported float value: {value!r}")


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric in (0.0, 1.0):
            return bool(int(numeric))
        raise ValueError(f"Unsupported boolean numeric value: {value!r}")
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def _resolve_results_csv(results_csv_path: str | Path | None) -> Path:
    if results_csv_path is not None:
        path = Path(results_csv_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Held-out KG results CSV not found: {path}")
        return path

    candidates = sorted(
        Path("results").glob("heldout_result_*.csv"),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No results/heldout_result_*.csv files found for reporting.")
    return candidates[-1].resolve()


def _resolve_selected_settings_json(
    selected_settings_path: str | Path | None,
    *,
    heldout_datasets: set[str],
) -> tuple[Path | None, bool]:
    if selected_settings_path is not None:
        path = Path(selected_settings_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Selected settings JSON not found: {path}")
        return path, True

    candidates = sorted(
        Path("results/comparisons").glob("**/heldout_selected_settings.json"),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if not candidates:
        return None, False

    matching_candidates: list[Path] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        candidate_datasets = payload.get("heldout_datasets")
        if not isinstance(candidate_datasets, list):
            continue
        candidate_dataset_set = {str(value) for value in candidate_datasets}
        if candidate_dataset_set == heldout_datasets:
            matching_candidates.append(candidate.resolve())

    if len(matching_candidates) == 1:
        return matching_candidates[0], False

    if len(matching_candidates) > 1:
        logger.warning(
            "Skipping transfer summary because multiple selected-settings artifacts match held-out datasets: %s",
            matching_candidates,
        )
        return None, False

    logger.warning(
        "Skipping transfer summary because no selected-settings artifact matches held-out datasets: %s",
        sorted(heldout_datasets),
    )
    return None, False


def _infer_query_level_csv_path(results_csv: Path) -> Path:
    stem = results_csv.stem
    if stem.startswith("heldout_result_"):
        suffix = stem.removeprefix("heldout_result_")
        name = f"heldout_query_result_{suffix}.csv"
    else:
        name = f"{stem}_query.csv"
    return results_csv.with_name(name)


def _resolve_query_level_csv(
    *,
    results_csv: Path,
    query_level_csv_path: str | Path | None,
) -> Path:
    if query_level_csv_path is not None:
        path = Path(query_level_csv_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Held-out query-level CSV not found: {path}")
        return path

    inferred = _infer_query_level_csv_path(results_csv).resolve()
    if inferred.exists():
        return inferred

    raise FileNotFoundError(
        "Could not auto-detect held-out query-level CSV for reporting. "
        f"Expected paired file: {inferred}. Pass --query-level-csv explicitly."
    )


def _find_recall_columns(frame: pd.DataFrame) -> list[str]:
    recall_columns = [column for column in frame.columns if column.startswith("recall_at_")]
    if not recall_columns:
        raise KgHeldoutReportingValidationError(
            "Held-out KG reporting requires at least one recall_at_<k> column."
        )
    return sorted(recall_columns, key=lambda column: int(column.removeprefix("recall_at_")))


def _validate_query_frame(
    frame: pd.DataFrame,
    *,
    methods: list[str],
    k_max: int,
) -> pd.DataFrame:
    missing_columns = sorted(REQUIRED_QUERY_COLUMNS - set(frame.columns))
    if missing_columns:
        raise KgHeldoutReportingValidationError(
            "Missing required held-out query-level column(s): " + ", ".join(missing_columns)
        )

    validated = frame.copy()
    for column in (
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
        "retrieval_band",
    ):
        validated[column] = validated[column].astype(str)

    validated["gold_rank"] = validated["gold_rank"].astype(int)
    try:
        validated["retrieved_in_top_kmax"] = validated["retrieved_in_top_kmax"].map(_coerce_bool)
    except ValueError as exc:
        raise KgHeldoutReportingValidationError(
            "Held-out query-level CSV contains invalid retrieved_in_top_kmax values."
        ) from exc

    entity_types = set(validated["entity_type"].unique())
    if entity_types != REQUIRED_ENTITY_TYPES:
        raise KgHeldoutReportingValidationError(
            "Held-out query-level reporting requires exactly these entity types: "
            + ", ".join(sorted(REQUIRED_ENTITY_TYPES))
        )

    method_set = set(validated["method"].unique())
    expected_methods = set(methods)
    if method_set != expected_methods:
        raise KgHeldoutReportingValidationError(
            "Held-out query-level CSV method set must match held-out results CSV method set. "
            f"Expected {sorted(expected_methods)}, found {sorted(method_set)}."
        )

    bands = set(validated["retrieval_band"].unique())
    if not bands.issubset(set(RETRIEVAL_BANDS)):
        raise KgHeldoutReportingValidationError(
            "Held-out query-level CSV contains unsupported retrieval_band values."
        )

    duplicates = (
        validated.groupby(
            [
                "track",
                "version",
                "dataset",
                "entity_type",
                "method",
                "source_entity",
                "gold_target",
            ]
        )
        .size()
        .reset_index(name="row_count")
    )
    duplicated_rows = duplicates[duplicates["row_count"] != 1]
    if not duplicated_rows.empty:
        bad = duplicated_rows.iloc[0]
        raise KgHeldoutReportingValidationError(
            "Held-out query-level CSV requires exactly one row per "
            "track/version/dataset/entity_type/method/source_entity/gold_target. "
            f"Found {int(bad['row_count'])} row(s) for dataset='{bad['dataset']}', "
            f"entity_type='{bad['entity_type']}', method='{bad['method']}', "
            f"source_entity='{bad['source_entity']}'."
        )

    for row in validated.to_dict(orient="records"):
        gold_rank = int(_coerce_float(row["gold_rank"]))
        retrieved_in_top_kmax = _coerce_bool(row["retrieved_in_top_kmax"])
        expected_retrieved = gold_rank > 0 and gold_rank <= k_max
        if retrieved_in_top_kmax != expected_retrieved:
            raise KgHeldoutReportingValidationError(
                "Held-out query-level CSV contains inconsistent retrieved_in_top_kmax values."
            )
        if gold_rank <= 0:
            expected_band = "missed"
        elif gold_rank <= 10:
            expected_band = "strong"
        elif gold_rank <= 50:
            expected_band = "weak"
        else:
            expected_band = "missed"
        if str(row["retrieval_band"]) != expected_band:
            raise KgHeldoutReportingValidationError(
                "Held-out query-level CSV contains retrieval_band values inconsistent with gold_rank."
            )

    return validated.sort_values(
        ["track", "dataset", "entity_type", "method", "source_entity"],
        kind="mergesort",
    ).reset_index(drop=True)


def _validate_results_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise KgHeldoutReportingValidationError(
            "Missing required held-out reporting column(s): " + ", ".join(missing_columns)
        )

    validated = frame.copy()
    validated["track"] = validated["track"].astype(str)
    validated["version"] = validated["version"].astype(str)
    validated["dataset"] = validated["dataset"].astype(str)
    validated["entity_type"] = validated["entity_type"].astype(str)
    validated["method"] = validated["method"].astype(str)
    validated["hyperparameters"] = validated["hyperparameters"].astype(str)

    numeric_columns = [
        "gold_count",
        "target_pool_size",
        "retained_candidate_size",
        "candidate_reduction_ratio",
        "mrr",
        "runtime_seconds",
    ]
    for column in numeric_columns:
        validated[column] = validated[column].astype(float)

    recall_columns = _find_recall_columns(validated)
    for column in recall_columns:
        validated[column] = validated[column].astype(float)

    methods = ordered_method_names(validated["method"].unique())
    if not methods:
        raise KgHeldoutReportingValidationError(
            "Held-out KG reporting requires at least one method row."
        )

    entity_types = set(validated["entity_type"].unique())
    if entity_types != REQUIRED_ENTITY_TYPES:
        raise KgHeldoutReportingValidationError(
            "Held-out KG reporting requires exactly these entity types: "
            + ", ".join(sorted(REQUIRED_ENTITY_TYPES))
        )

    coverage = (
        validated.groupby(["track", "version", "dataset", "entity_type"])["method"]
        .nunique()
        .reset_index(name="method_count")
    )
    incomplete = coverage[coverage["method_count"] != len(methods)]
    if not incomplete.empty:
        bad = incomplete.iloc[0]
        raise KgHeldoutReportingValidationError(
            "Each dataset/entity_type pair must have one row for every reported method. "
            f"Found incomplete coverage for dataset='{bad['dataset']}', entity_type='{bad['entity_type']}'."
        )

    duplicates = (
        validated.groupby(["track", "version", "dataset", "entity_type", "method"])
        .size()
        .reset_index(name="row_count")
    )
    duplicated_rows = duplicates[duplicates["row_count"] != 1]
    if not duplicated_rows.empty:
        bad = duplicated_rows.iloc[0]
        raise KgHeldoutReportingValidationError(
            "Held-out reporting requires exactly one row per dataset/entity_type/method. "
            f"Found {int(bad['row_count'])} row(s) for dataset='{bad['dataset']}', "
            f"entity_type='{bad['entity_type']}', method='{bad['method']}'."
        )

    return validated.sort_values(
        ["track", "dataset", "entity_type", "method"], kind="mergesort"
    ).reset_index(drop=True), recall_columns, methods


def _build_by_type_summary(
    frame: pd.DataFrame,
    recall_columns: list[str],
    methods: list[str],
) -> pd.DataFrame:
    aggregation: dict[str, str] = {
        "dataset": "count",
        "gold_count": "sum",
        "target_pool_size": "mean",
        "retained_candidate_size": "mean",
        "candidate_reduction_ratio": "mean",
        "mrr": "mean",
    }
    for column in recall_columns:
        aggregation[column] = "mean"

    summary = (
        frame.groupby(["entity_type", "method"], as_index=False)
        .agg(aggregation)
        .rename(
            columns={
                "dataset": "dataset_count",
                "gold_count": "gold_count_sum",
                "target_pool_size": "target_pool_size_mean",
                "retained_candidate_size": "retained_candidate_size_mean",
                "candidate_reduction_ratio": "candidate_reduction_ratio_mean",
                "mrr": "mrr_mean",
                **{column: f"{column}_mean" for column in recall_columns},
            }
        )
    )
    method_order = {method: index for index, method in enumerate(methods)}
    summary["_method_order"] = summary["method"].map(
        lambda value: method_order.get(str(value), len(method_order))
    )
    return summary.sort_values(
        ["entity_type", "_method_order", "method"], kind="mergesort"
    ).drop(columns="_method_order").reset_index(drop=True)


def _delta_row(
    frame: pd.DataFrame,
    *,
    value_columns: list[str],
    method_column: str = "method",
) -> dict[str, object]:
    indexed = frame.set_index(method_column)
    if not supports_primary_comparison(indexed.index):
        raise KgHeldoutReportingValidationError("Expected tfidf and bm25 rows for delta computation.")

    row: dict[str, object] = {method_column: "delta_tfidf_minus_bm25"}
    for column in value_columns:
        left_value = _coerce_float(indexed.loc[PRIMARY_COMPARISON_METHODS[0], column])
        right_value = _coerce_float(indexed.loc[PRIMARY_COMPARISON_METHODS[1], column])
        row[column] = float(
            left_value - right_value
        )
    return row


def _build_macro_summary(
    by_type_summary: pd.DataFrame,
    recall_columns: list[str],
    methods: list[str],
) -> pd.DataFrame:
    value_columns = [
        "dataset_count",
        "gold_count_sum",
        "target_pool_size_mean",
        "retained_candidate_size_mean",
        "candidate_reduction_ratio_mean",
        "mrr_mean",
        *(f"{column}_mean" for column in recall_columns),
    ]
    method_rows: list[dict[str, object]] = []
    for method in methods:
        group = by_type_summary[by_type_summary["method"] == method].sort_values(
            ["entity_type"], kind="mergesort"
        )
        row = {"method": method, "type_count": int(len(group))}
        for column in value_columns:
            row[column] = float(group[column].mean())
        method_rows.append(row)

    summary = pd.DataFrame(method_rows)
    if supports_primary_comparison(methods):
        summary = pd.concat(
            [summary, pd.DataFrame([_delta_row(summary, value_columns=value_columns)])],
            ignore_index=True,
        )
    summary = summary.rename(
        columns={
            "dataset_count": "dataset_count_macro_mean",
            "gold_count_sum": "gold_count_sum_macro_mean",
        }
    )
    return summary.sort_values(["method"], kind="mergesort").reset_index(drop=True)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise KgHeldoutReportingValidationError("Gold-weighted micro-average requires positive total gold_count.")
    return float((values * weights).sum() / total_weight)


def _build_micro_summary(
    frame: pd.DataFrame,
    recall_columns: list[str],
    methods: list[str],
) -> pd.DataFrame:
    method_rows: list[dict[str, object]] = []
    for method in methods:
        group = frame[frame["method"] == method].sort_values(
            ["track", "dataset", "entity_type"], kind="mergesort"
        )
        weights = group["gold_count"]
        row: dict[str, object] = {
            "method": method,
            "dataset_type_row_count": int(len(group)),
            "gold_count_sum": float(group["gold_count"].sum()),
            "target_pool_size_sum": float(group["target_pool_size"].sum()),
            "retained_candidate_size_sum": float(group["retained_candidate_size"].sum()),
            "candidate_reduction_ratio": _weighted_mean(
                group["candidate_reduction_ratio"], weights
            ),
            "mrr": _weighted_mean(group["mrr"], weights),
        }
        for column in recall_columns:
            row[column] = _weighted_mean(group[column], weights)
        method_rows.append(row)

    value_columns = [
        "dataset_type_row_count",
        "gold_count_sum",
        "target_pool_size_sum",
        "retained_candidate_size_sum",
        "candidate_reduction_ratio",
        "mrr",
        *recall_columns,
    ]
    summary = pd.DataFrame(method_rows)
    if supports_primary_comparison(methods):
        summary = pd.concat(
            [summary, pd.DataFrame([_delta_row(summary, value_columns=value_columns)])],
            ignore_index=True,
        )
    return summary.sort_values(["method"], kind="mergesort").reset_index(drop=True)


def _build_reduction_effectiveness(
    frame: pd.DataFrame,
    recall_columns: list[str],
    methods: list[str],
) -> pd.DataFrame:
    base_columns = [
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
        *recall_columns,
        "mrr",
        "runtime_seconds",
    ]
    enriched = frame[base_columns].copy()

    if not supports_primary_comparison(methods):
        return enriched.sort_values(
            ["track", "dataset", "entity_type", "method"], kind="mergesort"
        ).reset_index(drop=True)

    primary_pair = frame[frame["method"].isin(PRIMARY_COMPARISON_METHODS)].copy()
    paired = (
        primary_pair.pivot(
            index=["track", "version", "dataset", "entity_type"],
            columns="method",
            values=["candidate_reduction_ratio", "mrr", *recall_columns],
        )
        .sort_index()
    )

    def paired_value(
        *,
        track: str,
        version: str,
        dataset: str,
        entity_type: str,
        metric: str,
        method: str,
    ) -> float:
        return _coerce_float(
            paired.loc[
                (track, version, dataset, entity_type),
                (metric, method),
            ]
        )

    other_method_map = {
        PRIMARY_COMPARISON_METHODS[0]: PRIMARY_COMPARISON_METHODS[1],
        PRIMARY_COMPARISON_METHODS[1]: PRIMARY_COMPARISON_METHODS[0],
    }
    enriched["other_method"] = enriched["method"].map(other_method_map)
    for metric in ["candidate_reduction_ratio", "mrr", *recall_columns]:
        delta_column = f"{metric}_delta_vs_other_method"
        deltas: list[float | None] = []
        for row in enriched.itertuples(index=False):
            current_method = str(row.method)
            if current_method not in other_method_map:
                deltas.append(None)
                continue
            current_value = paired_value(
                track=str(row.track),
                version=str(row.version),
                dataset=str(row.dataset),
                entity_type=str(row.entity_type),
                metric=metric,
                method=current_method,
            )
            other_value = paired_value(
                track=str(row.track),
                version=str(row.version),
                dataset=str(row.dataset),
                entity_type=str(row.entity_type),
                metric=metric,
                method=other_method_map[current_method],
            )
            deltas.append(float(current_value - other_value))
        enriched[delta_column] = deltas

    return enriched.sort_values(
        ["track", "dataset", "entity_type", "method"], kind="mergesort"
    ).reset_index(drop=True)


def _build_error_cases(query_frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
    ]
    return query_frame[columns].copy().sort_values(
        ["track", "dataset", "entity_type", "method", "source_entity"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_error_by_type_summary(
    error_cases: pd.DataFrame,
    methods: list[str],
) -> pd.DataFrame:
    grouped = (
        error_cases.groupby(["entity_type", "method", "retrieval_band"], as_index=False)
        .size()
        .rename(columns={"size": "query_count"})
    )
    totals = (
        error_cases.groupby(["entity_type", "method"], as_index=False)
        .size()
        .rename(columns={"size": "total_queries"})
    )
    full_index = pd.MultiIndex.from_product(
        [sorted(REQUIRED_ENTITY_TYPES), methods, RETRIEVAL_BANDS],
        names=["entity_type", "method", "retrieval_band"],
    )
    full = (
        grouped.set_index(["entity_type", "method", "retrieval_band"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    merged = full.merge(
        totals,
        on=["entity_type", "method"],
        how="left",
    )
    merged["total_queries"] = merged["total_queries"].fillna(0).astype(int)
    merged["query_count"] = merged["query_count"].astype(int)
    merged["query_rate"] = merged.apply(
        lambda row: float(row["query_count"] / row["total_queries"])
        if int(row["total_queries"]) > 0
        else 0.0,
        axis=1,
    )
    return merged.sort_values(
        ["entity_type", "method", "retrieval_band"], kind="mergesort"
    ).reset_index(drop=True)


def _write_error_interpretation(
    output_path: Path,
    *,
    query_level_csv: Path,
    error_summary: pd.DataFrame,
    methods: list[str],
) -> None:
    lines = [
        "# KG Held-Out Error Analysis",
        "",
        f"- Query-level input CSV: `{query_level_csv}`",
        "- Retrieval bands: `strong` (rank 1-10), `weak` (rank 11-50), `missed` (not retrieved within k_max).",
        "",
    ]
    for entity_type in sorted(REQUIRED_ENTITY_TYPES):
        lines.append(f"## {entity_type.capitalize()}")
        lines.append("")
        section = error_summary[error_summary["entity_type"] == entity_type].copy()
        for method in methods:
            method_rows = section[section["method"] == method].copy()
            if method_rows.empty:
                continue
            ranked = method_rows.sort_values(
                ["query_count", "retrieval_band"],
                ascending=[False, True],
                kind="mergesort",
            )
            dominant = ranked.iloc[0]
            lines.append(
                f"- `{method}` dominant band: `{dominant['retrieval_band']}` "
                f"({int(dominant['query_count'])}/{int(dominant['total_queries'])}, {float(dominant['query_rate']):.6f})"
            )
            for row in method_rows.itertuples(index=False):
                lines.append(
                    f"  - `{row.retrieval_band}`: count `{row.query_count}` rate `{row.query_rate:.6f}`"
                )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pairwise_method_pairs(methods: list[str]) -> list[tuple[str, str]]:
    """Return canonical unordered method pairs following registry order."""
    return list(combinations(ordered_method_names(methods), 2))


PAIRWISE_INFERENCE_COLUMNS = [
    "metric",
    "method_a",
    "method_b",
    "paired_unit_count",
    "nonzero_delta_count",
    "method_a_mean",
    "method_b_mean",
    "paired_delta",
    "ci_lower",
    "ci_upper",
    "p_value",
]


def _empty_pairwise_inference_frame(*, scope_column: str) -> pd.DataFrame:
    return pd.DataFrame(columns=[scope_column, *PAIRWISE_INFERENCE_COLUMNS])


def _build_pairwise_inference_row(
    paired_frame: pd.DataFrame,
    *,
    metric: str,
    method_a: str,
    method_b: str,
    extra_fields: dict[str, object],
) -> dict[str, object]:
    method_a_values = paired_frame[method_a].astype(float)
    method_b_values = paired_frame[method_b].astype(float)
    deltas = method_a_values - method_b_values
    ci_lower, ci_upper = paired_bootstrap_confidence_interval(deltas.tolist())
    return {
        **extra_fields,
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        "paired_unit_count": int(len(paired_frame)),
        "nonzero_delta_count": int((deltas != 0.0).sum()),
        "method_a_mean": float(method_a_values.mean()),
        "method_b_mean": float(method_b_values.mean()),
        "paired_delta": float(deltas.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": paired_sign_flip_p_value(deltas.tolist()),
    }


def _build_pairwise_by_type_inference(
    frame: pd.DataFrame,
    recall_columns: list[str],
    methods: list[str],
) -> pd.DataFrame:
    method_pairs = _pairwise_method_pairs(methods)
    if not method_pairs:
        return _empty_pairwise_inference_frame(scope_column="entity_type")

    metrics = ["mrr", *recall_columns]
    rows: list[dict[str, object]] = []

    for entity_type in sorted(REQUIRED_ENTITY_TYPES):
        subset = frame[frame["entity_type"] == entity_type].copy()
        for metric in metrics:
            paired_metric = (
                subset.pivot(
                    index=["track", "version", "dataset"],
                    columns="method",
                    values=metric,
                )
                .sort_index()
            )
            for method_a, method_b in method_pairs:
                rows.append(
                    _build_pairwise_inference_row(
                        paired_metric[[method_a, method_b]],
                        metric=metric,
                        method_a=method_a,
                        method_b=method_b,
                        extra_fields={"entity_type": entity_type},
                    )
                )

    result = pd.DataFrame(rows)
    method_order = {method: index for index, method in enumerate(ordered_method_names(methods))}
    result["_method_a_order"] = result["method_a"].map(
        lambda value: method_order.get(str(value), len(method_order))
    )
    result["_method_b_order"] = result["method_b"].map(
        lambda value: method_order.get(str(value), len(method_order))
    )
    return result.sort_values(
        ["entity_type", "metric", "_method_a_order", "_method_b_order", "method_a", "method_b"],
        kind="mergesort",
    ).drop(columns=["_method_a_order", "_method_b_order"]).reset_index(drop=True)


def _build_pairwise_overall_inference(
    frame: pd.DataFrame,
    recall_columns: list[str],
    methods: list[str],
) -> pd.DataFrame:
    method_pairs = _pairwise_method_pairs(methods)
    if not method_pairs:
        return _empty_pairwise_inference_frame(scope_column="aggregation_scope")

    metrics = ["mrr", *recall_columns]
    rows: list[dict[str, object]] = []

    overall_pivot_by_metric: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        overall_pivot_by_metric[metric] = (
            frame.pivot(
                index=["track", "version", "dataset", "entity_type"],
                columns="method",
                values=metric,
            )
            .sort_index()
        )

    type_means = (
        frame.groupby(["entity_type", "method"], as_index=False)[metrics]
        .mean(numeric_only=True)
        .sort_values(["entity_type", "method"], kind="mergesort")
    )
    macro_pivot_by_metric: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        macro_pivot_by_metric[metric] = (
            type_means.pivot(index="entity_type", columns="method", values=metric).sort_index()
        )

    for metric in metrics:
        for method_a, method_b in method_pairs:
            rows.append(
                _build_pairwise_inference_row(
                    overall_pivot_by_metric[metric][[method_a, method_b]],
                    metric=metric,
                    method_a=method_a,
                    method_b=method_b,
                    extra_fields={"aggregation_scope": "overall_dataset_type_rows"},
                )
            )
            rows.append(
                _build_pairwise_inference_row(
                    macro_pivot_by_metric[metric][[method_a, method_b]],
                    metric=metric,
                    method_a=method_a,
                    method_b=method_b,
                    extra_fields={"aggregation_scope": "macro_entity_type_means"},
                )
            )

    result = pd.DataFrame(rows)
    method_order = {method: index for index, method in enumerate(ordered_method_names(methods))}
    result["_method_a_order"] = result["method_a"].map(
        lambda value: method_order.get(str(value), len(method_order))
    )
    result["_method_b_order"] = result["method_b"].map(
        lambda value: method_order.get(str(value), len(method_order))
    )
    return result.sort_values(
        ["aggregation_scope", "metric", "_method_a_order", "_method_b_order", "method_a", "method_b"],
        kind="mergesort",
    ).drop(columns=["_method_a_order", "_method_b_order"]).reset_index(drop=True)


def _load_selected_settings_payload(
    selected_settings_path: Path | None,
    *,
    strict: bool,
    methods: list[str],
) -> tuple[dict[str, dict[str, float]] | None, str | None]:
    if selected_settings_path is None:
        return None, "No uniquely matching selected-settings artifact was found; transfer summary skipped."
    if not supports_primary_comparison(methods):
        return None, "TF-IDF/BM25 pair not present; transfer summary skipped."

    try:
        payload = json.loads(selected_settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if strict:
            raise KgHeldoutReportingValidationError(
                f"Malformed selected settings JSON: {selected_settings_path}"
            ) from exc
        logger.warning("Skipping transfer summary due to malformed selected settings JSON: %s", selected_settings_path)
        return None, "Selected-settings artifact was malformed; transfer summary skipped."

    if not isinstance(payload, dict) or not isinstance(payload.get("selected_settings"), dict):
        if strict:
            raise KgHeldoutReportingValidationError(
                f"Selected settings JSON missing 'selected_settings' object: {selected_settings_path}"
            )
        logger.warning(
            "Skipping transfer summary due to invalid selected settings payload: %s",
            selected_settings_path,
        )
        return None, "Selected-settings artifact was invalid; transfer summary skipped."

    selected_settings = payload["selected_settings"]
    transfer_payload: dict[str, dict[str, float]] = {}
    try:
        for method in PRIMARY_COMPARISON_METHODS:
            entry = selected_settings[method]
            transfer_payload[method] = {
                "development_selection_score": float(entry["heldout_score"]),
                "development_mu": float(entry["mu"]),
                "development_sigma": float(entry["sigma"]),
            }
    except (KeyError, TypeError, ValueError) as exc:
        if strict:
            raise KgHeldoutReportingValidationError(
                f"Selected settings JSON missing transfer summary fields: {selected_settings_path}"
            ) from exc
        logger.warning(
            "Skipping transfer summary due to incomplete selected settings fields: %s",
            selected_settings_path,
        )
        return None, "Selected-settings artifact was incomplete; transfer summary skipped."

    return transfer_payload, None


def _build_transfer_summary(
    transfer_payload: dict[str, dict[str, float]],
    macro_summary: pd.DataFrame,
    micro_summary: pd.DataFrame,
) -> pd.DataFrame:
    macro_by_method = macro_summary[
        macro_summary["method"].isin(PRIMARY_COMPARISON_METHODS)
    ].set_index("method")
    micro_by_method = micro_summary[
        micro_summary["method"].isin(PRIMARY_COMPARISON_METHODS)
    ].set_index("method")

    rows: list[dict[str, object]] = []
    for method in PRIMARY_COMPARISON_METHODS:
        development = transfer_payload[method]
        heldout_macro_mrr = _coerce_float(macro_by_method.loc[method, "mrr_mean"])
        heldout_micro_mrr = _coerce_float(micro_by_method.loc[method, "mrr"])
        development_mu = _coerce_float(development["development_mu"])
        rows.append(
            {
                "method": method,
                "development_selection_score": _coerce_float(development["development_selection_score"]),
                "development_mu": development_mu,
                "development_sigma": _coerce_float(development["development_sigma"]),
                "heldout_macro_mrr": heldout_macro_mrr,
                "heldout_micro_mrr": heldout_micro_mrr,
                "macro_transfer_gap": float(heldout_macro_mrr - development_mu),
                "micro_transfer_gap": float(heldout_micro_mrr - development_mu),
            }
        )

    return pd.DataFrame(rows).sort_values(["method"], kind="mergesort").reset_index(drop=True)


def _write_interpretation_scaffold(
    output_path: Path,
    *,
    source_csv: Path,
    by_type_summary: pd.DataFrame,
    macro_summary: pd.DataFrame,
    micro_summary: pd.DataFrame,
    reduction_effectiveness: pd.DataFrame,
    pairwise_by_type_inference: pd.DataFrame,
    pairwise_overall_inference: pd.DataFrame,
    primary_recall_column: str,
    transfer_summary: pd.DataFrame | None,
    transfer_note: str | None,
    methods: list[str],
) -> None:
    def _winner_label(group: pd.DataFrame, column: str) -> str:
        best_value = float(group[column].max())
        winners = ordered_method_names(
            group[group[column] == best_value]["method"].astype(str).tolist()
        )
        if len(winners) == 1:
            return f"`{winners[0]}`"
        return "tie: " + ", ".join(f"`{method}`" for method in winners)

    winners: list[str] = []
    for entity_type in sorted(REQUIRED_ENTITY_TYPES):
        group = by_type_summary[by_type_summary["entity_type"] == entity_type].copy()

        winners.append(
            f"- `{entity_type}`: MRR winner {_winner_label(group, 'mrr_mean')}, "
            f"{primary_recall_column} winner {_winner_label(group, f'{primary_recall_column}_mean')}"
        )

    macro_rows = macro_summary[macro_summary["method"].isin(methods)].set_index("method")
    micro_rows = micro_summary[micro_summary["method"].isin(methods)].set_index("method")
    reduction_snapshot = reduction_effectiveness[
        [
            "entity_type",
            "method",
            "candidate_reduction_ratio",
            "mrr",
            primary_recall_column,
        ]
    ].copy()
    reduction_snapshot = reduction_snapshot.sort_values(
        ["entity_type", "method"], kind="mergesort"
    ).groupby(["entity_type", "method"], as_index=False).mean(numeric_only=True)

    mrr_pairwise_by_type = pairwise_by_type_inference[
        pairwise_by_type_inference["metric"] == "mrr"
    ].copy()
    strongest_pairwise: list[str] = []
    for entity_type in sorted(REQUIRED_ENTITY_TYPES):
        group = mrr_pairwise_by_type[mrr_pairwise_by_type["entity_type"] == entity_type].copy()
        if group.empty:
            continue
        ranked = group.assign(abs_delta=group["paired_delta"].abs()).sort_values(
            ["abs_delta", "method_a", "method_b"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        best = ranked.iloc[0]
        strongest_pairwise.append(
            f"- `{entity_type}`: `{best['method_a']}` vs `{best['method_b']}` "
            f"delta `{best['paired_delta']:.6f}`, CI [`{best['ci_lower']:.6f}`, `{best['ci_upper']:.6f}`], "
            f"p-value `{best['p_value']:.6f}`"
        )

    directional_consistency: list[str] = []
    for method_a, method_b in _pairwise_method_pairs(methods):
        pair_rows = mrr_pairwise_by_type[
            (mrr_pairwise_by_type["method_a"] == method_a)
            & (mrr_pairwise_by_type["method_b"] == method_b)
        ].copy()
        if pair_rows.empty:
            continue
        signs = {float(value) > 0.0 for value in pair_rows["paired_delta"] if float(value) != 0.0}
        status = "directionally consistent" if len(signs) <= 1 else "mixed direction"
        directional_consistency.append(f"- `{method_a}` vs `{method_b}`: {status}")

    overall_mrr = pairwise_overall_inference[
        pairwise_overall_inference["metric"] == "mrr"
    ].copy().sort_values(
        ["aggregation_scope", "method_a", "method_b"], kind="mergesort"
    )
    ci_overlap_zero = mrr_pairwise_by_type[
        (mrr_pairwise_by_type["ci_lower"] <= 0.0)
        & (mrr_pairwise_by_type["ci_upper"] >= 0.0)
    ].copy().sort_values(["entity_type", "method_a", "method_b"], kind="mergesort")

    lines = [
        "# KG Held-Out Interpretation Scaffold",
        "",
        "## Inputs",
        "",
        f"- Held-out results CSV: `{source_csv}`",
        "",
        "## Coverage",
        "",
        "- Entity types covered: `class`, `predicate`, `instance`",
        f"- Methods compared: {', '.join(f'`{method}`' for method in methods)}",
        "",
        "## Per-Type Winners",
        "",
        *winners,
        "",
        "## Macro Summary",
        "",
    ]
    for method in methods:
        lines.append(f"- `{method}` macro MRR: `{macro_rows.loc[method, 'mrr_mean']:.6f}`")
    lines.extend(
        [
            "",
            "## Micro Summary",
            "",
        ]
    )
    for method in methods:
        lines.append(f"- `{method}` micro MRR: `{micro_rows.loc[method, 'mrr']:.6f}`")
    lines.extend(
        [
            "",
        ]
    )

    lines.extend(
        [
            "## Reduction vs Effectiveness Prompts",
            "",
            f"- Review `{primary_recall_column}` alongside `candidate_reduction_ratio` for each entity type.",
            "- Check whether the higher-reduction method also preserves MRR consistency across entity types.",
            "- Candidate Reduction Ratio is descriptive only and is not included in significance testing.",
            "",
            "## Pairwise Inference Snapshot",
            "",
            "### Strongest Per-Type MRR Comparisons",
            "",
            *strongest_pairwise,
            "",
            "### Directional Consistency Across Entity Types",
            "",
            *directional_consistency,
            "",
            "### Overall MRR Inference",
            "",
        ]
    )
    for row in overall_mrr.itertuples(index=False):
        lines.append(
            f"- `{row.aggregation_scope}` / `{row.method_a}` vs `{row.method_b}`: "
            f"delta `{row.paired_delta:.6f}`, CI [`{row.ci_lower:.6f}`, `{row.ci_upper:.6f}`], "
            f"p-value `{row.p_value:.6f}`"
        )
    lines.extend(["", "### MRR Comparisons With CI Overlap At Zero", ""])
    if ci_overlap_zero.empty:
        lines.append("- None.")
    else:
        for row in ci_overlap_zero.itertuples(index=False):
            lines.append(
                f"- `{row.entity_type}` / `{row.method_a}` vs `{row.method_b}` "
                f"CI [`{row.ci_lower:.6f}`, `{row.ci_upper:.6f}`], p-value `{row.p_value:.6f}`"
            )
    lines.append("")

    if transfer_summary is not None:
        lines.extend(
            [
                "## Transfer Summary",
                "",
            ]
        )
        for row in transfer_summary.itertuples(index=False):
            lines.append(
                f"- `{row.method}`: development selection score `{row.development_selection_score:.6f}`, "
                f"held-out macro MRR `{row.heldout_macro_mrr:.6f}`, held-out micro MRR `{row.heldout_micro_mrr:.6f}`"
            )
        lines.append("")
    elif transfer_note is not None:
        lines.extend(
            [
                "## Transfer Summary",
                "",
                f"- {transfer_note}",
                "",
            ]
        )

    lines.extend(
        [
            "## Trade-Off Snapshot",
            "",
        ]
    )
    for row in reduction_snapshot.itertuples(index=False):
        lines.append(
            f"- `{row.entity_type}` / `{row.method}`: reduction `{row.candidate_reduction_ratio:.6f}`, "
            f"MRR `{row.mrr:.6f}`, {primary_recall_column} `{getattr(row, primary_recall_column):.6f}`"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_kg_heldout_reporting(
    results_csv_path: str | Path | None,
    output_dir: str | Path = "results/comparisons",
    selected_settings_path: str | Path | None = None,
    query_level_csv_path: str | Path | None = None,
) -> dict[str, Path]:
    """Generate held-out KG reporting artifacts."""
    source_csv = _resolve_results_csv(results_csv_path)
    frame = pd.read_csv(source_csv)
    validated, recall_columns, methods = _validate_results_frame(frame)
    k_max = max(int(column.removeprefix("recall_at_")) for column in recall_columns)
    query_source_csv = _resolve_query_level_csv(
        results_csv=source_csv,
        query_level_csv_path=query_level_csv_path,
    )
    query_frame = pd.read_csv(query_source_csv)
    validated_query = _validate_query_frame(query_frame, methods=methods, k_max=k_max)
    heldout_datasets = set(validated["dataset"].unique())

    by_type_summary = _build_by_type_summary(validated, recall_columns, methods)
    macro_summary = _build_macro_summary(by_type_summary, recall_columns, methods)
    micro_summary = _build_micro_summary(validated, recall_columns, methods)
    reduction_effectiveness = _build_reduction_effectiveness(validated, recall_columns, methods)
    pairwise_by_type_inference = _build_pairwise_by_type_inference(
        validated, recall_columns, methods
    )
    pairwise_overall_inference = _build_pairwise_overall_inference(
        validated, recall_columns, methods
    )
    error_cases = _build_error_cases(validated_query)
    error_by_type_summary = _build_error_by_type_summary(error_cases, methods)

    selected_settings_json, selected_settings_explicit = _resolve_selected_settings_json(
        selected_settings_path,
        heldout_datasets=heldout_datasets,
    )
    transfer_payload, transfer_note = _load_selected_settings_payload(
        selected_settings_json,
        strict=selected_settings_explicit,
        methods=methods,
    )
    transfer_summary = (
        _build_transfer_summary(transfer_payload, macro_summary, micro_summary)
        if transfer_payload is not None
        else None
    )

    run_id = source_csv.stem
    output_root = Path(output_dir) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    by_type_path = output_root / "kg_heldout_by_type_summary.csv"
    macro_path = output_root / "kg_heldout_macro_summary.csv"
    micro_path = output_root / "kg_heldout_micro_summary.csv"
    reduction_path = output_root / "kg_heldout_reduction_effectiveness.csv"
    pairwise_by_type_path = output_root / "kg_heldout_pairwise_by_type_inference.csv"
    pairwise_overall_path = output_root / "kg_heldout_pairwise_overall_inference.csv"
    error_cases_path = output_root / "kg_heldout_error_cases.csv"
    error_summary_path = output_root / "kg_heldout_error_by_type_summary.csv"
    error_interpretation_path = output_root / "kg_heldout_error_interpretation.md"
    interpretation_path = output_root / "kg_heldout_interpretation_scaffold.md"
    transfer_path = output_root / "kg_heldout_transfer_summary.csv"
    manifest_path = output_root / "manifest.txt"

    by_type_summary.to_csv(by_type_path, index=False)
    macro_summary.to_csv(macro_path, index=False)
    micro_summary.to_csv(micro_path, index=False)
    reduction_effectiveness.to_csv(reduction_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pairwise_by_type_inference.to_csv(pairwise_by_type_path, index=False)
    pairwise_overall_inference.to_csv(pairwise_overall_path, index=False)
    error_cases.to_csv(error_cases_path, index=False, quoting=csv.QUOTE_MINIMAL)
    error_by_type_summary.to_csv(error_summary_path, index=False)
    if transfer_summary is not None:
        transfer_summary.to_csv(transfer_path, index=False)

    primary_recall_column = recall_columns[-1]
    _write_interpretation_scaffold(
        interpretation_path,
        source_csv=source_csv,
        by_type_summary=by_type_summary,
        macro_summary=macro_summary,
        micro_summary=micro_summary,
        reduction_effectiveness=reduction_effectiveness,
        pairwise_by_type_inference=pairwise_by_type_inference,
        pairwise_overall_inference=pairwise_overall_inference,
        primary_recall_column=primary_recall_column,
        transfer_summary=transfer_summary,
        transfer_note=transfer_note,
        methods=methods,
    )
    _write_error_interpretation(
        error_interpretation_path,
        query_level_csv=query_source_csv,
        error_summary=error_by_type_summary,
        methods=methods,
    )

    manifest_lines = [
        f"source_csv={source_csv}",
        f"query_level_csv={query_source_csv}",
        f"output_dir={output_root}",
        f"selected_settings_json={selected_settings_json}" if selected_settings_json is not None else "selected_settings_json=",
    ]
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    artifacts: dict[str, Path] = {
        "source_csv": source_csv,
        "query_level_csv": query_source_csv,
        "output_dir": output_root,
        "kg_heldout_by_type_summary": by_type_path,
        "kg_heldout_macro_summary": macro_path,
        "kg_heldout_micro_summary": micro_path,
        "kg_heldout_reduction_effectiveness": reduction_path,
        "kg_heldout_pairwise_by_type_inference": pairwise_by_type_path,
        "kg_heldout_pairwise_overall_inference": pairwise_overall_path,
        "kg_heldout_error_cases": error_cases_path,
        "kg_heldout_error_by_type_summary": error_summary_path,
        "kg_heldout_error_interpretation": error_interpretation_path,
        "kg_heldout_interpretation_scaffold": interpretation_path,
        "manifest": manifest_path,
    }
    if transfer_summary is not None:
        artifacts["kg_heldout_transfer_summary"] = transfer_path
    return artifacts
