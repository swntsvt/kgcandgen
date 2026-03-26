"""Reporting utilities for secondary held-out class-only evaluation artifacts."""

from __future__ import annotations

import csv
from itertools import combinations
import json
import logging
from pathlib import Path

import pandas as pd

from src.analysis.common import coerce_float, resolve_results_csv
from src.analysis.heldout_inference import (
    paired_bootstrap_confidence_interval,
    paired_sign_flip_p_value,
)
from src.method_registry import (
    PRIMARY_COMPARISON_METHODS,
    ordered_method_names,
    supports_primary_comparison,
)

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


class ClassHeldoutReportingValidationError(ValueError):
    """Raised when held-out class-only reporting inputs are invalid."""


def _coerce_float(value: object) -> float:
    """Backward-compatible wrapper around shared float coercion."""
    return coerce_float(value)


def _resolve_results_csv(results_csv_path: str | Path | None) -> Path:
    """Resolve held-out class reporting source CSV from explicit path or latest run artifact."""
    return resolve_results_csv(
        results_csv_path,
        default_glob="heldout_class_result_*.csv",
        explicit_label="Held-out class-only results CSV",
        latest_not_found_message="No results/heldout_class_result_*.csv files found for reporting.",
    )


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
        if {str(value) for value in candidate_datasets} == heldout_datasets:
            matching_candidates.append(candidate.resolve())

    if len(matching_candidates) == 1:
        return matching_candidates[0], False
    return None, False


def _find_recall_columns(frame: pd.DataFrame) -> list[str]:
    recall_columns = [column for column in frame.columns if column.startswith("recall_at_")]
    if not recall_columns:
        raise ClassHeldoutReportingValidationError(
            "Held-out class-only reporting requires at least one recall_at_<k> column."
        )
    return sorted(recall_columns, key=lambda column: int(column.removeprefix("recall_at_")))


def _validate_results_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ClassHeldoutReportingValidationError(
            "Missing required held-out reporting column(s): " + ", ".join(missing_columns)
        )

    validated = frame.copy()
    for column in ("track", "version", "dataset", "entity_type", "method", "hyperparameters"):
        validated[column] = validated[column].astype(str)

    if set(validated["entity_type"].unique()) != {"class"}:
        raise ClassHeldoutReportingValidationError(
            "Held-out class-only reporting requires class-only rows (entity_type='class')."
        )

    for column in (
        "gold_count",
        "target_pool_size",
        "retained_candidate_size",
        "candidate_reduction_ratio",
        "mrr",
        "runtime_seconds",
    ):
        validated[column] = validated[column].astype(float)

    recall_columns = _find_recall_columns(validated)
    for column in recall_columns:
        validated[column] = validated[column].astype(float)

    methods = ordered_method_names(validated["method"].unique())
    if not methods:
        raise ClassHeldoutReportingValidationError(
            "Held-out class-only reporting requires at least one method row."
        )

    coverage = (
        validated.groupby(["track", "version", "dataset"])["method"]
        .nunique()
        .reset_index(name="method_count")
    )
    incomplete = coverage[coverage["method_count"] != len(methods)]
    if not incomplete.empty:
        bad = incomplete.iloc[0]
        raise ClassHeldoutReportingValidationError(
            "Each secondary held-out dataset must have one row for every reported method. "
            f"Found incomplete coverage for dataset='{bad['dataset']}'."
        )

    duplicates = (
        validated.groupby(["track", "version", "dataset", "method"])
        .size()
        .reset_index(name="row_count")
    )
    duplicated_rows = duplicates[duplicates["row_count"] != 1]
    if not duplicated_rows.empty:
        bad = duplicated_rows.iloc[0]
        raise ClassHeldoutReportingValidationError(
            "Held-out class-only reporting requires exactly one row per dataset/method. "
            f"Found {int(bad['row_count'])} row(s) for dataset='{bad['dataset']}', method='{bad['method']}'."
        )

    return validated.sort_values(
        ["track", "dataset", "method"], kind="mergesort"
    ).reset_index(drop=True), recall_columns, methods


def _build_by_method_summary(
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
        frame.groupby(["method"], as_index=False)
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
    return summary.sort_values(["_method_order", "method"], kind="mergesort").drop(
        columns="_method_order"
    ).reset_index(drop=True)


def _delta_row(
    frame: pd.DataFrame,
    *,
    value_columns: list[str],
    method_column: str = "method",
) -> dict[str, object]:
    indexed = frame.set_index(method_column)
    row: dict[str, object] = {method_column: "delta_tfidf_minus_bm25"}
    for column in value_columns:
        left = _coerce_float(indexed.loc[PRIMARY_COMPARISON_METHODS[0], column])
        right = _coerce_float(indexed.loc[PRIMARY_COMPARISON_METHODS[1], column])
        row[column] = float(left - right)
    return row


def _build_macro_summary(
    by_method_summary: pd.DataFrame,
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
    summary = by_method_summary.copy()
    summary.insert(1, "type_count", 1)
    if supports_primary_comparison(methods):
        summary = pd.concat(
            [summary, pd.DataFrame([_delta_row(summary, value_columns=value_columns)])],
            ignore_index=True,
        )
    return summary.rename(
        columns={
            "dataset_count": "dataset_count_macro_mean",
            "gold_count_sum": "gold_count_sum_macro_mean",
        }
    ).sort_values(["method"], kind="mergesort").reset_index(drop=True)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise ClassHeldoutReportingValidationError(
            "Gold-weighted micro-average requires positive total gold_count."
        )
    return float((values * weights).sum() / total_weight)


def _build_micro_summary(
    frame: pd.DataFrame,
    recall_columns: list[str],
    methods: list[str],
) -> pd.DataFrame:
    method_rows: list[dict[str, object]] = []
    for method in methods:
        group = frame[frame["method"] == method].sort_values(
            ["track", "dataset"], kind="mergesort"
        )
        weights = group["gold_count"]
        row: dict[str, object] = {
            "method": method,
            "dataset_row_count": int(len(group)),
            "gold_count_sum": float(group["gold_count"].sum()),
            "target_pool_size_sum": float(group["target_pool_size"].sum()),
            "retained_candidate_size_sum": float(group["retained_candidate_size"].sum()),
            "candidate_reduction_ratio": _weighted_mean(group["candidate_reduction_ratio"], weights),
            "mrr": _weighted_mean(group["mrr"], weights),
        }
        for column in recall_columns:
            row[column] = _weighted_mean(group[column], weights)
        method_rows.append(row)

    value_columns = [
        "dataset_row_count",
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
        return enriched.sort_values(["track", "dataset", "method"], kind="mergesort").reset_index(drop=True)

    primary_pair = frame[frame["method"].isin(PRIMARY_COMPARISON_METHODS)].copy()
    paired = (
        primary_pair.pivot(
            index=["track", "version", "dataset"],
            columns="method",
            values=["candidate_reduction_ratio", "mrr", *recall_columns],
        )
        .sort_index()
    )
    other_method_map = {
        PRIMARY_COMPARISON_METHODS[0]: PRIMARY_COMPARISON_METHODS[1],
        PRIMARY_COMPARISON_METHODS[1]: PRIMARY_COMPARISON_METHODS[0],
    }
    enriched["other_method"] = enriched["method"].map(other_method_map)
    for metric in ["candidate_reduction_ratio", "mrr", *recall_columns]:
        deltas: list[float | None] = []
        for row in enriched.itertuples(index=False):
            current_method = str(row.method)
            if current_method not in other_method_map:
                deltas.append(None)
                continue
            current_value = _coerce_float(paired.loc[(row.track, row.version, row.dataset), (metric, current_method)])
            other_value = _coerce_float(
                paired.loc[
                    (row.track, row.version, row.dataset),
                    (metric, other_method_map[current_method]),
                ]
            )
            deltas.append(float(current_value - other_value))
        enriched[f"{metric}_delta_vs_other_method"] = deltas
    return enriched.sort_values(["track", "dataset", "method"], kind="mergesort").reset_index(drop=True)


def _pairwise_method_pairs(methods: list[str]) -> list[tuple[str, str]]:
    return list(combinations(ordered_method_names(methods), 2))


def _build_pairwise_overall_inference(
    frame: pd.DataFrame,
    recall_columns: list[str],
    methods: list[str],
) -> pd.DataFrame:
    method_pairs = _pairwise_method_pairs(methods)
    if not method_pairs:
        return pd.DataFrame(
            columns=[
                "aggregation_scope",
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
        )

    rows: list[dict[str, object]] = []
    metrics = ["mrr", *recall_columns]
    for metric in metrics:
        overall_pivot = frame.pivot(
            index=["track", "version", "dataset"], columns="method", values=metric
        ).sort_index()
        track_means = frame.groupby(["track", "method"], as_index=False)[metric].mean(
            numeric_only=True
        )
        macro_track_pivot = track_means.pivot(index="track", columns="method", values=metric).sort_index()

        for method_a, method_b in method_pairs:
            for scope, pivot in (
                ("overall_dataset_rows", overall_pivot),
                ("macro_track_means", macro_track_pivot),
            ):
                method_a_values = pivot[method_a].astype(float)
                method_b_values = pivot[method_b].astype(float)
                deltas = method_a_values - method_b_values
                ci_lower, ci_upper = paired_bootstrap_confidence_interval(deltas.tolist())
                rows.append(
                    {
                        "aggregation_scope": scope,
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "paired_unit_count": int(len(pivot)),
                        "nonzero_delta_count": int((deltas != 0.0).sum()),
                        "method_a_mean": float(method_a_values.mean()),
                        "method_b_mean": float(method_b_values.mean()),
                        "paired_delta": float(deltas.mean()),
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "p_value": paired_sign_flip_p_value(deltas.tolist()),
                    }
                )

    method_order = {method: index for index, method in enumerate(methods)}
    result = pd.DataFrame(rows)
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
        return None, "No selected-settings artifact was resolved; transfer summary skipped."
    if not supports_primary_comparison(methods):
        return None, "TF-IDF/BM25 pair not present; transfer summary skipped."
    try:
        payload = json.loads(selected_settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        if strict:
            raise ClassHeldoutReportingValidationError(
                f"Malformed selected settings JSON: {selected_settings_path}"
            ) from exc
        return None, "Selected-settings artifact was malformed; transfer summary skipped."

    if not isinstance(payload, dict) or not isinstance(payload.get("selected_settings"), dict):
        if strict:
            raise ClassHeldoutReportingValidationError(
                f"Selected settings JSON missing 'selected_settings' object: {selected_settings_path}"
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
            raise ClassHeldoutReportingValidationError(
                f"Selected settings JSON missing transfer summary fields: {selected_settings_path}"
            ) from exc
        return None, "Selected-settings artifact was incomplete; transfer summary skipped."
    return transfer_payload, None


def _build_transfer_summary(
    transfer_payload: dict[str, dict[str, float]],
    macro_summary: pd.DataFrame,
    micro_summary: pd.DataFrame,
) -> pd.DataFrame:
    macro_by_method = macro_summary[macro_summary["method"].isin(PRIMARY_COMPARISON_METHODS)].set_index("method")
    micro_by_method = micro_summary[micro_summary["method"].isin(PRIMARY_COMPARISON_METHODS)].set_index("method")
    rows: list[dict[str, object]] = []
    for method in PRIMARY_COMPARISON_METHODS:
        development_mu = _coerce_float(transfer_payload[method]["development_mu"])
        heldout_macro_mrr = _coerce_float(macro_by_method.loc[method, "mrr_mean"])
        heldout_micro_mrr = _coerce_float(micro_by_method.loc[method, "mrr"])
        rows.append(
            {
                "method": method,
                "development_selection_score": _coerce_float(
                    transfer_payload[method]["development_selection_score"]
                ),
                "development_mu": development_mu,
                "development_sigma": _coerce_float(transfer_payload[method]["development_sigma"]),
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
    macro_summary: pd.DataFrame,
    micro_summary: pd.DataFrame,
    pairwise_overall: pd.DataFrame,
    primary_recall_column: str,
    transfer_summary: pd.DataFrame | None,
    transfer_note: str | None,
    methods: list[str],
) -> None:
    macro_rows = macro_summary[macro_summary["method"].isin(methods)].set_index("method")
    micro_rows = micro_summary[micro_summary["method"].isin(methods)].set_index("method")
    lines = [
        "# Secondary Held-Out Class-Only Interpretation Scaffold",
        "",
        f"- Source CSV: `{source_csv}`",
        f"- Methods compared: {', '.join(f'`{method}`' for method in methods)}",
        "",
        "## Macro Summary",
        "",
    ]
    for method in methods:
        lines.append(f"- `{method}` macro MRR: `{macro_rows.loc[method, 'mrr_mean']:.6f}`")
    lines.extend(["", "## Micro Summary", ""])
    for method in methods:
        lines.append(f"- `{method}` micro MRR: `{micro_rows.loc[method, 'mrr']:.6f}`")
    lines.extend(
        [
            "",
            "## Trade-Off Prompts",
            "",
            f"- Compare `{primary_recall_column}` and `candidate_reduction_ratio` per method.",
            "- Candidate Reduction Ratio remains descriptive and is not significance-tested.",
            "",
            "## Pairwise Inference (MRR)",
            "",
        ]
    )
    overall_mrr = pairwise_overall[pairwise_overall["metric"] == "mrr"].copy()
    for row in overall_mrr.itertuples(index=False):
        lines.append(
            f"- `{row.aggregation_scope}` / `{row.method_a}` vs `{row.method_b}`: "
            f"delta `{row.paired_delta:.6f}`, CI [`{row.ci_lower:.6f}`, `{row.ci_upper:.6f}`], "
            f"p-value `{row.p_value:.6f}`"
        )
    lines.append("")
    if transfer_summary is not None:
        lines.extend(["## Transfer Summary", ""])
        for row in transfer_summary.itertuples(index=False):
            lines.append(
                f"- `{row.method}` development score `{row.development_selection_score:.6f}`, "
                f"held-out macro MRR `{row.heldout_macro_mrr:.6f}`, held-out micro MRR `{row.heldout_micro_mrr:.6f}`"
            )
        lines.append("")
    elif transfer_note is not None:
        lines.extend(["## Transfer Summary", "", f"- {transfer_note}", ""])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_class_heldout_reporting(
    results_csv_path: str | Path | None,
    output_dir: str | Path = "results/comparisons",
    selected_settings_path: str | Path | None = None,
) -> dict[str, Path]:
    """Generate secondary held-out class-only reporting artifacts."""
    source_csv = _resolve_results_csv(results_csv_path)
    frame = pd.read_csv(source_csv)
    validated, recall_columns, methods = _validate_results_frame(frame)
    heldout_datasets = set(validated["dataset"].unique())

    by_method_summary = _build_by_method_summary(validated, recall_columns, methods)
    macro_summary = _build_macro_summary(by_method_summary, recall_columns, methods)
    micro_summary = _build_micro_summary(validated, recall_columns, methods)
    reduction_effectiveness = _build_reduction_effectiveness(validated, recall_columns, methods)
    pairwise_overall = _build_pairwise_overall_inference(validated, recall_columns, methods)

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

    output_root = Path(output_dir) / source_csv.stem
    output_root.mkdir(parents=True, exist_ok=True)
    by_method_path = output_root / "class_heldout_by_method_summary.csv"
    macro_path = output_root / "class_heldout_macro_summary.csv"
    micro_path = output_root / "class_heldout_micro_summary.csv"
    reduction_path = output_root / "class_heldout_reduction_effectiveness.csv"
    pairwise_path = output_root / "class_heldout_pairwise_overall_inference.csv"
    interpretation_path = output_root / "class_heldout_interpretation_scaffold.md"
    transfer_path = output_root / "class_heldout_transfer_summary.csv"
    manifest_path = output_root / "manifest.txt"

    by_method_summary.to_csv(by_method_path, index=False)
    macro_summary.to_csv(macro_path, index=False)
    micro_summary.to_csv(micro_path, index=False)
    reduction_effectiveness.to_csv(reduction_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pairwise_overall.to_csv(pairwise_path, index=False)
    if transfer_summary is not None:
        transfer_summary.to_csv(transfer_path, index=False)

    _write_interpretation_scaffold(
        interpretation_path,
        source_csv=source_csv,
        macro_summary=macro_summary,
        micro_summary=micro_summary,
        pairwise_overall=pairwise_overall,
        primary_recall_column=recall_columns[-1],
        transfer_summary=transfer_summary,
        transfer_note=transfer_note,
        methods=methods,
    )
    manifest_path.write_text(
        "\n".join(
            [
                f"source_csv={source_csv}",
                f"output_dir={output_root}",
                f"selected_settings_json={selected_settings_json}" if selected_settings_json is not None else "selected_settings_json=",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts: dict[str, Path] = {
        "source_csv": source_csv,
        "output_dir": output_root,
        "class_heldout_by_method_summary": by_method_path,
        "class_heldout_macro_summary": macro_path,
        "class_heldout_micro_summary": micro_path,
        "class_heldout_reduction_effectiveness": reduction_path,
        "class_heldout_pairwise_overall_inference": pairwise_path,
        "class_heldout_interpretation_scaffold": interpretation_path,
        "manifest": manifest_path,
    }
    if transfer_summary is not None:
        artifacts["class_heldout_transfer_summary"] = transfer_path
    return artifacts
