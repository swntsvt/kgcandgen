"""Reporting utilities for held-out KG evaluation artifacts."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

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
REQUIRED_METHODS = {"tfidf", "bm25"}
REQUIRED_ENTITY_TYPES = {"class", "predicate", "instance"}


class KgHeldoutReportingValidationError(ValueError):
    """Raised when held-out reporting inputs are invalid."""


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


def _find_recall_columns(frame: pd.DataFrame) -> list[str]:
    recall_columns = [column for column in frame.columns if column.startswith("recall_at_")]
    if not recall_columns:
        raise KgHeldoutReportingValidationError(
            "Held-out KG reporting requires at least one recall_at_<k> column."
        )
    return sorted(recall_columns, key=lambda column: int(column.removeprefix("recall_at_")))


def _validate_results_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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

    methods = set(validated["method"].unique())
    if methods != REQUIRED_METHODS:
        raise KgHeldoutReportingValidationError(
            "Held-out KG reporting requires exactly these methods: "
            + ", ".join(sorted(REQUIRED_METHODS))
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
    incomplete = coverage[coverage["method_count"] != len(REQUIRED_METHODS)]
    if not incomplete.empty:
        bad = incomplete.iloc[0]
        raise KgHeldoutReportingValidationError(
            "Each dataset/entity_type pair must have one TF-IDF row and one BM25 row. "
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
    ).reset_index(drop=True), recall_columns


def _build_by_type_summary(frame: pd.DataFrame, recall_columns: list[str]) -> pd.DataFrame:
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
    return summary.sort_values(["entity_type", "method"], kind="mergesort").reset_index(drop=True)


def _delta_row(
    frame: pd.DataFrame,
    *,
    value_columns: list[str],
    method_column: str = "method",
) -> dict[str, object]:
    indexed = frame.set_index(method_column)
    if "tfidf" not in indexed.index or "bm25" not in indexed.index:
        raise KgHeldoutReportingValidationError("Expected tfidf and bm25 rows for delta computation.")

    row: dict[str, object] = {method_column: "delta_tfidf_minus_bm25"}
    for column in value_columns:
        row[column] = float(indexed.loc["tfidf", column] - indexed.loc["bm25", column])
    return row


def _build_macro_summary(by_type_summary: pd.DataFrame, recall_columns: list[str]) -> pd.DataFrame:
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
    for method in sorted(REQUIRED_METHODS):
        group = by_type_summary[by_type_summary["method"] == method].sort_values(
            ["entity_type"], kind="mergesort"
        )
        row = {"method": method, "type_count": int(len(group))}
        for column in value_columns:
            row[column] = float(group[column].mean())
        method_rows.append(row)

    summary = pd.DataFrame(method_rows)
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


def _build_micro_summary(frame: pd.DataFrame, recall_columns: list[str]) -> pd.DataFrame:
    method_rows: list[dict[str, object]] = []
    for method in sorted(REQUIRED_METHODS):
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
    summary = pd.concat(
        [summary, pd.DataFrame([_delta_row(summary, value_columns=value_columns)])],
        ignore_index=True,
    )
    return summary.sort_values(["method"], kind="mergesort").reset_index(drop=True)


def _build_reduction_effectiveness(frame: pd.DataFrame, recall_columns: list[str]) -> pd.DataFrame:
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

    paired = (
        frame.pivot(
            index=["track", "version", "dataset", "entity_type"],
            columns="method",
            values=["candidate_reduction_ratio", "mrr", *recall_columns],
        )
        .sort_index()
    )

    def paired_value(row: pd.Series, metric: str, method: str) -> float:
        return float(paired.loc[(row["track"], row["version"], row["dataset"], row["entity_type"]), (metric, method)])

    enriched["other_method"] = enriched["method"].map({"tfidf": "bm25", "bm25": "tfidf"})
    for metric in ["candidate_reduction_ratio", "mrr", *recall_columns]:
        delta_column = f"{metric}_delta_vs_other_method"
        deltas: list[float] = []
        for row in enriched.itertuples(index=False):
            current_method = str(row.method)
            current_value = paired_value(pd.Series(row._asdict()), metric, current_method)
            other_value = paired_value(
                pd.Series(row._asdict()),
                metric,
                "bm25" if current_method == "tfidf" else "tfidf",
            )
            deltas.append(float(current_value - other_value))
        enriched[delta_column] = deltas

    return enriched.sort_values(
        ["track", "dataset", "entity_type", "method"], kind="mergesort"
    ).reset_index(drop=True)


def _load_selected_settings_payload(
    selected_settings_path: Path | None,
    *,
    strict: bool,
) -> tuple[dict[str, dict[str, float]] | None, str | None]:
    if selected_settings_path is None:
        return None, "No uniquely matching selected-settings artifact was found; transfer summary skipped."

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
        for method in sorted(REQUIRED_METHODS):
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
    macro_by_method = macro_summary[macro_summary["method"].isin(REQUIRED_METHODS)].set_index("method")
    micro_by_method = micro_summary[micro_summary["method"].isin(REQUIRED_METHODS)].set_index("method")

    rows: list[dict[str, object]] = []
    for method in sorted(REQUIRED_METHODS):
        development = transfer_payload[method]
        heldout_macro_mrr = float(macro_by_method.loc[method, "mrr_mean"])
        heldout_micro_mrr = float(micro_by_method.loc[method, "mrr"])
        development_mu = float(development["development_mu"])
        rows.append(
            {
                "method": method,
                "development_selection_score": float(development["development_selection_score"]),
                "development_mu": development_mu,
                "development_sigma": float(development["development_sigma"]),
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
    primary_recall_column: str,
    transfer_summary: pd.DataFrame | None,
    transfer_note: str | None,
) -> None:
    winners: list[str] = []
    for entity_type in sorted(REQUIRED_ENTITY_TYPES):
        group = by_type_summary[by_type_summary["entity_type"] == entity_type].set_index("method")
        mrr_tfidf = float(group.loc["tfidf", "mrr_mean"])
        mrr_bm25 = float(group.loc["bm25", "mrr_mean"])
        recall_tfidf = float(group.loc["tfidf", f"{primary_recall_column}_mean"])
        recall_bm25 = float(group.loc["bm25", f"{primary_recall_column}_mean"])

        def _winner(left: float, right: float) -> str:
            if left == right:
                return "tie"
            return "tfidf" if left > right else "bm25"

        winners.append(
            f"- `{entity_type}`: MRR winner `{_winner(mrr_tfidf, mrr_bm25)}`, "
            f"{primary_recall_column} winner `{_winner(recall_tfidf, recall_bm25)}`"
        )

    macro_rows = macro_summary[macro_summary["method"].isin(REQUIRED_METHODS)].set_index("method")
    micro_rows = micro_summary[micro_summary["method"].isin(REQUIRED_METHODS)].set_index("method")
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
        "- Methods compared: `tfidf`, `bm25`",
        "",
        "## Per-Type Winners",
        "",
        *winners,
        "",
        "## Macro Summary",
        "",
        f"- TF-IDF macro MRR: `{macro_rows.loc['tfidf', 'mrr_mean']:.6f}`",
        f"- BM25 macro MRR: `{macro_rows.loc['bm25', 'mrr_mean']:.6f}`",
        "",
        "## Micro Summary",
        "",
        f"- TF-IDF micro MRR: `{micro_rows.loc['tfidf', 'mrr']:.6f}`",
        f"- BM25 micro MRR: `{micro_rows.loc['bm25', 'mrr']:.6f}`",
        "",
        "## Reduction vs Effectiveness Prompts",
        "",
        f"- Review `{primary_recall_column}` alongside `candidate_reduction_ratio` for each entity type.",
        "- Check whether the higher-reduction method also preserves MRR consistency across entity types.",
        "",
    ]

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
) -> dict[str, Path]:
    """Generate held-out KG reporting artifacts."""
    source_csv = _resolve_results_csv(results_csv_path)
    frame = pd.read_csv(source_csv)
    validated, recall_columns = _validate_results_frame(frame)
    heldout_datasets = set(validated["dataset"].unique())

    by_type_summary = _build_by_type_summary(validated, recall_columns)
    macro_summary = _build_macro_summary(by_type_summary, recall_columns)
    micro_summary = _build_micro_summary(validated, recall_columns)
    reduction_effectiveness = _build_reduction_effectiveness(validated, recall_columns)

    selected_settings_json, selected_settings_explicit = _resolve_selected_settings_json(
        selected_settings_path,
        heldout_datasets=heldout_datasets,
    )
    transfer_payload, transfer_note = _load_selected_settings_payload(
        selected_settings_json,
        strict=selected_settings_explicit,
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
    interpretation_path = output_root / "kg_heldout_interpretation_scaffold.md"
    transfer_path = output_root / "kg_heldout_transfer_summary.csv"
    manifest_path = output_root / "manifest.txt"

    by_type_summary.to_csv(by_type_path, index=False)
    macro_summary.to_csv(macro_path, index=False)
    micro_summary.to_csv(micro_path, index=False)
    reduction_effectiveness.to_csv(reduction_path, index=False, quoting=csv.QUOTE_MINIMAL)
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
        primary_recall_column=primary_recall_column,
        transfer_summary=transfer_summary,
        transfer_note=transfer_note,
    )

    manifest_lines = [
        f"generated_at={datetime.now().isoformat(timespec='seconds')}",
        f"source_csv={source_csv}",
        f"output_dir={output_root}",
        f"selected_settings_json={selected_settings_json}" if selected_settings_json is not None else "selected_settings_json=",
    ]
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    artifacts: dict[str, Path] = {
        "source_csv": source_csv,
        "output_dir": output_root,
        "kg_heldout_by_type_summary": by_type_path,
        "kg_heldout_macro_summary": macro_path,
        "kg_heldout_micro_summary": micro_path,
        "kg_heldout_reduction_effectiveness": reduction_path,
        "kg_heldout_interpretation_scaffold": interpretation_path,
        "manifest": manifest_path,
    }
    if transfer_summary is not None:
        artifacts["kg_heldout_transfer_summary"] = transfer_path
    return artifacts
