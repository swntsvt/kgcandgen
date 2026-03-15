"""TF-IDF hyperparameter sensitivity and failure-surface analysis."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from src.analysis.plot_env import configure_plot_environment

configure_plot_environment()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config_loader import load_runtime_config

REQUIRED_COLUMNS = {
    "dataset",
    "track",
    "method",
    "hyperparameters",
    "mrr",
    "recall_at_10",
    "recall_at_50",
}


class TfidfSensitivityValidationError(ValueError):
    """Raised when TF-IDF sensitivity input data is invalid."""


def _resolve_results_csv(results_csv_path: str | Path | None) -> Path:
    if results_csv_path is not None:
        path = Path(results_csv_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Results CSV not found: {path}")
        return path

    candidates = sorted(
        Path("results").glob("result_*.csv"),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No results/result_*.csv files found for TF-IDF sensitivity.")
    return candidates[-1].resolve()


def _normalize_df_threshold(value: Any, *, field: str) -> int | float:
    if isinstance(value, bool):
        raise TfidfSensitivityValidationError(f"hyperparameters.{field} must be numeric, not bool.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return value
    raise TfidfSensitivityValidationError(f"hyperparameters.{field} must be numeric.")


def _parse_tfidf_hyperparameters(value: str) -> dict[str, Any]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise TfidfSensitivityValidationError(
            f"Malformed TF-IDF hyperparameters JSON: {value}"
        ) from exc

    if not isinstance(data, dict):
        raise TfidfSensitivityValidationError(
            f"TF-IDF hyperparameters must be a JSON object: {value}"
        )

    required = {"ngram_range", "min_df", "max_df", "sublinear_tf"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise TfidfSensitivityValidationError(
            f"TF-IDF hyperparameters missing key(s): {', '.join(missing)}"
        )

    raw_ngram = data["ngram_range"]
    if (
        not isinstance(raw_ngram, (list, tuple))
        or len(raw_ngram) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in raw_ngram)
    ):
        raise TfidfSensitivityValidationError(
            "hyperparameters.ngram_range must be a two-int list/tuple."
        )

    sublinear_tf = data["sublinear_tf"]
    if not isinstance(sublinear_tf, bool):
        raise TfidfSensitivityValidationError("hyperparameters.sublinear_tf must be boolean.")

    return {
        "ngram_range": (int(raw_ngram[0]), int(raw_ngram[1])),
        "min_df": _normalize_df_threshold(data["min_df"], field="min_df"),
        "max_df": _normalize_df_threshold(data["max_df"], field="max_df"),
        "sublinear_tf": sublinear_tf,
    }


def _ordered_unique(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _ngram_label(ngram_range: tuple[int, int]) -> str:
    return f"{ngram_range[0]}-{ngram_range[1]}"


def _df_label(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:g}"


def _prepare_expected_frame(
    *,
    datasets_in_scope: pd.DataFrame,
    config_path: str | Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    runtime_config = load_runtime_config(config_path=config_path)
    tfidf_grid = runtime_config.experiments.tfidf_grid
    if not tfidf_grid:
        raise TfidfSensitivityValidationError("Config experiments.tfidf_grid must be non-empty.")

    grid_records: list[dict[str, Any]] = []
    for index, grid_entry in enumerate(tfidf_grid):
        grid_records.append(
            {
                "grid_index": index,
                "ngram_range": tuple(grid_entry.ngram_range),
                "min_df": grid_entry.min_df,
                "max_df": grid_entry.max_df,
                "sublinear_tf": grid_entry.sublinear_tf,
                "ngram_label": _ngram_label(tuple(grid_entry.ngram_range)),
                "min_df_label": _df_label(grid_entry.min_df),
            }
        )

    expected_rows: list[dict[str, Any]] = []
    dataset_track_rows = datasets_in_scope[["dataset", "track"]].drop_duplicates()
    for row in dataset_track_rows.itertuples(index=False):
        for record in grid_records:
            expected_rows.append(
                {
                    "dataset": str(row.dataset),
                    "track": str(row.track),
                    **record,
                }
            )

    expected_frame = pd.DataFrame(expected_rows)
    ngram_order = _ordered_unique([record["ngram_label"] for record in grid_records])
    min_df_order = _ordered_unique([record["min_df_label"] for record in grid_records])
    return expected_frame, ngram_order, min_df_order


def _prepare_observed_tfidf_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise TfidfSensitivityValidationError(
            "Missing required TF-IDF sensitivity column(s): " + ", ".join(missing_columns)
        )

    tfidf = frame[frame["method"].astype(str) == "tfidf"].copy()
    if tfidf.empty:
        raise TfidfSensitivityValidationError("No TF-IDF rows found in results CSV.")

    parsed = tfidf["hyperparameters"].astype(str).apply(_parse_tfidf_hyperparameters)
    tfidf["ngram_range"] = parsed.apply(lambda value: value["ngram_range"])
    tfidf["min_df"] = parsed.apply(lambda value: value["min_df"])
    tfidf["max_df"] = parsed.apply(lambda value: value["max_df"])
    tfidf["sublinear_tf"] = parsed.apply(lambda value: value["sublinear_tf"])
    tfidf["ngram_label"] = tfidf["ngram_range"].apply(_ngram_label)
    tfidf["min_df_label"] = tfidf["min_df"].apply(_df_label)

    tfidf["dataset"] = tfidf["dataset"].astype(str)
    tfidf["track"] = tfidf["track"].astype(str)
    tfidf["mrr"] = tfidf["mrr"].astype(float)
    tfidf["recall_at_10"] = tfidf["recall_at_10"].astype(float)
    tfidf["recall_at_50"] = tfidf["recall_at_50"].astype(float)
    return tfidf


def _merge_expected_with_observed(
    expected_frame: pd.DataFrame,
    observed_frame: pd.DataFrame,
) -> pd.DataFrame:
    merge_keys = [
        "dataset",
        "track",
        "ngram_range",
        "min_df",
        "max_df",
        "sublinear_tf",
    ]
    observed_subset = observed_frame[
        merge_keys + ["mrr", "recall_at_10", "recall_at_50", "runtime_seconds"]
    ].drop_duplicates(subset=merge_keys, keep="first")

    combined = expected_frame.merge(
        observed_subset,
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    combined["status"] = combined["mrr"].apply(lambda value: "success" if pd.notna(value) else "failure")
    combined["failure_reason"] = combined["status"].apply(
        lambda status: "" if status == "success" else "missing_in_results"
    )
    combined["high_order"] = combined["ngram_range"].apply(lambda value: int(value[1]) >= 2)
    return combined


def _build_combo_summary(combined: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["ngram_label", "min_df_label", "sublinear_tf"]
    grouped = combined.groupby(group_cols, as_index=False)
    rows: list[dict[str, Any]] = []

    for keys, group in grouped:
        ngram_label, min_df_label, sublinear_tf = keys
        success_group = group[group["status"] == "success"]
        success_count = int(len(success_group))
        total_count = int(len(group))
        failure_count = total_count - success_count
        row = {
            "ngram_range": ngram_label,
            "min_df": min_df_label,
            "sublinear_tf": bool(sublinear_tf),
            "success_count": success_count,
            "failure_count": failure_count,
            "failure_rate": float(failure_count / total_count) if total_count else 0.0,
            "mrr_mean": float(success_group["mrr"].mean()) if success_count else float("nan"),
            "mrr_median": float(success_group["mrr"].median()) if success_count else float("nan"),
            "mrr_std": float(success_group["mrr"].std()) if success_count > 1 else 0.0 if success_count == 1 else float("nan"),
            "recall_at_10_mean": float(success_group["recall_at_10"].mean()) if success_count else float("nan"),
            "recall_at_10_median": float(success_group["recall_at_10"].median()) if success_count else float("nan"),
            "recall_at_10_std": float(success_group["recall_at_10"].std()) if success_count > 1 else 0.0 if success_count == 1 else float("nan"),
            "recall_at_50_mean": float(success_group["recall_at_50"].mean()) if success_count else float("nan"),
            "recall_at_50_median": float(success_group["recall_at_50"].median()) if success_count else float("nan"),
            "recall_at_50_std": float(success_group["recall_at_50"].std()) if success_count > 1 else 0.0 if success_count == 1 else float("nan"),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["sublinear_tf", "ngram_range", "min_df"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_track_summary(combined: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["track", "ngram_label", "min_df_label", "sublinear_tf"]
    grouped = combined.groupby(group_cols, as_index=False)
    rows: list[dict[str, Any]] = []

    for keys, group in grouped:
        track, ngram_label, min_df_label, sublinear_tf = keys
        success_group = group[group["status"] == "success"]
        success_count = int(len(success_group))
        total_count = int(len(group))
        failure_count = total_count - success_count
        rows.append(
            {
                "track": str(track),
                "ngram_range": ngram_label,
                "min_df": min_df_label,
                "sublinear_tf": bool(sublinear_tf),
                "success_count": success_count,
                "failure_count": failure_count,
                "failure_rate": float(failure_count / total_count) if total_count else 0.0,
                "mrr_mean": float(success_group["mrr"].mean()) if success_count else float("nan"),
                "recall_at_10_mean": float(success_group["recall_at_10"].mean()) if success_count else float("nan"),
                "recall_at_50_mean": float(success_group["recall_at_50"].mean()) if success_count else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["track", "sublinear_tf", "ngram_range", "min_df"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_interaction_summary(combined: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["ngram_label", "min_df_label"]
    grouped = combined.groupby(group_cols, as_index=False)
    rows: list[dict[str, Any]] = []

    for keys, group in grouped:
        ngram_label, min_df_label = keys
        success_group = group[group["status"] == "success"]
        success_count = int(len(success_group))
        total_count = int(len(group))
        failure_count = total_count - success_count
        rows.append(
            {
                "ngram_range": ngram_label,
                "min_df": min_df_label,
                "high_order": bool(max(int(part) for part in ngram_label.split("-")) >= 2),
                "success_count": success_count,
                "failure_count": failure_count,
                "failure_rate": float(failure_count / total_count) if total_count else 0.0,
                "mrr_mean": float(success_group["mrr"].mean()) if success_count else float("nan"),
                "recall_at_10_mean": float(success_group["recall_at_10"].mean()) if success_count else float("nan"),
                "recall_at_50_mean": float(success_group["recall_at_50"].mean()) if success_count else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["ngram_range", "min_df"],
        kind="mergesort",
    ).reset_index(drop=True)


def _draw_mrr_heatmaps(
    combo_summary: pd.DataFrame,
    *,
    output_root: Path,
    ngram_order: list[str],
    min_df_order: list[str],
) -> None:
    for sublinear_value in (False, True):
        subset = combo_summary[combo_summary["sublinear_tf"] == sublinear_value]
        pivoted = (
            subset.pivot(index="ngram_range", columns="min_df", values="mrr_mean")
            .reindex(index=ngram_order, columns=min_df_order)
        )

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        if pivoted.notna().any().any():
            sns.heatmap(
                pivoted,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                linewidths=0.5,
                linecolor="white",
                ax=ax,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No successful runs",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("min_df")
        ax.set_ylabel("ngram_range")
        ax.set_title(f"TF-IDF Mean MRR (sublinear_tf={sublinear_value})")
        fig.tight_layout()

        base = output_root / f"tfidf_mrr_heatmap_sublinear_{str(sublinear_value).lower()}"
        fig.savefig(base.with_suffix(".png"), dpi=300)
        fig.savefig(base.with_suffix(".pdf"))
        plt.close(fig)


def _draw_failure_rate_heatmap(
    combo_summary: pd.DataFrame,
    *,
    output_root: Path,
    ngram_order: list[str],
    min_df_order: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)
    for index, sublinear_value in enumerate((False, True)):
        subset = combo_summary[combo_summary["sublinear_tf"] == sublinear_value]
        pivoted = (
            subset.pivot(index="ngram_range", columns="min_df", values="failure_rate")
            .reindex(index=ngram_order, columns=min_df_order)
        )
        sns.heatmap(
            pivoted,
            annot=True,
            fmt=".2f",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5,
            linecolor="white",
            ax=axes[index],
        )
        axes[index].set_xlabel("min_df")
        axes[index].set_title(f"Failure Rate (sublinear_tf={sublinear_value})")
        if index == 0:
            axes[index].set_ylabel("ngram_range")
        else:
            axes[index].set_ylabel("")

    fig.suptitle("TF-IDF Failure Surface by Hyperparameter Combination", y=1.02)
    fig.tight_layout()
    base = output_root / "tfidf_failure_rate_heatmap"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _aggregate_possible_why(combined: pd.DataFrame) -> dict[str, float]:
    by_high_order = combined.groupby("high_order")
    high_order_failure_rate = float(
        by_high_order.get_group(True)["status"].eq("failure").mean()
    ) if True in by_high_order.groups else float("nan")
    unigram_failure_rate = float(
        by_high_order.get_group(False)["status"].eq("failure").mean()
    ) if False in by_high_order.groups else float("nan")

    by_min_df = combined.groupby("min_df")
    min_df_1 = by_min_df.get_group(1) if 1 in by_min_df.groups else pd.DataFrame()
    min_df_2 = by_min_df.get_group(2) if 2 in by_min_df.groups else pd.DataFrame()
    min_df_1_failure = float(min_df_1["status"].eq("failure").mean()) if not min_df_1.empty else float("nan")
    min_df_2_failure = float(min_df_2["status"].eq("failure").mean()) if not min_df_2.empty else float("nan")

    min_df_1_success = min_df_1[min_df_1["status"] == "success"]
    min_df_2_success = min_df_2[min_df_2["status"] == "success"]

    min_df_1_mrr = float(min_df_1_success["mrr"].mean()) if not min_df_1_success.empty else float("nan")
    min_df_2_mrr = float(min_df_2_success["mrr"].mean()) if not min_df_2_success.empty else float("nan")

    return {
        "high_order_failure_rate": high_order_failure_rate,
        "unigram_failure_rate": unigram_failure_rate,
        "min_df_1_failure_rate": min_df_1_failure,
        "min_df_2_failure_rate": min_df_2_failure,
        "min_df_1_success_mrr_mean": min_df_1_mrr,
        "min_df_2_success_mrr_mean": min_df_2_mrr,
    }


def _write_interpretation_scaffold(
    output_path: Path,
    *,
    source_csv: Path,
    combined: pd.DataFrame,
) -> None:
    computed = _aggregate_possible_why(combined)
    lines = [
        "# TF-IDF Sensitivity Interpretation Scaffold",
        "",
        f"Source results CSV: `{source_csv}`",
        "",
        "## What Happened",
        "- Summarize whether MRR was primarily driven by `ngram_range`, `min_df`, or `sublinear_tf`.",
        "- Note whether the top-performing region was stable across tracks or concentrated in one track.",
        "",
        "## Failure Surface",
        f"- High-order n-gram failure rate (`max_n>=2`): {computed['high_order_failure_rate']:.4f}",
        f"- Unigram-only failure rate (`1-1`): {computed['unigram_failure_rate']:.4f}",
        f"- `min_df=1` failure rate: {computed['min_df_1_failure_rate']:.4f}",
        f"- `min_df=2` failure rate: {computed['min_df_2_failure_rate']:.4f}",
        "",
        "## Possible Why",
        (
            "- If high-order n-gram failure rate is notably higher than unigram failure rate, "
            "the corpus likely becomes too sparse under stricter n-gram construction."
        ),
        (
            "- If `min_df=2` failure rate exceeds `min_df=1`, pruning likely removes too many "
            "terms for small/heterogeneous ontology pairs."
        ),
        (
            f"- Success-only MRR means: min_df=1 => {computed['min_df_1_success_mrr_mean']:.4f}, "
            f"min_df=2 => {computed['min_df_2_success_mrr_mean']:.4f}."
        ),
        "- Use these trends with `tfidf_interaction_summary.csv` to justify robustness claims.",
        "",
        "## Fill-In Prompts",
        "- Robust region claim: \"The most robust TF-IDF settings were ... because ...\"",
        "- Unstable region claim: \"The unstable region was ... with failure rate ... due to ...\"",
        "- Practical recommendation: \"For default TF-IDF in this benchmark, use ...\"",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_tfidf_sensitivity(
    results_csv_path: str | Path | None,
    config_path: str | Path = "config/datasets.yaml",
    output_dir: str | Path = "results/comparisons",
) -> dict[str, Path]:
    """Generate TF-IDF sensitivity summaries and charts."""
    source_csv = _resolve_results_csv(results_csv_path)
    frame = pd.read_csv(source_csv)
    observed_tfidf = _prepare_observed_tfidf_frame(frame)

    datasets_in_scope = observed_tfidf[["dataset", "track"]].drop_duplicates()
    expected_frame, ngram_order, min_df_order = _prepare_expected_frame(
        datasets_in_scope=datasets_in_scope,
        config_path=config_path,
    )
    combined = _merge_expected_with_observed(expected_frame, observed_tfidf)

    combo_summary = _build_combo_summary(combined)
    track_summary = _build_track_summary(combined)
    interaction_summary = _build_interaction_summary(combined)
    failure_records = combined[combined["status"] == "failure"].copy().sort_values(
        ["track", "dataset", "sublinear_tf", "ngram_label", "min_df_label"],
        kind="mergesort",
    )

    run_id = source_csv.stem
    output_root = Path(output_dir) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "tfidf_sensitivity_summary.csv"
    by_track_path = output_root / "tfidf_sensitivity_by_track.csv"
    interaction_path = output_root / "tfidf_interaction_summary.csv"
    failure_records_path = output_root / "tfidf_failure_records.csv"
    interpretation_path = output_root / "tfidf_sensitivity_interpretation.md"

    combo_summary.to_csv(summary_path, index=False)
    track_summary.to_csv(by_track_path, index=False)
    interaction_summary.to_csv(interaction_path, index=False)
    failure_records.to_csv(failure_records_path, index=False)

    _draw_mrr_heatmaps(
        combo_summary,
        output_root=output_root,
        ngram_order=ngram_order,
        min_df_order=min_df_order,
    )
    _draw_failure_rate_heatmap(
        combo_summary,
        output_root=output_root,
        ngram_order=ngram_order,
        min_df_order=min_df_order,
    )
    _write_interpretation_scaffold(
        interpretation_path,
        source_csv=source_csv,
        combined=combined,
    )

    generated_at = datetime.now().isoformat(timespec="seconds")
    manifest_path = output_root / "tfidf_sensitivity_manifest.txt"
    manifest_path.write_text(
        "\n".join(
            [
                f"generated_at={generated_at}",
                f"source_csv={source_csv}",
                f"config_path={Path(config_path).resolve()}",
                f"output_dir={output_root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "source_csv": source_csv,
        "output_dir": output_root,
        "tfidf_sensitivity_summary": summary_path,
        "tfidf_sensitivity_by_track": by_track_path,
        "tfidf_interaction_summary": interaction_path,
        "tfidf_failure_records": failure_records_path,
        "tfidf_mrr_heatmap_sublinear_false_png": output_root / "tfidf_mrr_heatmap_sublinear_false.png",
        "tfidf_mrr_heatmap_sublinear_false_pdf": output_root / "tfidf_mrr_heatmap_sublinear_false.pdf",
        "tfidf_mrr_heatmap_sublinear_true_png": output_root / "tfidf_mrr_heatmap_sublinear_true.png",
        "tfidf_mrr_heatmap_sublinear_true_pdf": output_root / "tfidf_mrr_heatmap_sublinear_true.pdf",
        "tfidf_failure_rate_heatmap_png": output_root / "tfidf_failure_rate_heatmap.png",
        "tfidf_failure_rate_heatmap_pdf": output_root / "tfidf_failure_rate_heatmap.pdf",
        "tfidf_sensitivity_interpretation": interpretation_path,
        "manifest": manifest_path,
    }
