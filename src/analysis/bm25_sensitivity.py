"""BM25 hyperparameter sensitivity and stability analysis."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

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
BOOTSTRAP_SEED = 42
BOOTSTRAP_SAMPLES = 1000


class Bm25SensitivityValidationError(ValueError):
    """Raised when BM25 sensitivity input data is invalid."""


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
        raise FileNotFoundError("No results/result_*.csv files found for BM25 sensitivity.")
    return candidates[-1].resolve()


def _ordered_unique(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _float_label(value: float) -> str:
    return f"{value:g}"


def _parse_bm25_hyperparameters(value: str) -> dict[str, float]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise Bm25SensitivityValidationError(
            f"Malformed BM25 hyperparameters JSON: {value}"
        ) from exc

    if not isinstance(data, dict):
        raise Bm25SensitivityValidationError(
            f"BM25 hyperparameters must be a JSON object: {value}"
        )

    required = {"k1", "b"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise Bm25SensitivityValidationError(
            f"BM25 hyperparameters missing key(s): {', '.join(missing)}"
        )

    k1 = data["k1"]
    b = data["b"]
    if isinstance(k1, bool) or not isinstance(k1, (int, float)):
        raise Bm25SensitivityValidationError("hyperparameters.k1 must be numeric.")
    if isinstance(b, bool) or not isinstance(b, (int, float)):
        raise Bm25SensitivityValidationError("hyperparameters.b must be numeric.")

    return {
        "k1": float(k1),
        "b": float(b),
    }


def _prepare_observed_bm25_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise Bm25SensitivityValidationError(
            "Missing required BM25 sensitivity column(s): " + ", ".join(missing_columns)
        )

    bm25 = frame[frame["method"].astype(str) == "bm25"].copy()
    if bm25.empty:
        raise Bm25SensitivityValidationError("No BM25 rows found in results CSV.")

    parsed = bm25["hyperparameters"].astype(str).apply(_parse_bm25_hyperparameters)
    bm25["k1"] = parsed.apply(lambda value: value["k1"])
    bm25["b"] = parsed.apply(lambda value: value["b"])
    bm25["k1_label"] = bm25["k1"].apply(_float_label)
    bm25["b_label"] = bm25["b"].apply(_float_label)

    bm25["dataset"] = bm25["dataset"].astype(str)
    bm25["track"] = bm25["track"].astype(str)
    bm25["mrr"] = bm25["mrr"].astype(float)
    bm25["recall_at_10"] = bm25["recall_at_10"].astype(float)
    bm25["recall_at_50"] = bm25["recall_at_50"].astype(float)
    return bm25


def _prepare_expected_frame(
    *,
    datasets_in_scope: pd.DataFrame,
    config_path: str | Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    runtime_config = load_runtime_config(config_path=config_path)
    bm25_grid = runtime_config.experiments.bm25_grid
    if not bm25_grid:
        raise Bm25SensitivityValidationError("Config experiments.bm25_grid must be non-empty.")

    grid_records: list[dict[str, Any]] = []
    for index, grid_entry in enumerate(bm25_grid):
        grid_records.append(
            {
                "grid_index": index,
                "k1": float(grid_entry.k1),
                "b": float(grid_entry.b),
                "k1_label": _float_label(float(grid_entry.k1)),
                "b_label": _float_label(float(grid_entry.b)),
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
    k1_order = _ordered_unique([record["k1_label"] for record in grid_records])
    b_order = _ordered_unique([record["b_label"] for record in grid_records])
    return expected_frame, k1_order, b_order


def _merge_expected_with_observed(
    expected_frame: pd.DataFrame,
    observed_frame: pd.DataFrame,
) -> pd.DataFrame:
    merge_keys = ["dataset", "track", "k1", "b"]
    observed_subset = observed_frame[
        merge_keys + ["mrr", "recall_at_10", "recall_at_50", "runtime_seconds"]
    ].drop_duplicates(subset=merge_keys, keep="first")

    combined = expected_frame.merge(
        observed_subset,
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    combined["status"] = combined["mrr"].apply(
        lambda value: "success" if pd.notna(value) else "failure"
    )
    combined["failure_reason"] = combined["status"].apply(
        lambda status: "" if status == "success" else "missing_in_results"
    )
    return combined


def _build_surface_summary(combined: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["k1_label", "b_label"]
    grouped = combined.groupby(group_cols, as_index=False)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        k1_label, b_label = keys
        success_group = group[group["status"] == "success"]
        success_count = int(len(success_group))
        total_count = int(len(group))
        failure_count = total_count - success_count
        rows.append(
            {
                "k1": k1_label,
                "b": b_label,
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
        )
    return pd.DataFrame(rows).sort_values(["k1", "b"], kind="mergesort").reset_index(drop=True)


def _build_surface_by_track(combined: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["track", "k1_label", "b_label"]
    grouped = combined.groupby(group_cols, as_index=False)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        track, k1_label, b_label = keys
        success_group = group[group["status"] == "success"]
        success_count = int(len(success_group))
        total_count = int(len(group))
        failure_count = total_count - success_count
        rows.append(
            {
                "track": str(track),
                "k1": k1_label,
                "b": b_label,
                "success_count": success_count,
                "failure_count": failure_count,
                "failure_rate": float(failure_count / total_count) if total_count else 0.0,
                "mrr_mean": float(success_group["mrr"].mean()) if success_count else float("nan"),
                "recall_at_10_mean": float(success_group["recall_at_10"].mean()) if success_count else float("nan"),
                "recall_at_50_mean": float(success_group["recall_at_50"].mean()) if success_count else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["track", "k1", "b"], kind="mergesort"
    ).reset_index(drop=True)


def _build_dataset_stability(combined: pd.DataFrame) -> pd.DataFrame:
    success = combined[combined["status"] == "success"]
    grouped = success.groupby(["track", "dataset"], as_index=False)
    rows: list[dict[str, Any]] = []
    for _, group in grouped:
        row = group.iloc[0]
        rows.append(
            {
                "track": str(row["track"]),
                "dataset": str(row["dataset"]),
                "mrr_range": float(group["mrr"].max() - group["mrr"].min()),
                "recall_at_10_range": float(group["recall_at_10"].max() - group["recall_at_10"].min()),
                "recall_at_50_range": float(group["recall_at_50"].max() - group["recall_at_50"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["track", "dataset"], kind="mergesort").reset_index(drop=True)


def _build_track_stability(surface_by_track: pd.DataFrame) -> pd.DataFrame:
    grouped = surface_by_track.groupby("track", as_index=False)
    rows: list[dict[str, Any]] = []
    for _, group in grouped:
        track = str(group.iloc[0]["track"])
        datasets_in_track = int(group["success_count"].max()) if not group.empty else 0
        rows.append(
            {
                "track": track,
                "datasets_in_track": datasets_in_track,
                "mrr_variance": float(group["mrr_mean"].var(ddof=0)),
                "mrr_std": float(group["mrr_mean"].std(ddof=0)),
                "recall_at_10_variance": float(group["recall_at_10_mean"].var(ddof=0)),
                "recall_at_10_std": float(group["recall_at_10_mean"].std(ddof=0)),
                "recall_at_50_variance": float(group["recall_at_50_mean"].var(ddof=0)),
                "recall_at_50_std": float(group["recall_at_50_mean"].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["track"], kind="mergesort").reset_index(drop=True)


def _bootstrap_ci_mean(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    n = len(array)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(array[0]), float(array[0])

    rng = pd.Series(range(BOOTSTRAP_SAMPLES))
    random_state = BOOTSTRAP_SEED
    sample_means: list[float] = []
    for iteration in rng:
        sample = pd.Series(array).sample(
            n=n, replace=True, random_state=random_state + int(iteration)
        )
        sample_means.append(float(sample.mean()))
    lower = float(pd.Series(sample_means).quantile(0.025))
    upper = float(pd.Series(sample_means).quantile(0.975))
    return lower, upper


def _build_top_settings_with_ci(combined: pd.DataFrame) -> pd.DataFrame:
    success = combined[combined["status"] == "success"]
    grouped = success.groupby(["k1_label", "b_label"], as_index=False)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        k1_label, b_label = keys
        ci_low, ci_high = _bootstrap_ci_mean(group["mrr"])
        rows.append(
            {
                "k1": k1_label,
                "b": b_label,
                "datasets_covered": int(len(group)),
                "mrr_mean": float(group["mrr"].mean()),
                "mrr_ci_lower": ci_low,
                "mrr_ci_upper": ci_high,
                "recall_at_10_mean": float(group["recall_at_10"].mean()),
                "recall_at_50_mean": float(group["recall_at_50"].mean()),
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["mrr_mean", "k1", "b"], ascending=[False, True, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    return ranked


def _draw_overall_mrr_heatmap(
    summary: pd.DataFrame, *, output_root: Path, k1_order: list[str], b_order: list[str]
) -> None:
    pivoted = summary.pivot(index="k1", columns="b", values="mrr_mean").reindex(
        index=k1_order, columns=b_order
    )
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.heatmap(
        pivoted,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("b")
    ax.set_ylabel("k1")
    ax.set_title("BM25 Mean MRR Surface (Overall)")
    fig.tight_layout()
    base = output_root / "bm25_mrr_heatmap_overall"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _draw_track_mrr_heatmaps(
    by_track: pd.DataFrame, *, output_root: Path, k1_order: list[str], b_order: list[str]
) -> None:
    tracks = sorted(by_track["track"].unique().tolist())
    fig, axes = plt.subplots(1, len(tracks), figsize=(6 * len(tracks), 5), dpi=300, sharey=True)
    if len(tracks) == 1:
        axes = [axes]
    for index, track in enumerate(tracks):
        subset = by_track[by_track["track"] == track]
        pivoted = subset.pivot(index="k1", columns="b", values="mrr_mean").reindex(
            index=k1_order, columns=b_order
        )
        sns.heatmap(
            pivoted,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.5,
            linecolor="white",
            ax=axes[index],
        )
        axes[index].set_title(f"{track}")
        axes[index].set_xlabel("b")
        if index == 0:
            axes[index].set_ylabel("k1")
        else:
            axes[index].set_ylabel("")
    fig.suptitle("BM25 Mean MRR Surface by Track", y=1.02)
    fig.tight_layout()
    base = output_root / "bm25_mrr_heatmap_by_track"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _draw_mrr_profiles_by_b(summary: pd.DataFrame, *, output_root: Path, k1_order: list[str], b_order: list[str]) -> None:
    plot_data = summary.copy()
    plot_data["k1"] = pd.Categorical(plot_data["k1"], categories=k1_order, ordered=True)
    plot_data["b"] = pd.Categorical(plot_data["b"], categories=b_order, ordered=True)
    plot_data = plot_data.sort_values(["b", "k1"], kind="mergesort")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.lineplot(
        data=plot_data,
        x="k1",
        y="mrr_mean",
        hue="b",
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("k1")
    ax.set_ylabel("Mean MRR")
    ax.set_title("BM25 MRR Profiles by Fixed b")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="b")
    fig.tight_layout()
    base = output_root / "bm25_mrr_profiles_by_b"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _draw_failure_rate_heatmap(
    summary: pd.DataFrame, *, output_root: Path, k1_order: list[str], b_order: list[str]
) -> None:
    pivoted = summary.pivot(index="k1", columns="b", values="failure_rate").reindex(
        index=k1_order, columns=b_order
    )
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    sns.heatmap(
        pivoted,
        annot=True,
        fmt=".2f",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("b")
    ax.set_ylabel("k1")
    ax.set_title("BM25 Failure Rate Surface")
    fig.tight_layout()
    base = output_root / "bm25_failure_rate_heatmap"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _write_interpretation_scaffold(
    output_path: Path,
    *,
    source_csv: Path,
    summary: pd.DataFrame,
    track_stability: pd.DataFrame,
    top_settings: pd.DataFrame,
) -> None:
    best_row = top_settings.iloc[0] if not top_settings.empty else None
    most_stable_track = (
        track_stability.sort_values("mrr_std", kind="mergesort").iloc[0]
        if not track_stability.empty
        else None
    )
    most_sensitive_track = (
        track_stability.sort_values("mrr_std", ascending=False, kind="mergesort").iloc[0]
        if not track_stability.empty
        else None
    )
    ci_overlap_with_rank2 = False
    if len(top_settings) >= 2:
        rank1 = top_settings.iloc[0]
        rank2 = top_settings.iloc[1]
        ci_overlap_with_rank2 = not (
            float(rank2["mrr_ci_upper"]) < float(rank1["mrr_ci_lower"])
            or float(rank2["mrr_ci_lower"]) > float(rank1["mrr_ci_upper"])
        )
    lines = [
        "# BM25 Sensitivity Interpretation Scaffold",
        "",
        f"Source results CSV: `{source_csv}`",
        "",
        "## What Happened",
    ]
    if best_row is not None:
        lines.append(
            f"- Top setting by mean MRR: (k1={best_row['k1']}, b={best_row['b']}) "
            f"with mean MRR={float(best_row['mrr_mean']):.4f} "
            f"[95% CI: {float(best_row['mrr_ci_lower']):.4f}, {float(best_row['mrr_ci_upper']):.4f}]."
        )
    lines.extend(
        [
            "- Summarize whether high-k1 settings produced a robust plateau or a sharp optimum.",
            "- Summarize whether b-normalization effects were consistent across tracks.",
            "",
            "## Stability Signals",
        ]
    )
    if most_stable_track is not None and most_sensitive_track is not None:
        lines.append(
            f"- Most stable track by MRR std: {most_stable_track['track']} "
            f"(std={float(most_stable_track['mrr_std']):.4f}, "
            f"datasets={int(most_stable_track['datasets_in_track'])})."
        )
        lines.append(
            f"- Most sensitive track by MRR std: {most_sensitive_track['track']} "
            f"(std={float(most_sensitive_track['mrr_std']):.4f}, "
            f"datasets={int(most_sensitive_track['datasets_in_track'])})."
        )
        lines.append(
            "- Caution: stability comparisons across tracks should account for "
            "dataset-count imbalance (smaller tracks can appear artificially stable)."
        )
    if best_row is not None:
        lines.append(
            "- Top-setting confidence profile: "
            + (
                "rank-1 and rank-2 CIs overlap, suggesting a plateau rather than a single sharp optimum."
                if ci_overlap_with_rank2
                else "rank-1 CI is separated from rank-2, suggesting a sharper optimum."
            )
        )
    lines.extend(
        [
            "",
            "## Possible Why",
            "- If MRR rises with k1 then plateaus, term-frequency saturation may be stabilizing relevance signals.",
            "- If extreme b values underperform, length normalization may be either too weak (b near 0) or too strong (b near 1).",
            "- Prefer regions where mean MRR is high and neighboring settings show similar scores/failure rates.",
            "",
            "## Fill-In Prompts",
            "- Generalization claim: \"The BM25 region that generalizes best is ... because ...\"",
            "- Sensitivity claim: \"The most sensitive region is ... where small k1/b changes caused ...\"",
            "- Default recommendation: \"For this benchmark, choose k1=... and b=... as default.\"",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_bm25_sensitivity(
    results_csv_path: str | Path | None,
    config_path: str | Path = "config/datasets.yaml",
    output_dir: str | Path = "results/comparisons",
) -> dict[str, Path]:
    """Generate BM25 sensitivity summaries, stability outputs, and charts."""
    source_csv = _resolve_results_csv(results_csv_path)
    frame = pd.read_csv(source_csv)
    observed_bm25 = _prepare_observed_bm25_frame(frame)

    datasets_in_scope = observed_bm25[["dataset", "track"]].drop_duplicates()
    expected_frame, k1_order, b_order = _prepare_expected_frame(
        datasets_in_scope=datasets_in_scope,
        config_path=config_path,
    )
    combined = _merge_expected_with_observed(expected_frame, observed_bm25)

    summary = _build_surface_summary(combined)
    by_track = _build_surface_by_track(combined)
    dataset_stability = _build_dataset_stability(combined)
    track_stability = _build_track_stability(by_track)
    top_settings = _build_top_settings_with_ci(combined)
    failure_records = combined[combined["status"] == "failure"].copy().sort_values(
        ["track", "dataset", "k1", "b"], kind="mergesort"
    )

    run_id = source_csv.stem
    output_root = Path(output_dir) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "bm25_sensitivity_summary.csv"
    by_track_path = output_root / "bm25_sensitivity_by_track.csv"
    dataset_stability_path = output_root / "bm25_dataset_stability.csv"
    track_stability_path = output_root / "bm25_track_stability.csv"
    top_settings_path = output_root / "bm25_top_settings_with_ci.csv"
    failure_records_path = output_root / "bm25_failure_records.csv"
    interpretation_path = output_root / "bm25_sensitivity_interpretation.md"

    summary.to_csv(summary_path, index=False)
    by_track.to_csv(by_track_path, index=False)
    dataset_stability.to_csv(dataset_stability_path, index=False)
    track_stability.to_csv(track_stability_path, index=False)
    top_settings.to_csv(top_settings_path, index=False)
    failure_records.to_csv(failure_records_path, index=False)

    _draw_overall_mrr_heatmap(summary, output_root=output_root, k1_order=k1_order, b_order=b_order)
    _draw_track_mrr_heatmaps(by_track, output_root=output_root, k1_order=k1_order, b_order=b_order)
    _draw_mrr_profiles_by_b(summary, output_root=output_root, k1_order=k1_order, b_order=b_order)
    _draw_failure_rate_heatmap(summary, output_root=output_root, k1_order=k1_order, b_order=b_order)
    _write_interpretation_scaffold(
        interpretation_path,
        source_csv=source_csv,
        summary=summary,
        track_stability=track_stability,
        top_settings=top_settings,
    )

    generated_at = datetime.now().isoformat(timespec="seconds")
    manifest_path = output_root / "bm25_sensitivity_manifest.txt"
    manifest_path.write_text(
        "\n".join(
            [
                f"generated_at={generated_at}",
                f"source_csv={source_csv}",
                f"config_path={Path(config_path).resolve()}",
                f"output_dir={output_root}",
                f"bootstrap_seed={BOOTSTRAP_SEED}",
                f"bootstrap_samples={BOOTSTRAP_SAMPLES}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "source_csv": source_csv,
        "output_dir": output_root,
        "bm25_sensitivity_summary": summary_path,
        "bm25_sensitivity_by_track": by_track_path,
        "bm25_dataset_stability": dataset_stability_path,
        "bm25_track_stability": track_stability_path,
        "bm25_top_settings_with_ci": top_settings_path,
        "bm25_failure_records": failure_records_path,
        "bm25_mrr_heatmap_overall_png": output_root / "bm25_mrr_heatmap_overall.png",
        "bm25_mrr_heatmap_overall_pdf": output_root / "bm25_mrr_heatmap_overall.pdf",
        "bm25_mrr_heatmap_by_track_png": output_root / "bm25_mrr_heatmap_by_track.png",
        "bm25_mrr_heatmap_by_track_pdf": output_root / "bm25_mrr_heatmap_by_track.pdf",
        "bm25_mrr_profiles_by_b_png": output_root / "bm25_mrr_profiles_by_b.png",
        "bm25_mrr_profiles_by_b_pdf": output_root / "bm25_mrr_profiles_by_b.pdf",
        "bm25_failure_rate_heatmap_png": output_root / "bm25_failure_rate_heatmap.png",
        "bm25_failure_rate_heatmap_pdf": output_root / "bm25_failure_rate_heatmap.pdf",
        "bm25_sensitivity_interpretation": interpretation_path,
        "manifest": manifest_path,
    }
