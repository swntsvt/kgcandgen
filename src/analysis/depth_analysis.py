"""Depth-wise retrieval behavior analysis for candidate cutoff decisions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.analysis.plot_env import configure_plot_environment

configure_plot_environment()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEPTHS = [1, 5, 10, 20, 50]
TRANSITIONS: list[tuple[int, int]] = [(1, 5), (5, 10), (10, 20), (20, 50)]
REQUIRED_COLUMNS = {
    "dataset",
    "track",
    "method",
    "candidate_size",
    "mrr",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "recall_at_50",
}


class DepthAnalysisValidationError(ValueError):
    """Raised when depth-analysis input data is invalid."""


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
        raise FileNotFoundError("No results/result_*.csv files found for depth analysis.")
    return candidates[-1].resolve()


def _transition_label(from_k: int, to_k: int) -> str:
    return f"{from_k}->{to_k}"


def _validate_and_prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise DepthAnalysisValidationError(
            "Missing required depth-analysis column(s): " + ", ".join(missing_columns)
        )

    prepared = frame.copy()
    prepared["dataset"] = prepared["dataset"].astype(str)
    prepared["track"] = prepared["track"].astype(str)
    prepared["method"] = prepared["method"].astype(str)
    prepared["candidate_size"] = prepared["candidate_size"].astype(int)
    prepared["mrr"] = prepared["mrr"].astype(float)
    for depth in DEPTHS:
        prepared[f"recall_at_{depth}"] = prepared[f"recall_at_{depth}"].astype(float)

    if (prepared["candidate_size"] <= 0).any():
        raise DepthAnalysisValidationError("candidate_size must be positive for all rows.")
    return prepared


def _select_best_settings(prepared: pd.DataFrame) -> pd.DataFrame:
    ranked = prepared.reset_index(drop=False).rename(columns={"index": "_row_order"})
    ranked = ranked.sort_values(
        ["dataset", "method", "mrr", "_row_order"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    best = ranked.groupby(["dataset", "method"], as_index=False).first()

    for depth in DEPTHS:
        best[f"effective_k_{depth}"] = best["candidate_size"].apply(
            lambda value: min(int(value), depth)
        )
    best = best.drop(columns=["_row_order"])
    return best.sort_values(["track", "dataset", "method"], kind="mergesort").reset_index(drop=True)


def _build_marginal_gains(best_settings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in best_settings.itertuples(index=False):
        for from_k, to_k in TRANSITIONS:
            gain = float(getattr(row, f"recall_at_{to_k}") - getattr(row, f"recall_at_{from_k}"))
            capped = int(row.candidate_size) < to_k
            rows.append(
                {
                    "track": str(row.track),
                    "dataset": str(row.dataset),
                    "method": str(row.method),
                    "candidate_size": int(row.candidate_size),
                    "from_k": from_k,
                    "to_k": to_k,
                    "transition": _transition_label(from_k, to_k),
                    "marginal_gain": gain,
                    "is_capped_before_to": capped,
                    "effective_to_k": min(int(row.candidate_size), to_k),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["track", "dataset", "method", "from_k"], kind="mergesort"
    ).reset_index(drop=True)


def _build_gain_summary_overall(marginal_gains: pd.DataFrame) -> pd.DataFrame:
    summary = (
        marginal_gains.groupby(["method", "transition", "from_k", "to_k"], as_index=False)
        .agg(
            marginal_gain_mean=("marginal_gain", "mean"),
            marginal_gain_median=("marginal_gain", "median"),
            marginal_gain_std=("marginal_gain", "std"),
        )
        .fillna(0.0)
        .sort_values(["method", "from_k"], kind="mergesort")
        .reset_index(drop=True)
    )
    return summary


def _build_gain_summary_by_track(marginal_gains: pd.DataFrame) -> pd.DataFrame:
    summary = (
        marginal_gains.groupby(["track", "method", "transition", "from_k", "to_k"], as_index=False)
        .agg(
            marginal_gain_mean=("marginal_gain", "mean"),
            marginal_gain_median=("marginal_gain", "median"),
            marginal_gain_std=("marginal_gain", "std"),
        )
        .fillna(0.0)
        .sort_values(["track", "method", "from_k"], kind="mergesort")
        .reset_index(drop=True)
    )
    return summary


def _build_transition_coverage(marginal_gains: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for transition, group in marginal_gains.groupby("transition", sort=False):
        total = int(len(group))
        capped = int(group["is_capped_before_to"].sum())
        rows.append(
            {
                "method": "ALL",
                "transition": str(transition),
                "to_k": int(group["to_k"].iloc[0]),
                "total_count": total,
                "full_coverage_count": total - capped,
                "capped_count": capped,
                "full_coverage_rate": float((total - capped) / total) if total else 0.0,
                "capped_rate": float(capped / total) if total else 0.0,
            }
        )
    for (method, transition), group in marginal_gains.groupby(["method", "transition"], sort=False):
        total = int(len(group))
        capped = int(group["is_capped_before_to"].sum())
        rows.append(
            {
                "method": str(method),
                "transition": str(transition),
                "to_k": int(group["to_k"].iloc[0]),
                "total_count": total,
                "full_coverage_count": total - capped,
                "capped_count": capped,
                "full_coverage_rate": float((total - capped) / total) if total else 0.0,
                "capped_rate": float(capped / total) if total else 0.0,
            }
        )
    coverage = pd.DataFrame(rows).sort_values(
        ["method", "to_k"], kind="mergesort"
    ).reset_index(drop=True)
    return coverage


def _draw_recall_gain_curves(best_settings: pd.DataFrame, output_root: Path) -> None:
    rows: list[dict[str, object]] = []
    for row in best_settings.itertuples(index=False):
        for depth in DEPTHS:
            rows.append(
                {
                    "method": str(row.method),
                    "k": depth,
                    "recall": float(getattr(row, f"recall_at_{depth}")),
                }
            )
    plot_data = pd.DataFrame(rows)
    curve_data = (
        plot_data.groupby(["method", "k"], as_index=False)
        .agg(recall_mean=("recall", "mean"))
        .sort_values(["method", "k"], kind="mergesort")
    )
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.lineplot(
        data=curve_data,
        x="k",
        y="recall_mean",
        hue="method",
        marker="o",
        ax=ax,
    )
    ax.set_xlabel("k")
    ax.set_ylabel("Mean Recall@k")
    ax.set_title("Depth-wise Recall Gain Curves (Best-MRR Settings)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Method")
    fig.tight_layout()
    base = output_root / "depth_recall_gain_curves_by_method"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _draw_marginal_gain_bars(gain_summary_overall: pd.DataFrame, output_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.barplot(
        data=gain_summary_overall,
        x="transition",
        y="marginal_gain_mean",
        hue="method",
        ax=ax,
    )
    ax.set_xlabel("Transition")
    ax.set_ylabel("Mean Marginal Gain")
    ax.set_title("Marginal Recall Gain by Cutoff Transition")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Method")
    fig.tight_layout()
    base = output_root / "depth_marginal_gain_bars"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _draw_transition_coverage(coverage: pd.DataFrame, output_root: Path) -> None:
    overall = coverage[coverage["method"] == "ALL"].copy()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.barplot(
        data=overall,
        x="transition",
        y="capped_rate",
        color="#d97706",
        ax=ax,
    )
    ax.set_xlabel("Transition")
    ax.set_ylabel("Capped Dataset Fraction")
    ax.set_title("Depth Transition Coverage (Cap-aware)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    base = output_root / "depth_transition_coverage"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _write_interpretation_scaffold(
    output_path: Path,
    *,
    source_csv: Path,
    gain_summary_overall: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    overall_by_transition = (
        gain_summary_overall.groupby(["transition", "from_k", "to_k"], as_index=False)
        .agg(mean_gain=("marginal_gain_mean", "mean"))
        .sort_values("from_k", kind="mergesort")
    )
    best_transition = (
        overall_by_transition.sort_values("mean_gain", ascending=False, kind="mergesort").iloc[0]
        if not overall_by_transition.empty
        else None
    )
    early = overall_by_transition[overall_by_transition["transition"].isin(["1->5", "5->10"])]
    late = overall_by_transition[overall_by_transition["transition"].isin(["10->20", "20->50"])]
    early_sum = float(early["mean_gain"].sum()) if not early.empty else 0.0
    late_sum = float(late["mean_gain"].sum()) if not late.empty else 0.0

    coverage_overall = coverage[coverage["method"] == "ALL"].copy().set_index("transition")
    capped_20_50 = (
        float(coverage_overall.loc["20->50", "capped_rate"])
        if "20->50" in coverage_overall.index
        else 0.0
    )
    lines = [
        "# Depth Analysis Interpretation Scaffold",
        "",
        f"Source results CSV: `{source_csv}`",
        "",
        "## What Happens Across k",
    ]
    if best_transition is not None:
        lines.append(
            f"- Largest average gain transition: {best_transition['transition']} "
            f"(mean gain={float(best_transition['mean_gain']):.4f})."
        )
    lines.extend(
        [
            f"- Early-depth gain total (`1->5` + `5->10`): {early_sum:.4f}.",
            f"- Late-depth gain total (`10->20` + `20->50`): {late_sum:.4f}.",
            f"- Capped fraction at `20->50`: {capped_20_50:.4f}.",
            "",
            "## Why Gains Saturate",
            "- If late-depth gain is much smaller than early-depth gain, ranking quality is mostly realized at low k.",
            "- If capped fraction is non-trivial at high depth, observed Recall@50 behavior is partially ceiling-limited by candidate_size.",
            "- Use the coverage table and transition bars jointly when recommending practical cutoffs.",
            "",
            "## Fill-In Prompts",
            "- Practical cutoff claim: \"A defensible default cutoff is k=... because ...\"",
            "- Saturation claim: \"Beyond k=..., gains flatten due to ...\"",
            "- Ceiling caveat: \"High-k interpretation should account for candidate_size ceilings because ...\"",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_depth_analysis(
    results_csv_path: str | Path | None,
    output_dir: str | Path = "results/comparisons",
) -> dict[str, Path]:
    """Generate depth-wise recall gain analysis artifacts."""
    source_csv = _resolve_results_csv(results_csv_path)
    prepared = _validate_and_prepare_frame(pd.read_csv(source_csv))
    best_settings = _select_best_settings(prepared)
    marginal_gains = _build_marginal_gains(best_settings)
    gain_summary_overall = _build_gain_summary_overall(marginal_gains)
    gain_summary_by_track = _build_gain_summary_by_track(marginal_gains)
    transition_coverage = _build_transition_coverage(marginal_gains)

    run_id = source_csv.stem
    output_root = Path(output_dir) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    best_settings_path = output_root / "depth_best_settings.csv"
    marginal_gains_path = output_root / "depth_marginal_gains.csv"
    summary_overall_path = output_root / "depth_gain_summary_overall.csv"
    summary_by_track_path = output_root / "depth_gain_summary_by_track.csv"
    coverage_path = output_root / "depth_transition_coverage.csv"
    interpretation_path = output_root / "depth_analysis_interpretation.md"

    best_settings.to_csv(best_settings_path, index=False)
    marginal_gains.to_csv(marginal_gains_path, index=False)
    gain_summary_overall.to_csv(summary_overall_path, index=False)
    gain_summary_by_track.to_csv(summary_by_track_path, index=False)
    transition_coverage.to_csv(coverage_path, index=False)

    _draw_recall_gain_curves(best_settings, output_root)
    _draw_marginal_gain_bars(gain_summary_overall, output_root)
    _draw_transition_coverage(transition_coverage, output_root)
    _write_interpretation_scaffold(
        interpretation_path,
        source_csv=source_csv,
        gain_summary_overall=gain_summary_overall,
        coverage=transition_coverage,
    )

    generated_at = datetime.now().isoformat(timespec="seconds")
    manifest_path = output_root / "depth_analysis_manifest.txt"
    manifest_path.write_text(
        "\n".join(
            [
                f"generated_at={generated_at}",
                f"source_csv={source_csv}",
                f"output_dir={output_root}",
                f"depths={','.join(str(depth) for depth in DEPTHS)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "source_csv": source_csv,
        "output_dir": output_root,
        "depth_best_settings": best_settings_path,
        "depth_marginal_gains": marginal_gains_path,
        "depth_gain_summary_overall": summary_overall_path,
        "depth_gain_summary_by_track": summary_by_track_path,
        "depth_transition_coverage": coverage_path,
        "depth_recall_gain_curves_by_method_png": output_root / "depth_recall_gain_curves_by_method.png",
        "depth_recall_gain_curves_by_method_pdf": output_root / "depth_recall_gain_curves_by_method.pdf",
        "depth_marginal_gain_bars_png": output_root / "depth_marginal_gain_bars.png",
        "depth_marginal_gain_bars_pdf": output_root / "depth_marginal_gain_bars.pdf",
        "depth_transition_coverage_png": output_root / "depth_transition_coverage.png",
        "depth_transition_coverage_pdf": output_root / "depth_transition_coverage.pdf",
        "depth_analysis_interpretation": interpretation_path,
        "manifest": manifest_path,
    }
