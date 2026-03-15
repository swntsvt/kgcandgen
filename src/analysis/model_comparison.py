"""Primary TF-IDF vs BM25 comparison reporting utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REQUIRED_COLUMNS = {"dataset", "track", "method", "mrr", "recall_at_10", "recall_at_50"}
REQUIRED_METHODS = {"tfidf", "bm25"}


class ComparisonValidationError(ValueError):
    """Raised when comparison input data is invalid."""


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
        raise FileNotFoundError("No results/result_*.csv files found for comparison.")
    return candidates[-1].resolve()


def _validate_results_frame(frame: pd.DataFrame) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ComparisonValidationError(
            "Missing required comparison column(s): " + ", ".join(missing_columns)
        )

    methods = set(frame["method"].astype(str).unique())
    missing_methods = sorted(REQUIRED_METHODS - methods)
    if missing_methods:
        raise ComparisonValidationError(
            "Comparison requires both methods. Missing: "
            + ", ".join(missing_methods)
        )


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["track", "dataset", "method"], kind="mergesort").reset_index(
        drop=True
    )


def _build_best_of_grid(frame: pd.DataFrame) -> pd.DataFrame:
    sorted_for_mrr = frame.sort_values(
        ["track", "dataset", "method", "mrr"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    best_mrr = sorted_for_mrr.groupby(["track", "dataset", "method"], as_index=False).first()

    sorted_for_recall = frame.sort_values(
        ["track", "dataset", "method", "recall_at_10"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    best_recall = sorted_for_recall.groupby(["track", "dataset", "method"], as_index=False).first()

    merged = best_mrr[["track", "dataset", "method", "mrr", "hyperparameters"]].rename(
        columns={
            "mrr": "best_mrr",
            "hyperparameters": "best_mrr_hyperparameters",
        }
    ).merge(
        best_recall[["track", "dataset", "method", "recall_at_10", "hyperparameters"]].rename(
            columns={
                "recall_at_10": "best_recall_at_10",
                "hyperparameters": "best_recall_at_10_hyperparameters",
            }
        ),
        on=["track", "dataset", "method"],
        how="inner",
        validate="one_to_one",
    ).merge(
        frame.sort_values(
            ["track", "dataset", "method", "recall_at_50"],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
        .groupby(["track", "dataset", "method"], as_index=False)
        .first()[["track", "dataset", "method", "recall_at_50", "hyperparameters"]]
        .rename(
            columns={
                "recall_at_50": "best_recall_at_50",
                "hyperparameters": "best_recall_at_50_hyperparameters",
            }
        ),
        on=["track", "dataset", "method"],
        how="inner",
        validate="one_to_one",
    )

    return _sort_frame(merged)


def _build_aggregate_of_grid(frame: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        frame.groupby(["track", "dataset", "method"], as_index=False)
        .agg(
            mrr_mean=("mrr", "mean"),
            mrr_median=("mrr", "median"),
            mrr_std=("mrr", "std"),
            recall_at_10_mean=("recall_at_10", "mean"),
            recall_at_10_median=("recall_at_10", "median"),
            recall_at_10_std=("recall_at_10", "std"),
            recall_at_50_mean=("recall_at_50", "mean"),
            recall_at_50_median=("recall_at_50", "median"),
            recall_at_50_std=("recall_at_50", "std"),
        )
        .fillna(0.0)
    )
    return _sort_frame(aggregate)


def _paired_method_frame(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Return only dataset rows where both TF-IDF and BM25 values are present."""
    pivoted = (
        frame.pivot(index=["track", "dataset"], columns="method", values=value_column)
        .reset_index()
        .sort_values(["track", "dataset"], kind="mergesort")
    )
    return pivoted.dropna(subset=["tfidf", "bm25"]).reset_index(drop=True)


def _win_counts_for_metric(best_frame: pd.DataFrame, metric_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivoted = _paired_method_frame(best_frame, metric_column)

    def compare_row(row: pd.Series) -> str:
        tfidf_value = float(row["tfidf"])
        bm25_value = float(row["bm25"])
        if tfidf_value == bm25_value:
            return "tie"
        return "tfidf_win" if tfidf_value > bm25_value else "bm25_win"

    pivoted["outcome"] = pivoted.apply(compare_row, axis=1)

    overall_outcomes = pivoted["outcome"].value_counts().to_dict()
    overall = pd.DataFrame(
        [
            {
                "metric": metric_column,
                "tfidf_wins": int(overall_outcomes.get("tfidf_win", 0)),
                "bm25_wins": int(overall_outcomes.get("bm25_win", 0)),
                "ties": int(overall_outcomes.get("tie", 0)),
                "total_datasets": int(len(pivoted)),
            }
        ]
    )

    by_track_rows: list[dict[str, Any]] = []
    for track, group in pivoted.groupby("track", sort=True):
        outcome_counts = group["outcome"].value_counts().to_dict()
        by_track_rows.append(
            {
                "track": str(track),
                "metric": metric_column,
                "tfidf_wins": int(outcome_counts.get("tfidf_win", 0)),
                "bm25_wins": int(outcome_counts.get("bm25_win", 0)),
                "ties": int(outcome_counts.get("tie", 0)),
                "total_datasets": int(len(group)),
            }
        )
    by_track = pd.DataFrame(by_track_rows).sort_values(["track", "metric"], kind="mergesort")

    return overall, by_track


def _draw_best_mrr_dumbbell(best_frame: pd.DataFrame, output_base: Path) -> None:
    plot_data = best_frame[["track", "dataset", "method", "best_mrr"]].copy()
    pivoted = _paired_method_frame(plot_data, "best_mrr")

    fig_height = max(6, 0.35 * len(pivoted))
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=300)

    y_positions = range(len(pivoted))
    for idx, row in pivoted.iterrows():
        ax.plot(
            [row["tfidf"], row["bm25"]],
            [idx, idx],
            color="#9ca3af",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )

    ax.scatter(pivoted["tfidf"], y_positions, color="#1f77b4", label="TF-IDF", s=24, zorder=2)
    ax.scatter(pivoted["bm25"], y_positions, color="#d62728", label="BM25", s=24, zorder=2)

    labels = [f"{row.track} | {row.dataset}" for row in pivoted.itertuples(index=False)]
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Best MRR")
    ax.set_ylabel("Dataset")
    ax.set_title("Best-of-Grid MRR per Dataset: TF-IDF vs BM25")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def _draw_track_distribution_plot(
    best_frame: pd.DataFrame,
    output_base: Path,
    *,
    kind: str,
) -> None:
    plot_data = best_frame[["track", "method", "best_mrr"]].copy()
    plot_data = plot_data.sort_values(["track", "method"], kind="mergesort")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    if kind == "box":
        sns.boxplot(data=plot_data, x="track", y="best_mrr", hue="method", ax=ax)
    else:
        sns.violinplot(data=plot_data, x="track", y="best_mrr", hue="method", ax=ax, cut=0)

    ax.set_xlabel("Track")
    ax.set_ylabel("Best MRR")
    title_suffix = "Box" if kind == "box" else "Violin"
    ax.set_title(f"Track-wise Best-of-Grid MRR by Method ({title_suffix})")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Method")

    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def _write_interpretation_scaffold(
    output_path: Path,
    *,
    source_csv: Path,
    wins_overall: pd.DataFrame,
    wins_by_track: pd.DataFrame,
) -> None:
    overall_lookup = {str(row.metric): row for row in wins_overall.itertuples(index=False)}
    by_track_lookup = {
        (str(row.track), str(row.metric)): row for row in wins_by_track.itertuples(index=False)
    }

    mrr_row = overall_lookup.get("best_mrr")
    recall_row = overall_lookup.get("best_recall_at_10")
    recall_50_row = overall_lookup.get("best_recall_at_50")

    def _winner_text(metric_row: Any) -> str:
        if metric_row is None:
            return "insufficient data"
        if int(metric_row.tfidf_wins) == int(metric_row.bm25_wins):
            return "tie"
        if int(metric_row.tfidf_wins) > int(metric_row.bm25_wins):
            return "TF-IDF"
        return "BM25"

    overall_mrr_winner = _winner_text(mrr_row)
    overall_recall_winner = _winner_text(recall_row)
    overall_recall_50_winner = _winner_text(recall_50_row)

    bm25_mrr_tracks: list[str] = []
    tfidf_or_match_mrr_tracks: list[str] = []
    for track, metric in sorted(by_track_lookup.keys()):
        if metric != "best_mrr":
            continue
        row = by_track_lookup[(track, metric)]
        if int(row.bm25_wins) > int(row.tfidf_wins):
            bm25_mrr_tracks.append(track)
        else:
            tfidf_or_match_mrr_tracks.append(track)

    lines = [
        "# Model Comparison Interpretation Scaffold",
        "",
        f"Source results CSV: `{source_csv}`",
        "",
        "## Overall (Best-of-Grid)",
    ]

    for row in wins_overall.itertuples(index=False):
        lines.append(
            "- "
            f"{row.metric}: tfidf_wins={row.tfidf_wins}, "
            f"bm25_wins={row.bm25_wins}, ties={row.ties}, total_datasets={row.total_datasets}"
        )

    lines.extend(["", "## Track-wise (Best-of-Grid)"])
    for row in wins_by_track.itertuples(index=False):
        lines.append(
            "- "
            f"track={row.track}, metric={row.metric}, tfidf_wins={row.tfidf_wins}, "
            f"bm25_wins={row.bm25_wins}, ties={row.ties}, total_datasets={row.total_datasets}"
        )

    lines.extend(
        [
            "",
            "## What Happened (Fill-In Guide)",
            (
                "- Overall winner by MRR: "
                f"{overall_mrr_winner} "
                f"(tfidf_wins={int(mrr_row.tfidf_wins) if mrr_row else 0}, "
                f"bm25_wins={int(mrr_row.bm25_wins) if mrr_row else 0}, "
                f"ties={int(mrr_row.ties) if mrr_row else 0})."
            ),
            (
                "- Overall winner by Recall@10: "
                f"{overall_recall_winner} "
                f"(tfidf_wins={int(recall_row.tfidf_wins) if recall_row else 0}, "
                f"bm25_wins={int(recall_row.bm25_wins) if recall_row else 0}, "
                f"ties={int(recall_row.ties) if recall_row else 0})."
            ),
            (
                "- Overall winner by Recall@50: "
                f"{overall_recall_50_winner} "
                f"(tfidf_wins={int(recall_50_row.tfidf_wins) if recall_50_row else 0}, "
                f"bm25_wins={int(recall_50_row.bm25_wins) if recall_50_row else 0}, "
                f"ties={int(recall_50_row.ties) if recall_50_row else 0})."
            ),
            (
                "- Tracks where BM25 clearly outperformed TF-IDF (best MRR view): "
                + (", ".join(bm25_mrr_tracks) if bm25_mrr_tracks else "none")
                + "."
            ),
            (
                "- Tracks where TF-IDF matched or exceeded BM25 (best MRR view): "
                + (", ".join(tfidf_or_match_mrr_tracks) if tfidf_or_match_mrr_tracks else "none")
                + "."
            ),
            (
                "- Notable dataset-level exceptions: inspect `best_of_grid_summary.csv` for cases "
                "where the non-overall winner leads on individual datasets."
            ),
            (
                "- Implications for default candidate generation method: if maximizing rank quality "
                "(MRR), prefer the MRR overall winner; if prioritizing candidate-set coverage, "
                "use Recall@10 and Recall@50 outcomes plus tie profiles to decide."
            ),
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_model_comparison(
    results_csv_path: str | Path | None,
    output_dir: str | Path = "results/comparisons",
) -> dict[str, Path]:
    """Generate TF-IDF vs BM25 model comparison outputs from an experiment CSV."""
    source_csv = _resolve_results_csv(results_csv_path)
    frame = pd.read_csv(source_csv)
    _validate_results_frame(frame)
    if "hyperparameters" not in frame.columns:
        frame["hyperparameters"] = ""

    frame["mrr"] = frame["mrr"].astype(float)
    frame["recall_at_10"] = frame["recall_at_10"].astype(float)
    frame["recall_at_50"] = frame["recall_at_50"].astype(float)
    frame["method"] = frame["method"].astype(str)
    frame["track"] = frame["track"].astype(str)
    frame["dataset"] = frame["dataset"].astype(str)

    sorted_frame = _sort_frame(frame)
    best_of_grid = _build_best_of_grid(sorted_frame)
    aggregate_of_grid = _build_aggregate_of_grid(sorted_frame)

    mrr_overall, mrr_by_track = _win_counts_for_metric(best_of_grid, "best_mrr")
    r10_overall, r10_by_track = _win_counts_for_metric(best_of_grid, "best_recall_at_10")
    r50_overall, r50_by_track = _win_counts_for_metric(best_of_grid, "best_recall_at_50")

    wins_overall = pd.concat([mrr_overall, r10_overall, r50_overall], ignore_index=True)
    wins_by_track = pd.concat([mrr_by_track, r10_by_track, r50_by_track], ignore_index=True)
    wins_by_track = wins_by_track.sort_values(["track", "metric"], kind="mergesort").reset_index(
        drop=True
    )

    run_id = source_csv.stem
    output_root = Path(output_dir) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    best_summary_path = output_root / "best_of_grid_summary.csv"
    aggregate_summary_path = output_root / "aggregate_of_grid_summary.csv"
    wins_overall_path = output_root / "wins_overall.csv"
    wins_by_track_path = output_root / "wins_by_track.csv"
    interpretation_path = output_root / "interpretation_scaffold.md"

    best_of_grid.to_csv(best_summary_path, index=False)
    aggregate_of_grid.to_csv(aggregate_summary_path, index=False)
    wins_overall.to_csv(wins_overall_path, index=False)
    wins_by_track.to_csv(wins_by_track_path, index=False)

    _draw_best_mrr_dumbbell(best_of_grid, output_root / "best_mrr_dumbbell")
    _draw_track_distribution_plot(best_of_grid, output_root / "best_mrr_track_box", kind="box")
    _draw_track_distribution_plot(
        best_of_grid,
        output_root / "best_mrr_track_violin",
        kind="violin",
    )

    _write_interpretation_scaffold(
        interpretation_path,
        source_csv=source_csv,
        wins_overall=wins_overall,
        wins_by_track=wins_by_track,
    )

    generated_at = datetime.now().isoformat(timespec="seconds")
    manifest_path = output_root / "manifest.txt"
    manifest_path.write_text(
        "\n".join(
            [
                f"generated_at={generated_at}",
                f"source_csv={source_csv}",
                f"output_dir={output_root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "source_csv": source_csv,
        "output_dir": output_root,
        "best_of_grid_summary": best_summary_path,
        "aggregate_of_grid_summary": aggregate_summary_path,
        "wins_overall": wins_overall_path,
        "wins_by_track": wins_by_track_path,
        "best_mrr_dumbbell_png": (output_root / "best_mrr_dumbbell.png"),
        "best_mrr_dumbbell_pdf": (output_root / "best_mrr_dumbbell.pdf"),
        "best_mrr_track_box_png": (output_root / "best_mrr_track_box.png"),
        "best_mrr_track_box_pdf": (output_root / "best_mrr_track_box.pdf"),
        "best_mrr_track_violin_png": (output_root / "best_mrr_track_violin.png"),
        "best_mrr_track_violin_pdf": (output_root / "best_mrr_track_violin.pdf"),
        "interpretation_scaffold": interpretation_path,
        "manifest": manifest_path,
    }
