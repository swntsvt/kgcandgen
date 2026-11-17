"""Draw the revised paper's Results figures from a canonical run.

    python scripts/paper_figures.py --label v1.2-sncs-r1 --output-dir <paper>/images

Reads only files listed in ``results/canonical/<label>/manifest.json``:

- ``grid_diagnostics.png/.pdf``: development mean MRR of every grid cell, per method,
  coloured by diagnostic category (valid, reduced scoring, feature loss, failed).
- ``latency_scaling.png/.pdf``: median single-thread query latency against target pool
  size (pools of at least 1,000 targets), log-log, one line per method.
- ``marvel_scaling.png/.pdf`` (with ``--scaling-csv``): latency and Recall@50 of the Marvel
  instance queries on nested Marvel subsamples, the full pool and the pooled KG set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

CATEGORY_STYLE = {
    "valid": ("#2a7fb8", "o", "Valid"),
    "reduced_scoring": ("#e39b2d", "s", "Reduced-scoring"),
    "feature_loss": ("#c8453b", "D", "Feature-loss"),
    "failed": ("#555555", "X", "Failed on $\\geq$1 pair"),
}
GRID_METHODS = [
    ("bm25", "BM25\n(20 cells)"),
    ("tfidf", "TF-IDF\n(20 cells)"),
    ("char_ngram_sensitivity", "Char $n$-gram\n(24 cells)"),
]
LATENCY_METHODS = [
    ("exact_match", "Exact match", "#7f7f7f", "v"),
    ("bm25", "BM25", "#e39b2d", "s"),
    ("tfidf", "TF-IDF", "#9467bd", "^"),
    ("char_ngram", "Char $n$-gram", "#2a7fb8", "o"),
    ("dense_minilm", "MiniLM", "#2ca02c", "D"),
    ("hybrid_char_minilm", "Hybrid (MiniLM)", "#c8453b", "P"),
    ("dense_biolord", "BioLORD", "#17becf", "d"),
    ("hybrid_char_biolord", "Hybrid (BioLORD)", "#8c564b", "X"),
]


def _artifact(manifest: dict, suffix: str) -> Path:
    matches = [path for path in manifest["artifacts_sha256"] if path.endswith(suffix)]
    if len(matches) != 1:
        raise SystemExit(f"Expected one manifest artefact ending in {suffix!r}, found {matches}")
    return ROOT / matches[0]


def _save(fig, output_dir: Path, name: str) -> None:
    for suffix in (".pdf", ".png"):
        fig.savefig(output_dir / f"{name}{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_grid_diagnostics(diagnostics: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rng = np.random.default_rng(0)
    for position, (method, _label) in enumerate(GRID_METHODS):
        cells = diagnostics[diagnostics["method"] == method]
        for category, (colour, marker, _name) in CATEGORY_STYLE.items():
            subset = cells[cells["category"] == category]
            if subset.empty:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=len(subset))
            ax.scatter(
                position + jitter, subset["mean_mrr"], s=26, c=colour, marker=marker,
                edgecolors="white", linewidths=0.4, zorder=3,
            )
    handles = [
        plt.Line2D([], [], color=colour, marker=marker, linestyle="", markersize=6, label=name)
        for colour, marker, name in CATEGORY_STYLE.values()
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)
    ax.set_xticks(range(len(GRID_METHODS)), [label for _m, label in GRID_METHODS])
    ax.set_xlim(-0.6, len(GRID_METHODS) - 0.4)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Mean MRR, 24 development pairs")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output_dir, "grid_diagnostics")


def draw_latency_scaling(efficiency: pd.DataFrame, output_dir: Path) -> None:
    frame = efficiency[efficiency["target_pool_size"] >= 1000]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for method, label, colour, marker in LATENCY_METHODS:
        rows = frame[frame["method"] == method].sort_values("target_pool_size")
        if rows.empty:
            continue
        slope = np.polyfit(np.log(rows["target_pool_size"]), np.log(rows["latency_median_ms"]), 1)[0]
        ax.plot(
            rows["target_pool_size"], rows["latency_median_ms"], color=colour, marker=marker,
            markersize=4, linewidth=0.9, label=f"{label} (slope {slope:.2f})",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Target pool size (entities)")
    ax.set_ylabel("Median query latency (ms, one thread)")
    ax.grid(which="major", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    _save(fig, output_dir, "latency_scaling")


def draw_marvel_scaling(scaling: pd.DataFrame, output_dir: Path) -> None:
    """Latency and Recall@50 of the fixed Marvel query set as the target pool grows."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    pooled = scaling.loc[scaling["pool"] == "pooled_kg", "target_pool_size"].max()
    for method, label, colour, marker in LATENCY_METHODS:
        rows = scaling[scaling["method"] == method].sort_values("target_pool_size")
        if rows.empty:
            continue
        style = dict(color=colour, marker=marker, markersize=4, linewidth=0.9)
        left.plot(rows["target_pool_size"], rows["latency_median_ms"], label=label, **style)
        right.plot(rows["target_pool_size"], rows["recall_at_50"], **style)
    for ax in (left, right):
        ax.set_xscale("log")
        ax.axvline(pooled, color="#999999", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Target pool size (entities)")
        ax.grid(which="major", color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    left.set_yscale("log")
    left.set_ylabel("Median query latency (ms)")
    right.set_ylabel("Recall@50")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=3)
    _save(fig, output_dir, "marvel_scaling")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scaling-csv", type=Path, default=None,
                        help="Output of scripts/scaling_benchmark.py; adds marvel_scaling.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "results" / "canonical" / args.label / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise SystemExit(f"Run {args.label} is not complete.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_grid_diagnostics(pd.read_csv(_artifact(manifest, "/grid_diagnostics.csv")), args.output_dir)
    draw_latency_scaling(pd.read_csv(_artifact(manifest, "/efficiency.csv")), args.output_dir)
    if args.scaling_csv is not None:
        draw_marvel_scaling(pd.read_csv(args.scaling_csv), args.output_dir)


if __name__ == "__main__":
    main()
