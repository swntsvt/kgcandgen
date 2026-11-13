"""Classify grid configurations as valid, reduced-scoring, feature-losing or failed.

Distinguishes configurations that are structurally unsuitable from ones that are valid
but rank less well:

- ``failed``: at least one dataset run raised (for example an empty TF-IDF vocabulary).
- ``feature_loss``: the configuration leaves noticeably more queries with no candidate
  than the best configuration of the same method, i.e. queries cannot be represented
  (TF-IDF without unigrams, ``min_df`` > 1 on labels whose terms occur once).
- ``reduced_scoring``: BM25 with ``k1 = 0`` or ``b = 0``. When label terms occur at most
  once per label, both reduce BM25 to summed IDF; candidates are unchanged, only the order.
- ``valid``: none of the above.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_FEATURE_LOSS_THRESHOLD = 0.05
GRID_METHODS: tuple[str, ...] = ("tfidf", "bm25", "char_ngram_sensitivity")


def _reduces_to_summed_idf(method: str, hyperparameters: dict[str, object]) -> bool:
    if method != "bm25":
        return False
    return float(hyperparameters.get("k1", 1.0)) == 0.0 or float(hyperparameters.get("b", 1.0)) == 0.0


def diagnose_grid(
    results: pd.DataFrame,
    errors: pd.DataFrame | None = None,
    *,
    feature_loss_threshold: float = DEFAULT_FEATURE_LOSS_THRESHOLD,
) -> pd.DataFrame:
    """Return one row per (method, configuration) with summary metrics and a category."""
    required = {"method", "hyperparameters", "dataset", "mrr", "recall_at_50", "zero_candidate_query_share"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"results are missing columns: {missing}")
    frame = results[results["method"].isin(GRID_METHODS)].copy()
    if frame.empty:
        raise ValueError("results contain no grid methods.")

    summary = (
        frame.groupby(["method", "hyperparameters"], as_index=False)
        .agg(
            datasets_completed=("dataset", "nunique"),
            mean_mrr=("mrr", "mean"),
            mean_recall_at_50=("recall_at_50", "mean"),
            mean_zero_candidate_query_share=("zero_candidate_query_share", "mean"),
        )
    )
    failures = pd.Series(dtype=int)
    if errors is not None and not errors.empty:
        failures = (
            errors[errors["scope"] == "model_run"]
            .groupby(["method", "hyperparameters"])
            .size()
        )
    summary["failed_runs"] = [
        int(failures.get((row.method, row.hyperparameters), 0)) for row in summary.itertuples()
    ]
    best_share = summary.groupby("method")["mean_zero_candidate_query_share"].transform("min")
    summary["excess_zero_candidate_share"] = summary["mean_zero_candidate_query_share"] - best_share

    categories = []
    for row in summary.itertuples():
        hyperparameters = json.loads(row.hyperparameters)
        if row.failed_runs > 0:
            categories.append("failed")
        elif row.excess_zero_candidate_share > feature_loss_threshold:
            categories.append("feature_loss")
        elif _reduces_to_summed_idf(row.method, hyperparameters):
            categories.append("reduced_scoring")
        else:
            categories.append("valid")
    summary["category"] = categories
    return summary.sort_values(["method", "mean_mrr"], ascending=[True, False]).reset_index(drop=True)


def run_grid_diagnostics(
    results_csv: str | Path,
    *,
    errors_csv: str | Path | None = None,
    output_csv: str | Path | None = None,
    feature_loss_threshold: float = DEFAULT_FEATURE_LOSS_THRESHOLD,
) -> Path:
    results_path = Path(results_csv)
    errors_path = (
        Path(errors_csv)
        if errors_csv is not None
        else results_path.with_name(f"{results_path.stem}_errors.csv")
    )
    errors = pd.read_csv(errors_path) if errors_path.exists() else None
    diagnosed = diagnose_grid(
        pd.read_csv(results_path), errors, feature_loss_threshold=feature_loss_threshold
    )
    output_path = (
        Path(output_csv)
        if output_csv is not None
        else Path("results") / "comparisons" / results_path.stem / "grid_diagnostics.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnosed.to_csv(output_path, index=False)
    return output_path
