"""Selection regret: how much held-out MRR each development-side selection rule gives up.

Development results drive several selection rules; held-out full-grid results (run after
the fact and never used for selection) give the post-hoc best configuration. For each
held-out tier and method:

    regret = held-out score of the post-hoc best configuration
             - held-out score of the configuration the rule picked

Scores are macro (mean over dataset x entity-type units) and micro (weighted by gold pairs)
MRR@k_max. ``rank`` is the picked configuration's held-out rank among all configurations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from src.analysis.heldout_selection import (
    _build_selection_summary,
    _canonicalize_hyperparameters,
)

SELECTED_METHODS: tuple[str, ...] = ("tfidf", "bm25")
DEFAULT_LAMBDAS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame[frame["method"].isin(SELECTED_METHODS)].copy()
    prepared["hyperparameters"] = prepared["hyperparameters"].astype(str).map(
        _canonicalize_hyperparameters
    )
    prepared["mrr"] = prepared["mrr"].astype(float)
    return prepared


def _argmax(scores: pd.Series) -> str:
    """Highest score; ties go to the lexicographically smallest configuration string."""
    ordered = scores.reset_index()
    ordered.columns = ["hyperparameters", "score"]
    return str(
        ordered.sort_values(["score", "hyperparameters"], ascending=[False, True]).iloc[0][
            "hyperparameters"
        ]
    )


def development_rule_picks(
    development: pd.DataFrame, lambdas: Sequence[float] = DEFAULT_LAMBDAS
) -> pd.DataFrame:
    """Configuration picked by each selection rule, per tuned method."""
    frame = _prepare(development)
    rows: list[dict[str, object]] = []

    def add(rule: str, picks: dict[str, str]) -> None:
        for method, hyperparameters in picks.items():
            rows.append({"rule": rule, "method": method, "hyperparameters": hyperparameters})

    for lam in lambdas:
        summary = _build_selection_summary(frame, lambda_penalty=lam)
        picked = summary[summary["selected"]]
        add(f"lambda={lam:g}", dict(zip(picked["method"], picked["hyperparameters"])))

    simple_rules: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "mean_mrr_dataset_weighted": lambda f: f.groupby("hyperparameters")["mrr"].mean(),
        "mean_mrr_track_weighted": lambda f: f.groupby(["hyperparameters", "track"])["mrr"]
        .mean()
        .groupby("hyperparameters")
        .mean(),
        "worst_track_mrr": lambda f: f.groupby(["hyperparameters", "track"])["mrr"]
        .mean()
        .groupby("hyperparameters")
        .min(),
    }
    for rule, score in simple_rules.items():
        add(rule, {m: _argmax(score(frame[frame["method"] == m])) for m in SELECTED_METHODS})

    for held_out_track in sorted(frame["track"].unique()):
        subset = frame[frame["track"] != held_out_track]
        summary = _build_selection_summary(subset, lambda_penalty=0.5)
        picked = summary[summary["selected"]]
        add(
            f"leave_out_{held_out_track}(lambda=0.5)",
            dict(zip(picked["method"], picked["hyperparameters"])),
        )
    return pd.DataFrame(rows)


def heldout_config_scores(heldout_grid: pd.DataFrame) -> pd.DataFrame:
    """Macro and micro held-out MRR per (method, configuration), with ranks."""
    frame = _prepare(heldout_grid)
    if "entity_type" not in frame.columns:
        frame["entity_type"] = "class"
    frame["weighted"] = frame["mrr"] * frame["gold_count"]
    scores = (
        frame.groupby(["method", "hyperparameters"])
        .agg(
            units=("mrr", "size"),
            macro_mrr=("mrr", "mean"),
            weighted=("weighted", "sum"),
            gold_pairs=("gold_count", "sum"),
        )
        .reset_index()
    )
    scores["micro_mrr"] = scores["weighted"] / scores["gold_pairs"]
    for aggregation in ("macro", "micro"):
        scores[f"{aggregation}_rank"] = scores.groupby("method")[f"{aggregation}_mrr"].rank(
            ascending=False, method="min"
        )
    return scores.drop(columns=["weighted"])


def selection_regret(
    development: pd.DataFrame,
    heldout_grids: dict[str, pd.DataFrame],
    *,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> pd.DataFrame:
    picks = development_rule_picks(development, lambdas)
    rows: list[dict[str, object]] = []
    for tier, grid in heldout_grids.items():
        scores = heldout_config_scores(grid).set_index(["method", "hyperparameters"])
        configs = scores.groupby(level="method").size()
        for pick in picks.itertuples():
            key = (pick.method, pick.hyperparameters)
            if key not in scores.index:
                raise ValueError(f"{tier}: picked configuration missing from held-out grid: {key}")
            method_scores = scores.loc[pick.method]
            picked = scores.loc[key]
            row: dict[str, object] = {
                "tier": tier,
                "rule": pick.rule,
                "method": pick.method,
                "hyperparameters": pick.hyperparameters,
                "configurations": int(configs[pick.method]),
            }
            for aggregation in ("macro", "micro"):
                best = method_scores[f"{aggregation}_mrr"].max()
                row[f"heldout_{aggregation}_mrr"] = float(picked[f"{aggregation}_mrr"])
                row[f"best_{aggregation}_mrr"] = float(best)
                row[f"best_{aggregation}_hyperparameters"] = _argmax(method_scores[f"{aggregation}_mrr"])
                row[f"regret_{aggregation}"] = float(best - picked[f"{aggregation}_mrr"])
                row[f"rank_{aggregation}"] = int(picked[f"{aggregation}_rank"])
            rows.append(row)
    return pd.DataFrame(rows)


def run_selection_regret(
    *,
    development_csv: str | Path,
    heldout_grid_csvs: dict[str, str | Path],
    output_csv: str | Path,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> Path:
    result = selection_regret(
        pd.read_csv(development_csv),
        {tier: pd.read_csv(path) for tier, path in heldout_grid_csvs.items()},
        lambdas=lambdas,
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return output_path


def parse_tier_csv_arguments(values: Sequence[str]) -> dict[str, str]:
    """Parse ``tier=path`` arguments."""
    parsed: dict[str, str] = {}
    for value in values:
        tier, separator, path = value.partition("=")
        if not separator or not tier or not path:
            raise ValueError(f"Expected tier=path, got {value!r}.")
        parsed[tier] = path
    return parsed


__all__ = [
    "development_rule_picks",
    "heldout_config_scores",
    "parse_tier_csv_arguments",
    "run_selection_regret",
    "selection_regret",
]
