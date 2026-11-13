"""Paired method comparisons from query-level output with a dataset-clustered bootstrap.

Each gold pair is a unit of observation. Queries from one dataset are not independent
(and KG entity types of one dataset share a source graph), so uncertainty is estimated by
resampling whole datasets with replacement. Two aggregations are reported side by side:

- micro: mean over all gold pairs, so large alignments dominate;
- macro: mean over evaluation units (dataset x entity type), each weighted equally.

Every delta is ``method_a - method_b``.

Confidence intervals come from the dataset-clustered percentile bootstrap. P-values do
not: with as few as five clusters the cluster bootstrap under-covers and returns p close to
zero whenever all datasets agree in sign. P-values therefore come from an exact
cluster-level sign-flip test (each dataset's paired contribution is negated or kept), which
is valid for any number of queries and whose smallest attainable two-sided value,
``2 / 2**n_datasets``, is reported as ``p_floor``. Holm's method adjusts them within each
(scope, metric, aggregation) family of comparisons.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_METRICS: tuple[str, ...] = ("mrr", "recall_at_1", "recall_at_10", "recall_at_50")
FIXED_METHODS: tuple[str, ...] = ("exact_match", "char_ngram")
PAIR_KEY: tuple[str, ...] = ("dataset", "entity_type", "source_entity", "gold_target")


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    m = len(p_values)
    order = np.argsort(p_values, kind="mergesort")
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _metric_values(gold_rank: pd.Series, metric: str) -> pd.Series:
    rank = gold_rank.astype(int)
    if metric == "mrr":
        return (1.0 / rank.where(rank > 0)).fillna(0.0)
    if metric.startswith("recall_at_"):
        k = int(metric.removeprefix("recall_at_"))
        return ((rank > 0) & (rank <= k)).astype(float)
    raise ValueError(f"Unsupported metric {metric!r}.")


def filter_to_selected(
    queries: pd.DataFrame, selected_settings: dict[str, dict[str, object]] | None
) -> pd.DataFrame:
    """Keep frozen settings for tuned methods and the fixed baselines; drop sensitivity rows."""
    keep = queries["method"].isin(FIXED_METHODS)
    for method, settings in (selected_settings or {}).items():
        encoded = json.dumps(settings, sort_keys=True, separators=(",", ":"))
        keep |= (queries["method"] == method) & (queries["hyperparameters"] == encoded)
    if not selected_settings:
        tuned = ~queries["method"].isin(FIXED_METHODS)
        counts = queries[tuned].groupby("method")["hyperparameters"].nunique()
        if (counts > 1).any():
            raise ValueError(
                "Query file holds several settings for "
                f"{sorted(counts[counts > 1].index)}; pass selected settings."
            )
        keep |= tuned
    return queries[keep & (queries["method"] != "char_ngram_sensitivity")].copy()


def _paired_frame(queries: pd.DataFrame, method_a: str, method_b: str, metric: str) -> pd.DataFrame:
    frames = []
    for method, column in ((method_a, "a"), (method_b, "b")):
        subset = queries[queries["method"] == method]
        if subset.duplicated(list(PAIR_KEY)).any():
            raise ValueError(f"Duplicate query rows for method {method!r}.")
        frames.append(
            subset.assign(**{column: _metric_values(subset["gold_rank"], metric)})[
                [*PAIR_KEY, column]
            ]
        )
    paired = frames[0].merge(frames[1], on=list(PAIR_KEY), how="inner", validate="one_to_one")
    paired["unit"] = paired["dataset"] + "|" + paired["entity_type"]
    return paired


def _bootstrap(
    paired: pd.DataFrame, resamples: int, rng: np.random.Generator
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    datasets = sorted(paired["dataset"].unique())
    # Per (dataset, unit) sums let each resample be aggregated without touching queries.
    unit_stats = (
        paired.assign(delta=paired["a"] - paired["b"])
        .groupby(["dataset", "unit"])
        .agg(n=("delta", "size"), sum_a=("a", "sum"), sum_b=("b", "sum"), sum_delta=("delta", "sum"))
        .reset_index()
    )
    unit_stats["unit_mean_delta"] = unit_stats["sum_delta"] / unit_stats["n"]
    per_dataset = (
        unit_stats.groupby("dataset")
        .agg(
            total_delta=("sum_delta", "sum"),
            total_n=("n", "sum"),
            unit_mean_sum=("unit_mean_delta", "sum"),
            unit_count=("unit", "size"),
        )
        .reindex(datasets)
    )
    total_delta = per_dataset["total_delta"].to_numpy(float)
    total_n = per_dataset["total_n"].to_numpy(float)
    unit_mean_sum = per_dataset["unit_mean_sum"].to_numpy(float)
    unit_count = per_dataset["unit_count"].to_numpy(float)

    # Resampling whole datasets: micro = pooled pair mean, macro = mean of unit means.
    draws = rng.integers(0, len(datasets), size=(resamples, len(datasets)))
    micro_boot = total_delta[draws].sum(axis=1) / total_n[draws].sum(axis=1)
    macro_boot = unit_mean_sum[draws].sum(axis=1) / unit_count[draws].sum(axis=1)
    micro_point = total_delta.sum() / total_n.sum()
    macro_point = unit_mean_sum.sum() / unit_count.sum()

    point = {
        "n_datasets": float(len(datasets)),
        "n_units": float(len(unit_stats)),
        "n_pairs": float(unit_stats["n"].sum()),
        "mean_a_micro": unit_stats["sum_a"].sum() / unit_stats["n"].sum(),
        "mean_b_micro": unit_stats["sum_b"].sum() / unit_stats["n"].sum(),
        "mean_a_macro": float((unit_stats["sum_a"] / unit_stats["n"]).mean()),
        "mean_b_macro": float((unit_stats["sum_b"] / unit_stats["n"]).mean()),
        "delta_micro": float(micro_point),
        "delta_macro": float(macro_point),
    }
    contributions = {
        "micro": (total_delta, total_n),
        "macro": (unit_mean_sum, unit_count),
    }
    return point, micro_boot, macro_boot, contributions


EXACT_SIGN_FLIP_MAX_CLUSTERS = 16


def cluster_sign_flip_p(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    rng: np.random.Generator,
    monte_carlo_draws: int = 100_000,
) -> float:
    """Two-sided p for sum(numerators)/sum(denominators) under per-cluster sign flips.

    Exact enumeration up to ``EXACT_SIGN_FLIP_MAX_CLUSTERS`` clusters, else Monte Carlo.
    """
    observed = abs(numerators.sum() / denominators.sum())
    if observed == 0.0:
        return 1.0
    n = numerators.size
    if n <= EXACT_SIGN_FLIP_MAX_CLUSTERS:
        signs = 1.0 - 2.0 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    else:
        signs = rng.choice((-1.0, 1.0), size=(monte_carlo_draws, n))
    flipped = np.abs(signs @ numerators) / denominators.sum()
    return float(np.mean(flipped >= observed - 1e-12))


def paired_inference(
    queries: pd.DataFrame,
    *,
    scope: str,
    methods: Sequence[str] | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    resamples: int = 10_000,
    seed: int = 20260326,
) -> pd.DataFrame:
    """All pairwise comparisons for one scope, with Holm adjustment per metric/aggregation."""
    present = [m for m in ("tfidf", "bm25", "exact_match", "char_ngram") if m in set(queries["method"])]
    present += sorted(set(queries["method"]) - set(present))
    methods = list(methods) if methods is not None else present
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for method_a, method_b in itertools.combinations(methods, 2):
            paired = _paired_frame(queries, method_a, method_b, metric)
            if paired.empty:
                continue
            rng = np.random.default_rng(seed)
            point, micro_boot, macro_boot, contributions = _bootstrap(paired, resamples, rng)
            p_values = {
                aggregation: cluster_sign_flip_p(numerators, denominators, rng=rng)
                for aggregation, (numerators, denominators) in contributions.items()
            }
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "delta_definition": "method_a - method_b",
                    **point,
                    "ci_low_micro": float(np.percentile(micro_boot, 2.5)),
                    "ci_high_micro": float(np.percentile(micro_boot, 97.5)),
                    "ci_low_macro": float(np.percentile(macro_boot, 2.5)),
                    "ci_high_macro": float(np.percentile(macro_boot, 97.5)),
                    "p_micro": p_values["micro"],
                    "p_macro": p_values["macro"],
                    "p_floor": float(min(1.0, 2.0 / 2 ** point["n_datasets"])),
                    "p_method": "exact cluster sign-flip"
                    if point["n_datasets"] <= EXACT_SIGN_FLIP_MAX_CLUSTERS
                    else "Monte Carlo cluster sign-flip",
                    "resamples": resamples,
                    "seed": seed,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for aggregation in ("micro", "macro"):
        result[f"p_{aggregation}_holm"] = np.nan
        for _, group in result.groupby("metric"):
            result.loc[group.index, f"p_{aggregation}_holm"] = holm_adjust(
                group[f"p_{aggregation}"].tolist()
            )
    return result


def run_paired_inference(
    query_csvs: Sequence[str | Path],
    *,
    selected_settings_path: str | Path | None = None,
    output_csv: str | Path,
    metrics: Sequence[str] = DEFAULT_METRICS,
    resamples: int = 10_000,
    seed: int = 20260326,
    by_entity_type: bool = True,
) -> Path:
    """Run every scope in each query CSV (whole file, plus per entity type when several)."""
    selected = None
    if selected_settings_path is not None:
        payload = json.loads(Path(selected_settings_path).read_text(encoding="utf-8"))
        selected = {
            method: entry["hyperparameters"]
            for method, entry in payload["selected_settings"].items()
        }
    frames = []
    for query_csv in query_csvs:
        queries = filter_to_selected(pd.read_csv(query_csv), selected)
        track = "+".join(sorted(queries["track"].astype(str).unique()))
        frames.append(paired_inference(queries, scope=f"{track}:all", metrics=metrics, resamples=resamples, seed=seed))
        entity_types = sorted(queries["entity_type"].unique())
        if by_entity_type and len(entity_types) > 1:
            for entity_type in entity_types:
                frames.append(
                    paired_inference(
                        queries[queries["entity_type"] == entity_type],
                        scope=f"{track}:{entity_type}",
                        metrics=metrics,
                        resamples=resamples,
                        seed=seed,
                    )
                )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
    return output_path
