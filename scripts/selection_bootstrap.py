"""Bootstrap uncertainty of the development-based configuration selection (R2-2).

    python scripts/selection_bootstrap.py --label v1.2-sncs-r1 --resamples 1000 \
        --output-csv results/selection_bootstrap/selection_bootstrap_v1.2-sncs-r1.csv

Development datasets are resampled with replacement within each track (stratified by track,
so every resample keeps all three tracks), and the stability-aware rule (lambda = 0.5, the
rule used in the paper) is re-applied to each resample. For every configuration picked at
least once the output gives how often it was picked, its grid-diagnostic category, its
full-sample development MRR, and its held-out macro MRR and regret against the post-hoc best
on each held-out tier. Inputs are read only from files listed in the canonical run manifest.

The settings file (``.json`` next to the CSV) also records the configuration picked by a
z-score variant of the rule on the full development tier: each track's mean MRR per cell is
standardised to a z-score within the track (population SD) instead of a rank score, and the
cell maximising mean(z) - lambda * SD(z) over tracks is picked, ties to the smallest
configuration string.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.heldout_selection import _build_selection_summary  # noqa: E402
from src.analysis.selection_regret import _prepare, heldout_config_scores  # noqa: E402


def manifest_csv(manifest: dict, predicate) -> pd.DataFrame:
    matches = [p for p in manifest["artifacts_sha256"] if predicate(p)]
    if len(matches) != 1:
        raise SystemExit(f"Expected one matching artefact, found {matches}")
    return pd.read_csv(ROOT / matches[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--lambda-penalty", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads((ROOT / "results" / "canonical" / args.label / "manifest.json").read_text())
    development = _prepare(manifest_csv(manifest, lambda p: Path(p).name.startswith("result_")
                                        and p.endswith(".csv") and "_errors" not in p))
    diagnostics = manifest_csv(manifest, lambda p: p.endswith("/grid_diagnostics.csv"))
    diagnostics["hyperparameters"] = diagnostics.hyperparameters.map(
        lambda h: json.dumps(json.loads(h), sort_keys=True, separators=(",", ":")))
    category = diagnostics.set_index(["method", "hyperparameters"])["category"]
    heldout = {
        "kg": manifest_csv(manifest, lambda p: Path(p).name.startswith("heldout_grid_result_") and "_errors" not in p),
        "bioml": manifest_csv(manifest, lambda p: Path(p).name.startswith("heldout_class_grid_result_")
                              and "_errors" not in p),
    }
    heldout_scores = {tier: heldout_config_scores(grid).set_index(["method", "hyperparameters"])
                      for tier, grid in heldout.items()}

    def pick(frame: pd.DataFrame) -> dict[str, str]:
        summary = _build_selection_summary(frame, lambda_penalty=args.lambda_penalty)
        chosen = summary[summary["selected"]]
        return dict(zip(chosen["method"], chosen["hyperparameters"]))

    full_pick = pick(development)

    def zscore_pick(frame: pd.DataFrame) -> dict[str, str]:
        picks = {}
        for method, sub in frame.groupby("method"):
            track = sub.groupby(["track", "hyperparameters"])["mrr"].mean().reset_index()
            track["z"] = track.groupby("track")["mrr"].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
            agg = track.groupby("hyperparameters")["z"].agg(mu="mean", sd=lambda x: x.std(ddof=0))
            agg["score"] = agg["mu"] - args.lambda_penalty * agg["sd"]
            best = agg.reset_index().sort_values(["score", "hyperparameters"], ascending=[False, True])
            picks[method] = str(best.iloc[0]["hyperparameters"])
        return picks
    datasets_by_track = {t: sorted(g.dataset.unique()) for t, g in development.groupby("track")}
    by_dataset = {d: g for d, g in development.groupby("dataset")}
    rng = np.random.default_rng(args.seed)
    counts: dict[tuple[str, str], int] = {}
    for _ in range(args.resamples):
        parts = []
        for track, names in datasets_by_track.items():
            for i, name in enumerate(rng.choice(names, size=len(names), replace=True)):
                parts.append(by_dataset[name].assign(dataset=f"{name}#{i}"))
        for method, hp in pick(pd.concat(parts, ignore_index=True)).items():
            counts[(method, hp)] = counts.get((method, hp), 0) + 1

    dev_mrr = development.groupby(["method", "hyperparameters"])["mrr"].mean()
    rows = []
    for (method, hp), n in sorted(counts.items(), key=lambda kv: (kv[0][0], -kv[1])):
        row = {"method": method, "hyperparameters": hp, "picks": n, "share": n / args.resamples,
               "full_sample_pick": full_pick.get(method) == hp,
               "category": category.get((method, hp), "unknown"),
               "development_mean_mrr": float(dev_mrr[(method, hp)])}
        for tier, scores in heldout_scores.items():
            complete = scores[scores["complete"]].loc[method]
            row[f"{tier}_macro_mrr"] = float(scores.loc[(method, hp), "macro_mrr"])
            row[f"{tier}_regret_macro"] = float(complete["macro_mrr"].max() - row[f"{tier}_macro_mrr"])
        rows.append(row)
    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    meta = {"label": args.label, "resamples": args.resamples, "lambda": args.lambda_penalty,
            "seed": args.seed, "stratified_by": "track", "unit": "dataset",
            "datasets_per_track": {t: len(v) for t, v in datasets_by_track.items()},
            "full_sample_pick": full_pick, "zscore_rule_pick": zscore_pick(development)}
    args.output_csv.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
