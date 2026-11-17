"""Zero-candidate share over every source class, without gold mappings (K1).

    KGCANDGEN_DATA_ROOT=/path/to/oaei_track_cache \
    python scripts/all_source_zero_share.py --label v1.2-sncs-r1

The grid diagnostics flag a configuration as ``feature_loss`` from the share of *gold*
queries that get no candidate. This script asks whether the same symptom shows up without
the gold standard: for every development pair and every TF-IDF, BM25 and character n-gram
grid cell it queries with *every* source owl:Class entity, labelled and preprocessed
exactly as in the canonical run, and counts the sources for which the retriever returns
no candidate (no target with a positive score).

Outputs (default ``results/all_source_zero_share/``):

- ``all_source_zero_share_<label>.csv``: one row per (pair, method, configuration) with
  the zero-candidate share over all sources, over gold sources and over non-gold sources.
- ``all_source_zero_share_<label>_summary.csv``: one row per configuration with the mean
  shares over pairs, the grid-diagnostic category from the canonical run, and the category
  the same rule gives when the all-source share replaces the gold-query share.
- ``all_source_zero_share_<label>.json``: settings, a consistency check against the
  canonical gold-query shares, category agreement and the separation between cells.

The configuration and the gold-query reference shares are read only from files listed in
the canonical run manifest. Gold mappings are used only to split the counts and for the
consistency check; the all-source share itself never reads them.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.grid_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_LOSS_THRESHOLD,
    GRID_METHODS,
    _reduces_to_summed_idf,
)
from src.config_loader import load_runtime_config  # noqa: E402
from src.experiments.experiment_runner import (  # noqa: E402
    CHAR_NGRAM_SENSITIVITY_METHOD,
    ModelRunSpec,
    _build_labels,
    _build_model_runs,
    _extract_owl_class_entities,
    _load_store_with_fallback,
    _preprocess_labels,
)
from src.preprocessing.text_preprocessor import validate_nltk_assets  # noqa: E402
from src.rdf_utils.alignment_parser import (  # noqa: E402
    filter_alignment_targets,
    load_alignment_targets,
)
from src.retrieval.bm25_retriever import Bm25Retriever  # noqa: E402
from src.retrieval.char_ngram_retriever import CharNgramRetriever  # noqa: E402
from src.retrieval.tfidf_retriever import TfidfRetriever  # noqa: E402

logger = logging.getLogger("all_source_zero_share")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def hp_key(hyperparameters: dict[str, object]) -> str:
    return json.dumps(hyperparameters, sort_keys=True, separators=(",", ":"))


def manifest_csv(manifest: dict, predicate) -> pd.DataFrame:
    matches = [p for p in manifest["artifacts_sha256"] if predicate(p)]
    if len(matches) != 1:
        raise SystemExit(f"Expected one matching artefact, found {matches}")
    return pd.read_csv(ROOT / matches[0])


def zero_candidate_flags(run: ModelRunSpec, target_entities: list[str], target_labels: list[str],
                         target_tokens: list[list[str]], target_joined: list[str],
                         source_labels: list[str], source_tokens: list[list[str]],
                         source_joined: list[str]) -> list[bool]:
    """Fit ``run`` on the targets as the canonical runner does and flag empty candidate lists."""
    if run.model_name == "tfidf":
        retriever = cast(TfidfRetriever, run.retriever)
        retriever.fit_preprocessed(target_entities, target_joined)
        return [not retriever.retrieve_preprocessed(q, k=1) for q in source_joined]
    if run.model_name == "bm25":
        retriever = cast(Bm25Retriever, run.retriever)
        retriever.fit_tokenized(target_entities, target_tokens)
        return [not retriever.retrieve_tokenized(q, k=1) for q in source_tokens]
    if run.model_name == CHAR_NGRAM_SENSITIVITY_METHOD:
        retriever = cast(CharNgramRetriever, run.retriever)
        retriever.fit(target_entities, target_labels)
        return [not retriever.retrieve(q, k=1) for q in source_labels]
    raise ValueError(f"Not a grid method: {run.model_name}")


def categorise(summary: pd.DataFrame, share_column: str, threshold: float) -> list[str]:
    """The grid-diagnostics rule, applied to ``share_column`` instead of the gold-query share."""
    best = summary.groupby("method")[share_column].transform("min")
    categories = []
    for row, excess in zip(summary.itertuples(), summary[share_column] - best):
        if row.failed_runs > 0:
            categories.append("failed")
        elif excess > threshold:
            categories.append("feature_loss")
        elif _reduces_to_summed_idf(row.method, json.loads(row.hyperparameters)):
            categories.append("reduced_scoring")
        else:
            categories.append("valid")
    return categories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "all_source_zero_share")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Restrict to these development pairs (smoke tests only).")
    parser.add_argument("--feature-loss-threshold", type=float, default=DEFAULT_FEATURE_LOSS_THRESHOLD)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest = json.loads((ROOT / "results" / "canonical" / args.label / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise SystemExit(f"Canonical run {args.label} is not complete.")
    reference = manifest_csv(manifest, lambda p: Path(p).name.startswith("result_")
                             and p.endswith(".csv") and "_errors" not in p)
    diagnostics = manifest_csv(manifest, lambda p: p.endswith("/grid_diagnostics.csv"))
    diagnostics["hyperparameters"] = diagnostics.hyperparameters.map(lambda h: hp_key(json.loads(h)))

    validate_nltk_assets()
    config = load_runtime_config(config_path=ROOT / manifest["config"])
    experiments = config.experiments
    runs = [r for r in _build_model_runs(experiments.tfidf_grid, experiments.bm25_grid,
                                         experiments.candidate_policy, experiments.char_ngram_grid)
            if r.model_name in GRID_METHODS]
    datasets = config.development_datasets
    names = list(datasets) if not args.datasets else args.datasets
    unknown = sorted(set(names) - set(datasets))
    if unknown:
        raise SystemExit(f"Unknown development pairs: {unknown}")

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for name in names:
        dataset = datasets[name]
        t0 = time.perf_counter()
        source_store = _load_store_with_fallback(dataset.source_rdf)
        target_store = _load_store_with_fallback(dataset.target_rdf)
        sources = _extract_owl_class_entities(source_store)
        targets = _extract_owl_class_entities(target_store)
        gold = filter_alignment_targets(load_alignment_targets(dataset.alignment_rdf, experiments.alignment_policy),
                                        set(sources), set(targets))
        target_labels = _build_labels(target_store, targets)
        source_labels = _build_labels(source_store, sources)
        target_tokens, target_joined = _preprocess_labels(target_labels)
        source_tokens, source_joined = _preprocess_labels(source_labels)
        is_gold = [s in gold for s in sources]
        n_gold = sum(is_gold)
        logger.info("%s: %d sources (%d gold), %d targets", name, len(sources), n_gold, len(targets))
        for run in runs:
            row: dict[str, object] = {
                "track": dataset.track, "dataset": name, "method": run.model_name,
                "hyperparameters": hp_key(run.hyperparameters), "source_entities": len(sources),
                "gold_sources": n_gold, "nongold_sources": len(sources) - n_gold,
                "target_entities": len(targets),
            }
            try:
                empty = zero_candidate_flags(run, targets, target_labels, target_tokens, target_joined,
                                             source_labels, source_tokens, source_joined)
            except Exception as exc:  # the same cells the canonical run records as failed
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                continue
            zero_all = sum(empty)
            zero_gold = sum(e for e, g in zip(empty, is_gold) if g)
            nongold = len(sources) - n_gold
            row.update(
                status="ok", error="",
                zero_all=zero_all, share_all=zero_all / len(sources),
                zero_gold=zero_gold, share_gold=zero_gold / n_gold if n_gold else float("nan"),
                zero_nongold=zero_all - zero_gold,
                share_nongold=(zero_all - zero_gold) / nongold if nongold else float("nan"),
            )
            rows.append(row)
        logger.info("%s done in %.1fs", name, time.perf_counter() - t0)

    out = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"all_source_zero_share_{args.label}" + ("_subset" if args.datasets else "")
    out.to_csv(args.output_dir / f"{stem}.csv", index=False)

    ok = out[out.status == "ok"]
    # Consistency: gold-source share must equal the canonical run's zero_candidate_query_share.
    ref = reference[reference.method.isin(GRID_METHODS)].copy()
    ref["hyperparameters"] = ref.hyperparameters.map(lambda h: hp_key(json.loads(h)))
    merged = ok.merge(ref[["dataset", "method", "hyperparameters", "zero_candidate_query_share"]],
                      on=["dataset", "method", "hyperparameters"], how="left")
    missing_ref = int(merged.zero_candidate_query_share.isna().sum())
    max_gold_diff = float((merged.share_gold - merged.zero_candidate_query_share).abs().max())

    summary = (ok.groupby(["method", "hyperparameters"], as_index=False)
               .agg(pairs=("dataset", "nunique"), mean_share_all=("share_all", "mean"),
                    mean_share_gold=("share_gold", "mean"), mean_share_nongold=("share_nongold", "mean")))
    failed = out[out.status == "failed"].groupby(["method", "hyperparameters"]).size()
    summary["failed_runs"] = [int(failed.get((r.method, r.hyperparameters), 0)) for r in summary.itertuples()]
    summary["excess_share_all"] = summary.mean_share_all - summary.groupby("method").mean_share_all.transform("min")
    summary["category_all_sources"] = categorise(summary, "mean_share_all", args.feature_loss_threshold)
    category = diagnostics.set_index(["method", "hyperparameters"])["category"]
    summary["category"] = [category.get((r.method, r.hyperparameters), "unknown") for r in summary.itertuples()]
    summary = summary.sort_values(["method", "mean_share_all", "hyperparameters"]).reset_index(drop=True)
    summary.to_csv(args.output_dir / f"{stem}_summary.csv", index=False)

    # Separation: smallest excess among feature-loss cells against the largest among the rest.
    separation = {}
    for method, sub in summary[summary.category != "failed"].groupby("method"):
        loss = sub[sub.category == "feature_loss"].excess_share_all
        rest = sub[sub.category != "feature_loss"].excess_share_all
        separation[method] = {
            "feature_loss_cells": int(loss.size), "other_cells": int(rest.size),
            "min_excess_feature_loss": float(loss.min()) if loss.size else None,
            "max_excess_other": float(rest.max()) if rest.size else None,
        }
    agree = summary.category == summary.category_all_sources
    meta = {
        "label": args.label, "config": manifest["config"], "canonical_commit": manifest["commit"],
        "commit": git("rev-parse", "HEAD"),
        "retrieval_code_matches_canonical": subprocess.run(
            ["git", "diff", "--quiet", manifest["commit"], "HEAD", "--", "src"], cwd=ROOT).returncode == 0,
        "datasets": names, "feature_loss_threshold": args.feature_loss_threshold,
        "rows": int(len(out)), "failed_rows": int((out.status == "failed").sum()),
        "consistency": {"rows_without_reference": missing_ref,
                        "max_abs_diff_gold_share_vs_canonical": max_gold_diff},
        "category_agreement": {"agree": int(agree.sum()), "cells": int(len(summary)),
                               "disagreements": summary.loc[~agree, ["method", "hyperparameters", "category",
                                                                     "category_all_sources"]].to_dict("records")},
        "separation": separation,
        "seconds": round(time.perf_counter() - started, 1),
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps({k: meta[k] for k in ("consistency", "category_agreement", "separation")}, indent=2))
    if missing_ref or max_gold_diff > 1e-12:
        raise SystemExit("Gold-source shares do not match the canonical run; see the JSON file.")


if __name__ == "__main__":
    main()
