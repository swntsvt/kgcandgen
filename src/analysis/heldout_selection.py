"""Stability-aware held-out hyperparameter selection utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.config_loader import load_runtime_config

REQUIRED_COLUMNS = {"dataset", "track", "method", "hyperparameters", "mrr"}
SUPPORTED_METHODS = {"tfidf", "bm25"}


class HeldoutSelectionValidationError(ValueError):
    """Raised when held-out selection input data is invalid."""


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
        raise FileNotFoundError("No results/result_*.csv files found for held-out selection.")
    return candidates[-1].resolve()


def _canonicalize_hyperparameters(value: str) -> str:
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HeldoutSelectionValidationError(
            f"Malformed hyperparameters JSON: {value}"
        ) from exc

    if not isinstance(data, dict):
        raise HeldoutSelectionValidationError(
            f"Hyperparameters must be a JSON object: {value}"
        )
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _validate_results_frame(frame: pd.DataFrame, *, development_datasets: set[str]) -> pd.DataFrame:
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise HeldoutSelectionValidationError(
            "Missing required held-out selection column(s): " + ", ".join(missing_columns)
        )

    filtered = frame[frame["method"].astype(str).isin(SUPPORTED_METHODS)].copy()
    if filtered.empty:
        raise HeldoutSelectionValidationError("No TF-IDF or BM25 rows found in results CSV.")

    filtered["dataset"] = filtered["dataset"].astype(str)
    filtered["track"] = filtered["track"].astype(str)
    filtered["method"] = filtered["method"].astype(str)
    filtered["mrr"] = filtered["mrr"].astype(float)
    filtered["hyperparameters"] = filtered["hyperparameters"].astype(str).apply(
        _canonicalize_hyperparameters
    )

    unknown = sorted(set(filtered["dataset"]) - development_datasets)
    if unknown:
        raise HeldoutSelectionValidationError(
            "Results CSV contains dataset(s) outside development_datasets: " + ", ".join(unknown)
        )
    return filtered


def _normalize_rank_score(rank: int, candidate_count: int) -> float:
    if candidate_count <= 1:
        return 1.0
    return float(1.0 - ((rank - 1) / (candidate_count - 1)))


def _population_std(values: list[float]) -> float:
    if not values:
        return float("nan")
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))


def _build_method_summary(method_frame: pd.DataFrame, *, lambda_penalty: float) -> pd.DataFrame:
    track_scores: list[dict[str, object]] = []
    grouped = (
        method_frame.groupby(["track", "hyperparameters"], as_index=False)
        .agg(track_mrr_mean=("mrr", "mean"))
    )
    all_tracks = sorted(str(track) for track in grouped["track"].unique())
    all_hyperparameters = sorted(str(value) for value in grouped["hyperparameters"].unique())

    for track, track_group in grouped.groupby("track", sort=True):
        sorted_group = track_group.sort_values(
            ["track_mrr_mean", "hyperparameters"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        candidate_count = len(sorted_group)
        for index, row in enumerate(sorted_group.itertuples(index=False), start=1):
            track_scores.append(
                {
                    "track": str(track),
                    "hyperparameters": str(row.hyperparameters),
                    "track_mrr_mean": float(row.track_mrr_mean),
                    "track_rank": index,
                    "track_score": _normalize_rank_score(index, candidate_count),
                }
            )

    track_frame = pd.DataFrame(track_scores)
    if track_frame.empty:
        raise HeldoutSelectionValidationError(
            f"No valid rows available for method '{method_frame['method'].iloc[0]}'."
        )

    summary_rows: list[dict[str, object]] = []
    method_name = str(method_frame["method"].iloc[0])
    track_candidate_counts = (
        track_frame.groupby("track")["hyperparameters"].nunique().astype(int).to_dict()
    )
    for hyperparameters in all_hyperparameters:
        candidate_group = track_frame[track_frame["hyperparameters"] == hyperparameters].copy()
        observed_by_track = {
            str(row.track): row
            for row in candidate_group.sort_values(["track"], kind="mergesort").itertuples(index=False)
        }

        completed_rows: list[dict[str, object]] = []
        for track in all_tracks:
            observed = observed_by_track.get(track)
            if observed is None:
                completed_rows.append(
                    {
                        "track": track,
                        "hyperparameters": hyperparameters,
                        "track_mrr_mean": None,
                        "track_rank": int(track_candidate_counts[track]) + 1,
                        "track_score": 0.0,
                        "status": "missing",
                    }
                )
                continue

            completed_rows.append(
                {
                    "track": track,
                    "hyperparameters": hyperparameters,
                    "track_mrr_mean": float(observed.track_mrr_mean),
                    "track_rank": int(observed.track_rank),
                    "track_score": float(observed.track_score),
                    "status": "observed",
                }
            )

        completed_group = pd.DataFrame(completed_rows).sort_values(["track"], kind="mergesort").reset_index(drop=True)
        scores = [float(value) for value in completed_group["track_score"]]
        mu = float(sum(scores) / len(scores))
        sigma = _population_std(scores)
        heldout_score = float(mu - (lambda_penalty * sigma))

        track_mrr_means = {
            str(row.track): (
                round(float(row.track_mrr_mean), 12)
                if row.track_mrr_mean is not None
                else None
            )
            for row in completed_group.itertuples(index=False)
        }
        track_ranks = {
            str(row.track): int(row.track_rank)
            for row in completed_group.itertuples(index=False)
        }
        track_normalized_scores = {
            str(row.track): round(float(row.track_score), 12)
            for row in completed_group.itertuples(index=False)
        }
        track_statuses = {
            str(row.track): str(row.status)
            for row in completed_group.itertuples(index=False)
        }

        summary_rows.append(
            {
                "method": method_name,
                "hyperparameters": str(hyperparameters),
                "tracks_observed": int(sum(1 for status in track_statuses.values() if status == "observed")),
                "tracks_total": int(len(all_tracks)),
                "track_mrr_means_json": json.dumps(
                    track_mrr_means, sort_keys=True, separators=(",", ":")
                ),
                "track_ranks_json": json.dumps(
                    track_ranks, sort_keys=True, separators=(",", ":")
                ),
                "track_normalized_scores_json": json.dumps(
                    track_normalized_scores, sort_keys=True, separators=(",", ":")
                ),
                "track_statuses_json": json.dumps(
                    track_statuses, sort_keys=True, separators=(",", ":")
                ),
                "mu": mu,
                "sigma": sigma,
                "heldout_score": heldout_score,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["heldout_score", "mu", "sigma", "hyperparameters"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary["selection_rank"] = range(1, len(summary) + 1)
    summary["selected"] = False
    if not summary.empty:
        summary.loc[0, "selected"] = True
    return summary


def _build_selection_summary(frame: pd.DataFrame, *, lambda_penalty: float) -> pd.DataFrame:
    method_summaries: list[pd.DataFrame] = []
    for method in sorted(SUPPORTED_METHODS):
        method_frame = frame[frame["method"] == method].copy()
        if method_frame.empty:
            raise HeldoutSelectionValidationError(f"Missing method rows for '{method}'.")
        method_summaries.append(
            _build_method_summary(method_frame, lambda_penalty=lambda_penalty)
        )

    summary = pd.concat(method_summaries, ignore_index=True)
    return summary.sort_values(["method", "selection_rank"], kind="mergesort").reset_index(drop=True)


def _selected_settings_payload(summary: pd.DataFrame) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for method in sorted(SUPPORTED_METHODS):
        selected = summary[(summary["method"] == method) & (summary["selected"])].reset_index(drop=True)
        if len(selected) != 1:
            raise HeldoutSelectionValidationError(
                f"Expected exactly one selected setting for method '{method}'."
            )
        row = selected.iloc[0]
        payload[method] = {
            "method": method,
            "hyperparameters": json.loads(str(row["hyperparameters"])),
            "hyperparameters_json": str(row["hyperparameters"]),
            "mu": float(row["mu"]),
            "sigma": float(row["sigma"]),
            "heldout_score": float(row["heldout_score"]),
            "tracks_observed": int(row["tracks_observed"]),
        }
    return payload


def _write_manifest(
    output_path: Path,
    *,
    source_csv: Path,
    config_path: Path,
    policy: dict[str, object],
    selected_settings: dict[str, dict[str, object]],
    artifacts: dict[str, Path],
) -> None:
    lines = [
        "# Held-Out Setting Selection",
        "",
        "## Scope",
        "",
        f"- Source development results CSV: `{source_csv}`",
        f"- Runtime config: `{config_path}`",
        "- Selection uses development-track results only.",
        "",
        "## Policy",
        "",
        f"- Metric: `{policy['metric']}`",
        f"- Lambda: `{policy['lambda']}`",
        f"- Weighting: `{policy['weighting']}`",
        f"- Ranking: `{policy['ranking']}`",
        "- Tie-breaks: higher heldout score, higher mu, lower sigma, canonical hyperparameters JSON ascending.",
        "",
        "## Winners",
        "",
        f"- TF-IDF: `{selected_settings['tfidf']['hyperparameters_json']}`",
        f"- BM25: `{selected_settings['bm25']['hyperparameters_json']}`",
        "",
        "## Artifacts",
        "",
        f"- Summary CSV: `{artifacts['heldout_selection_summary']}`",
        f"- Selected settings JSON: `{artifacts['heldout_selected_settings']}`",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_heldout_selection(
    results_csv_path: str | Path | None,
    config_path: str | Path = "config/runtime.yaml",
    output_dir: str | Path = "results/comparisons",
) -> dict[str, Path]:
    """Generate deterministic held-out setting selection artifacts."""
    source_csv = _resolve_results_csv(results_csv_path)
    runtime_config = load_runtime_config(config_path=config_path, require_heldout=True)
    if runtime_config.heldout is None:
        raise HeldoutSelectionValidationError(
            "Held-out selection requires a 'heldout' config section."
        )
    policy = runtime_config.heldout.selection.as_dict()
    frame = pd.read_csv(source_csv)
    validated = _validate_results_frame(
        frame,
        development_datasets=set(runtime_config.development_datasets.keys()),
    )
    summary = _build_selection_summary(
        validated,
        lambda_penalty=float(policy["lambda"]),
    )
    selected_settings = _selected_settings_payload(summary)

    run_id = source_csv.stem
    output_root = Path(output_dir) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "heldout_selection_summary.csv"
    selected_settings_path = output_root / "heldout_selected_settings.json"
    manifest_path = output_root / "heldout_selection_manifest.md"

    summary.to_csv(summary_path, index=False)
    selected_settings_path.write_text(
        json.dumps(
            {
                "source_csv": str(source_csv),
                "config_path": str(Path(config_path).resolve()),
                "policy": policy,
                "selected_settings": selected_settings,
                "heldout_datasets": sorted(runtime_config.heldout_datasets.keys()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "source_csv": source_csv,
        "output_dir": output_root,
        "heldout_selection_summary": summary_path,
        "heldout_selected_settings": selected_settings_path,
        "heldout_selection_manifest": manifest_path,
    }
    _write_manifest(
        manifest_path,
        source_csv=source_csv,
        config_path=Path(config_path).resolve(),
        policy=policy,
        selected_settings=selected_settings,
        artifacts=artifacts,
    )
    return artifacts
