"""Shared run metadata and timestamped artifact path helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess


def get_git_short_sha(*, repo_root: Path | None = None) -> str:
    """Return short git SHA for `repo_root`, or 'nogit' when unavailable."""
    resolved_repo_root = repo_root if repo_root is not None else Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=resolved_repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "nogit"


def build_timestamped_results_path(prefix: str, *, directory: str | Path = "results") -> Path:
    """Build a `results/<prefix>_<timestamp>_<gitsha>.csv` path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha = get_git_short_sha()
    return Path(directory) / f"{prefix}_{timestamp}_{git_sha}.csv"
