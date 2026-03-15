"""Utilities to configure plotting cache paths in restricted environments."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_plot_environment() -> None:
    """Set writable cache dirs for matplotlib/fontconfig if not already configured."""
    base_cache_dir = Path(tempfile.gettempdir()) / "kgcandgen_plot_cache"
    mpl_cache_dir = base_cache_dir / "matplotlib"
    xdg_cache_dir = base_cache_dir / "xdg_cache"

    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
