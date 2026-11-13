"""Keep tests from writing into the repository's real ``results/`` directory.

Runners fall back to ``build_timestamped_results_path`` when a test gives no output path.
``isolate_results(module_names)`` patches that helper in the named runner modules so the
fallback lands in a temporary directory, removed when the test module finishes. Call the
returned ``start``/``stop`` from ``setUpModule``/``tearDownModule``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.run_metadata import build_timestamped_results_path


def isolate_results(*module_names: str):
    state: dict[str, object] = {}

    def start() -> None:
        tmpdir = Path(tempfile.mkdtemp(prefix="kgcandgen-results-"))

        def _redirected(prefix: str, *, directory: str | Path = "results") -> Path:
            return build_timestamped_results_path(prefix, directory=tmpdir / Path(directory).name)

        patchers = [patch(f"{name}.build_timestamped_results_path", _redirected) for name in module_names]
        for patcher in patchers:
            patcher.start()
        state.update(tmpdir=tmpdir, patchers=patchers)

    def stop() -> None:
        for patcher in state.get("patchers", []):
            patcher.stop()
        if "tmpdir" in state:
            shutil.rmtree(state["tmpdir"], ignore_errors=True)

    return start, stop
