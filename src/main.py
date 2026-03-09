"""CLI entry point for running retrieval experiments."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.experiments.experiment_runner import run_experiments
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run KG candidate-generation experiments."
    )
    parser.add_argument(
        "--config-path",
        default="config/datasets.yaml",
        help="Path to dataset YAML config (default: config/datasets.yaml).",
    )
    parser.add_argument(
        "--output-csv-path",
        default=None,
        help=(
            "Output CSV path. If omitted, a run-stamped file is created in results/ "
            "(result_YYYYMMDD_HHMMSS_<gitsha>.csv)."
        ),
    )
    return parser


def _collect_existing_result_files(results_dir: Path = Path("results")) -> set[Path]:
    if not results_dir.exists():
        return set()
    return {path.resolve() for path in results_dir.glob("result_*.csv")}


def _resolve_new_result_file(
    existing_files: set[Path], results_dir: Path = Path("results")
) -> Path:
    current_files = {path.resolve() for path in results_dir.glob("result_*.csv")}
    new_files = sorted(current_files - existing_files, key=lambda p: p.stat().st_mtime)
    if not new_files:
        raise FileNotFoundError(
            "No new results file was created in 'results/' for this run."
        )
    return new_files[-1]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging()
    try:
        existing_files = (
            _collect_existing_result_files() if args.output_csv_path is None else set()
        )
        run_experiments(config_path=args.config_path, output_csv_path=args.output_csv_path)
        if args.output_csv_path:
            final_output_path = Path(args.output_csv_path)
            if not final_output_path.exists():
                raise FileNotFoundError(
                    f"Expected output CSV was not created: {final_output_path}"
                )
        else:
            final_output_path = _resolve_new_result_file(existing_files)

        print(f"Results CSV: {final_output_path}")
        return 0
    except Exception as exc:
        logger.exception("CLI execution failed")
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
