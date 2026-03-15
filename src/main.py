"""CLI entry point for running experiments and comparison analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.analysis.model_comparison import generate_model_comparison
from src.experiments.experiment_runner import run_experiments
from src.logging_utils import setup_logging

logger = logging.getLogger("src.main")


def _build_run_parser(parser: argparse.ArgumentParser) -> None:
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
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        action="store_true",
        help="Force-enable progress bars.",
    )
    progress_group.add_argument(
        "--no-progress",
        action="store_true",
        help="Force-disable progress bars.",
    )


def _build_compare_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-csv",
        default=None,
        help=(
            "Path to experiment results CSV. If omitted, the latest results/result_*.csv "
            "is used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/comparisons",
        help="Directory where comparison artifacts are written (default: results/comparisons).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run KG candidate-generation experiments and comparison reports."
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run retrieval experiments.")
    _build_run_parser(run_parser)

    compare_parser = subparsers.add_parser(
        "compare-models",
        help="Generate TF-IDF vs BM25 comparison artifacts from results CSV.",
    )
    _build_compare_parser(compare_parser)

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


def _run_experiment_cli(args: argparse.Namespace) -> int:
    show_progress: bool | None
    if args.progress:
        show_progress = True
    elif args.no_progress:
        show_progress = False
    else:
        show_progress = None

    existing_files = (
        _collect_existing_result_files() if args.output_csv_path is None else set()
    )
    run_experiments(
        config_path=args.config_path,
        output_csv_path=args.output_csv_path,
        show_progress=show_progress,
    )
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


def _run_compare_cli(args: argparse.Namespace) -> int:
    artifacts = generate_model_comparison(
        results_csv_path=args.results_csv,
        output_dir=args.output_dir,
    )
    print(f"Comparison Report Dir: {artifacts['output_dir']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()

    normalized_argv = list(sys.argv[1:] if argv is None else argv)
    if not normalized_argv:
        normalized_argv = ["run", *normalized_argv]
    elif normalized_argv[0] not in {"-h", "--help"} and normalized_argv[0].startswith("-"):
        normalized_argv = ["run", *normalized_argv]

    args = parser.parse_args(normalized_argv)

    setup_logging()
    try:
        if args.command == "compare-models":
            return _run_compare_cli(args)
        return _run_experiment_cli(args)
    except Exception as exc:
        logger.exception(
            "CLI execution failed (command=%s, config_path=%s, output_csv_path=%s, results_csv=%s, output_dir=%s)",
            getattr(args, "command", None),
            getattr(args, "config_path", None),
            getattr(args, "output_csv_path", None),
            getattr(args, "results_csv", None),
            getattr(args, "output_dir", None),
        )
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
