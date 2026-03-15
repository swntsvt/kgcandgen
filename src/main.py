"""CLI entry point for running experiments and comparison analysis."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import sys
import time
from pathlib import Path

from src.experiments.experiment_runner import run_experiments
from src.logging_utils import setup_logging

logger = logging.getLogger("src.main")


def generate_model_comparison(*, results_csv_path: str | Path | None, output_dir: str | Path) -> dict[str, Path]:
    from src.analysis.model_comparison import generate_model_comparison as _impl

    return _impl(results_csv_path=results_csv_path, output_dir=output_dir)


def generate_tfidf_sensitivity(
    *, results_csv_path: str | Path | None, config_path: str | Path, output_dir: str | Path
) -> dict[str, Path]:
    from src.analysis.tfidf_sensitivity import generate_tfidf_sensitivity as _impl

    return _impl(results_csv_path=results_csv_path, config_path=config_path, output_dir=output_dir)


def generate_bm25_sensitivity(
    *, results_csv_path: str | Path | None, config_path: str | Path, output_dir: str | Path
) -> dict[str, Path]:
    from src.analysis.bm25_sensitivity import generate_bm25_sensitivity as _impl

    return _impl(results_csv_path=results_csv_path, config_path=config_path, output_dir=output_dir)


def generate_depth_analysis(*, results_csv_path: str | Path | None, output_dir: str | Path) -> dict[str, Path]:
    from src.analysis.depth_analysis import generate_depth_analysis as _impl

    return _impl(results_csv_path=results_csv_path, output_dir=output_dir)


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


def _build_tfidf_sensitivity_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-csv",
        default=None,
        help=(
            "Path to experiment results CSV. If omitted, the latest results/result_*.csv "
            "is used."
        ),
    )
    parser.add_argument(
        "--config-path",
        default="config/datasets.yaml",
        help="Path to dataset YAML config (default: config/datasets.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/comparisons",
        help="Directory where sensitivity artifacts are written (default: results/comparisons).",
    )


def _build_bm25_sensitivity_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-csv",
        default=None,
        help=(
            "Path to experiment results CSV. If omitted, the latest results/result_*.csv "
            "is used."
        ),
    )
    parser.add_argument(
        "--config-path",
        default="config/datasets.yaml",
        help="Path to dataset YAML config (default: config/datasets.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/comparisons",
        help="Directory where BM25 sensitivity artifacts are written (default: results/comparisons).",
    )


def _build_depth_analysis_parser(parser: argparse.ArgumentParser) -> None:
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
        help="Directory where depth-analysis artifacts are written (default: results/comparisons).",
    )


def _build_full_run_parser(parser: argparse.ArgumentParser) -> None:
    _build_run_parser(parser)
    parser.add_argument(
        "--output-dir",
        default="results/comparisons",
        help="Directory where report artifacts are written (default: results/comparisons).",
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

    tfidf_sensitivity_parser = subparsers.add_parser(
        "tfidf-sensitivity",
        help="Generate TF-IDF hyperparameter sensitivity artifacts from results CSV.",
    )
    _build_tfidf_sensitivity_parser(tfidf_sensitivity_parser)

    bm25_sensitivity_parser = subparsers.add_parser(
        "bm25-sensitivity",
        help="Generate BM25 hyperparameter sensitivity artifacts from results CSV.",
    )
    _build_bm25_sensitivity_parser(bm25_sensitivity_parser)

    depth_analysis_parser = subparsers.add_parser(
        "depth-analysis",
        help="Generate retrieval-depth behavior artifacts from results CSV.",
    )
    _build_depth_analysis_parser(depth_analysis_parser)

    full_run_parser = subparsers.add_parser(
        "full-run",
        help="Run experiments and generate all analysis reports in one command.",
    )
    _build_full_run_parser(full_run_parser)

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
    show_progress = _resolve_show_progress(args)
    existing_files = _collect_existing_result_files() if args.output_csv_path is None else set()
    run_experiments(
        config_path=args.config_path,
        output_csv_path=args.output_csv_path,
        show_progress=show_progress,
    )
    final_output_path = _resolve_results_csv_from_run(
        output_csv_path=args.output_csv_path,
        existing_files=existing_files,
    )

    print(f"Results CSV: {final_output_path}")
    return 0


def _resolve_show_progress(args: argparse.Namespace) -> bool | None:
    if args.progress:
        return True
    if args.no_progress:
        return False
    return None


def _resolve_results_csv_from_run(
    *,
    output_csv_path: str | None,
    existing_files: set[Path],
) -> Path:
    if output_csv_path:
        final_output_path = Path(output_csv_path)
        if not final_output_path.exists():
            raise FileNotFoundError(
                f"Expected output CSV was not created: {final_output_path}"
            )
        return final_output_path
    return _resolve_new_result_file(existing_files)


def _run_compare_cli(args: argparse.Namespace) -> int:
    artifacts = generate_model_comparison(
        results_csv_path=args.results_csv,
        output_dir=args.output_dir,
    )
    print(f"Comparison Report Dir: {artifacts['output_dir']}")
    return 0


def _run_tfidf_sensitivity_cli(args: argparse.Namespace) -> int:
    artifacts = generate_tfidf_sensitivity(
        results_csv_path=args.results_csv,
        config_path=args.config_path,
        output_dir=args.output_dir,
    )
    print(f"TF-IDF Sensitivity Dir: {artifacts['output_dir']}")
    return 0


def _run_bm25_sensitivity_cli(args: argparse.Namespace) -> int:
    artifacts = generate_bm25_sensitivity(
        results_csv_path=args.results_csv,
        config_path=args.config_path,
        output_dir=args.output_dir,
    )
    print(f"BM25 Sensitivity Dir: {artifacts['output_dir']}")
    return 0


def _run_depth_analysis_cli(args: argparse.Namespace) -> int:
    artifacts = generate_depth_analysis(
        results_csv_path=args.results_csv,
        output_dir=args.output_dir,
    )
    print(f"Depth Analysis Dir: {artifacts['output_dir']}")
    return 0


def _run_full_run_cli(args: argparse.Namespace) -> int:
    started_at = datetime.now()
    start_perf = time.perf_counter()
    show_progress = _resolve_show_progress(args)
    existing_files = _collect_existing_result_files() if args.output_csv_path is None else set()

    stage_status: dict[str, str] = {}

    try:
        stage_name = "run_experiments"
        logger.info(
            "Starting full-run stage=%s config_path=%s output_csv_path=%s progress=%s",
            stage_name,
            args.config_path,
            args.output_csv_path,
            show_progress,
        )
        run_experiments(
            config_path=args.config_path,
            output_csv_path=args.output_csv_path,
            show_progress=show_progress,
        )
        stage_status[stage_name] = "success"
        logger.info("Completed full-run stage=%s", stage_name)

        results_csv = _resolve_results_csv_from_run(
            output_csv_path=args.output_csv_path,
            existing_files=existing_files,
        )

        stage_name = "model_comparison"
        logger.info(
            "Starting full-run stage=%s results_csv=%s output_dir=%s",
            stage_name,
            results_csv,
            args.output_dir,
        )
        model_artifacts = generate_model_comparison(
            results_csv_path=results_csv,
            output_dir=args.output_dir,
        )
        stage_status[stage_name] = "success"
        logger.info("Completed full-run stage=%s", stage_name)

        stage_name = "tfidf_sensitivity"
        logger.info(
            "Starting full-run stage=%s results_csv=%s config_path=%s output_dir=%s",
            stage_name,
            results_csv,
            args.config_path,
            args.output_dir,
        )
        tfidf_artifacts = generate_tfidf_sensitivity(
            results_csv_path=results_csv,
            config_path=args.config_path,
            output_dir=args.output_dir,
        )
        stage_status[stage_name] = "success"
        logger.info("Completed full-run stage=%s", stage_name)

        stage_name = "bm25_sensitivity"
        logger.info(
            "Starting full-run stage=%s results_csv=%s config_path=%s output_dir=%s",
            stage_name,
            results_csv,
            args.config_path,
            args.output_dir,
        )
        bm25_artifacts = generate_bm25_sensitivity(
            results_csv_path=results_csv,
            config_path=args.config_path,
            output_dir=args.output_dir,
        )
        stage_status[stage_name] = "success"
        logger.info("Completed full-run stage=%s", stage_name)

        stage_name = "depth_analysis"
        logger.info(
            "Starting full-run stage=%s results_csv=%s output_dir=%s",
            stage_name,
            results_csv,
            args.output_dir,
        )
        depth_artifacts = generate_depth_analysis(
            results_csv_path=results_csv,
            output_dir=args.output_dir,
        )
        stage_status[stage_name] = "success"
        logger.info("Completed full-run stage=%s", stage_name)
    except Exception as exc:
        stage_status[stage_name] = "failed"
        logger.exception(
            "Full-run stage failed stage=%s results_csv=%s config_path=%s output_dir=%s error=%s: %s",
            stage_name,
            locals().get("results_csv"),
            args.config_path,
            args.output_dir,
            type(exc).__name__,
            exc,
        )
        raise

    ended_at = datetime.now()
    elapsed_seconds = time.perf_counter() - start_perf
    manifest_root = Path(model_artifacts["output_dir"])
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / "full_run_manifest.txt"
    manifest_path.write_text(
        "\n".join(
            [
                f"generated_at_start={started_at.isoformat(timespec='seconds')}",
                f"generated_at_end={ended_at.isoformat(timespec='seconds')}",
                f"elapsed_seconds={elapsed_seconds:.6f}",
                f"config_path={Path(args.config_path).resolve()}",
                f"results_csv={Path(results_csv).resolve()}",
                f"stage_run_experiments={stage_status.get('run_experiments', 'not_run')}",
                f"stage_model_comparison={stage_status.get('model_comparison', 'not_run')}",
                f"stage_tfidf_sensitivity={stage_status.get('tfidf_sensitivity', 'not_run')}",
                f"stage_bm25_sensitivity={stage_status.get('bm25_sensitivity', 'not_run')}",
                f"stage_depth_analysis={stage_status.get('depth_analysis', 'not_run')}",
                f"model_comparison_output_dir={Path(model_artifacts['output_dir']).resolve()}",
                f"tfidf_sensitivity_output_dir={Path(tfidf_artifacts['output_dir']).resolve()}",
                f"bm25_sensitivity_output_dir={Path(bm25_artifacts['output_dir']).resolve()}",
                f"depth_analysis_output_dir={Path(depth_artifacts['output_dir']).resolve()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Results CSV: {results_csv}")
    print(f"Model Comparison Dir: {model_artifacts['output_dir']}")
    print(f"TF-IDF Sensitivity Dir: {tfidf_artifacts['output_dir']}")
    print(f"BM25 Sensitivity Dir: {bm25_artifacts['output_dir']}")
    print(f"Depth Analysis Dir: {depth_artifacts['output_dir']}")
    print(f"Full Run Manifest: {manifest_path}")
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
        if args.command == "tfidf-sensitivity":
            return _run_tfidf_sensitivity_cli(args)
        if args.command == "bm25-sensitivity":
            return _run_bm25_sensitivity_cli(args)
        if args.command == "depth-analysis":
            return _run_depth_analysis_cli(args)
        if args.command == "full-run":
            return _run_full_run_cli(args)
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
