"""CLI entry point for running experiments and comparison analysis."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.experiments.experiment_runner import run_experiments
from src.logging_utils import setup_logging
from src.run_metadata import get_git_short_sha

logger = logging.getLogger("src.main")
_ADVANCED_HELP = argparse.SUPPRESS

if TYPE_CHECKING:
    from src.experiments.heldout_class_runner import HeldoutClassResultRecord
    from src.experiments.heldout_kg_runner import HeldoutKgResultRecord


def _get_git_short_sha() -> str:
    """Backward-compatible wrapper around shared git metadata helper."""
    return get_git_short_sha(repo_root=Path(__file__).resolve().parents[1])


def generate_heldout_selection(
    *, results_csv_path: str | Path | None, config_path: str | Path, output_dir: str | Path
) -> dict[str, Path]:
    from src.analysis.heldout_selection import generate_heldout_selection as _impl

    return _impl(results_csv_path=results_csv_path, config_path=config_path, output_dir=output_dir)


def generate_kg_heldout_reporting(
    *,
    results_csv_path: str | Path | None,
    output_dir: str | Path,
    selected_settings_path: str | Path | None,
    query_level_csv_path: str | Path | None = None,
) -> dict[str, Path]:
    from src.analysis.kg_heldout_reporting import generate_kg_heldout_reporting as _impl

    return _impl(
        results_csv_path=results_csv_path,
        output_dir=output_dir,
        selected_settings_path=selected_settings_path,
        query_level_csv_path=query_level_csv_path,
    )


def generate_class_heldout_reporting(
    *,
    results_csv_path: str | Path | None,
    output_dir: str | Path,
    selected_settings_path: str | Path | None,
) -> dict[str, Path]:
    from src.analysis.class_heldout_reporting import (
        generate_class_heldout_reporting as _impl,
    )

    return _impl(
        results_csv_path=results_csv_path,
        output_dir=output_dir,
        selected_settings_path=selected_settings_path,
    )


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


def run_heldout_kg_experiments(
    *,
    config_path: str | Path,
    selected_settings_path: str | Path | None,
    output_csv_path: str | Path | None,
    show_progress: bool | None,
) -> list[HeldoutKgResultRecord]:
    from src.experiments.heldout_kg_runner import (
        run_heldout_kg_experiments as _impl,
    )

    return _impl(
        config_path=config_path,
        selected_settings_path=selected_settings_path,
        output_csv_path=output_csv_path,
        show_progress=show_progress,
    )


def run_heldout_class_experiments(
    *,
    config_path: str | Path,
    selected_settings_path: str | Path | None,
    output_csv_path: str | Path | None,
    show_progress: bool | None,
) -> list[HeldoutClassResultRecord]:
    from src.experiments.heldout_class_runner import (
        run_heldout_class_experiments as _impl,
    )

    return _impl(
        config_path=config_path,
        selected_settings_path=selected_settings_path,
        output_csv_path=output_csv_path,
        show_progress=show_progress,
    )


def _build_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-path",
        default="config/runtime.yaml",
        help="Path to runtime YAML config (default: config/runtime.yaml).",
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
        help=_ADVANCED_HELP,
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
        default="config/runtime.yaml",
        help="Path to runtime YAML config (default: config/runtime.yaml).",
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
        default="config/runtime.yaml",
        help="Path to runtime YAML config (default: config/runtime.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/comparisons",
        help="Directory where BM25 sensitivity artifacts are written (default: results/comparisons).",
    )


def _build_heldout_selection_parser(parser: argparse.ArgumentParser) -> None:
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
        default="config/runtime.yaml",
        help="Path to runtime YAML config (default: config/runtime.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/comparisons",
        help="Directory where held-out selection artifacts are written (default: results/comparisons).",
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


def _build_report_heldout_kg_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-csv",
        default=None,
        help=(
            "Path to held-out KG results CSV. If omitted, the latest results/heldout_result_*.csv "
            "is used."
        ),
    )
    parser.add_argument(
        "--selected-settings-json",
        default=None,
        help=_ADVANCED_HELP,
    )
    parser.add_argument(
        "--output-dir",
        default="results/comparisons",
        help="Directory where held-out reporting artifacts are written (default: results/comparisons).",
    )
    parser.add_argument(
        "--query-level-csv-path",
        default=None,
        help=_ADVANCED_HELP,
    )


def _build_report_heldout_class_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--results-csv",
        default=None,
        help=(
            "Path to held-out class-only results CSV. If omitted, the latest "
            "results/heldout_class_result_*.csv is used."
        ),
    )
    parser.add_argument(
        "--selected-settings-json",
        default=None,
        help=_ADVANCED_HELP,
    )
    parser.add_argument(
        "--output-dir",
        default="results/comparisons",
        help="Directory where held-out class-only reporting artifacts are written (default: results/comparisons).",
    )


def _build_run_heldout_kg_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-path",
        default="config/runtime.yaml",
        help="Path to runtime YAML config (default: config/runtime.yaml).",
    )
    parser.add_argument(
        "--selected-settings-json",
        default=None,
        help=_ADVANCED_HELP,
    )
    parser.add_argument(
        "--output-csv-path",
        default=None,
        help=(
            "Output CSV path. If omitted, a heldout run-stamped file is created in results/ "
            "(heldout_result_YYYYMMDD_HHMMSS_<gitsha>.csv)."
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
        help=_ADVANCED_HELP,
    )


def _build_run_heldout_class_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-path",
        default="config/runtime.yaml",
        help="Path to runtime YAML config (default: config/runtime.yaml).",
    )
    parser.add_argument(
        "--selected-settings-json",
        default=None,
        help=_ADVANCED_HELP,
    )
    parser.add_argument(
        "--output-csv-path",
        default=None,
        help=(
            "Output CSV path. If omitted, a heldout class-only run-stamped file is created in results/ "
            "(heldout_class_result_YYYYMMDD_HHMMSS_<gitsha>.csv)."
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
        help=_ADVANCED_HELP,
    )


def _build_heldout_full_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-path",
        default="config/runtime.yaml",
        help="Path to runtime YAML config (default: config/runtime.yaml).",
    )
    parser.add_argument(
        "--dev-results-csv",
        default=None,
        help=(
            "Path to development results CSV used for held-out selection. If omitted, the latest "
            "results/result_*.csv is used when selection runs."
        ),
    )
    parser.add_argument(
        "--selected-settings-json",
        default=None,
        help=_ADVANCED_HELP,
    )
    parser.add_argument(
        "--heldout-results-csv",
        default=None,
        help=_ADVANCED_HELP,
    )
    parser.add_argument(
        "--output-dir",
        default="results/comparisons",
        help="Directory where held-out report artifacts are written (default: results/comparisons).",
    )
    parser.add_argument(
        "--output-csv-path",
        default=None,
        help=_ADVANCED_HELP,
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        action="store_true",
        help="Force-enable progress bars for the held-out KG run stage.",
    )
    progress_group.add_argument(
        "--no-progress",
        action="store_true",
        help=_ADVANCED_HELP,
    )


def _build_heldout_class_full_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-path",
        default="config/runtime.yaml",
        help="Path to runtime YAML config (default: config/runtime.yaml).",
    )
    parser.add_argument(
        "--dev-results-csv",
        default=None,
        help=(
            "Path to development results CSV used for held-out selection. If omitted, the latest "
            "results/result_*.csv is used when selection runs."
        ),
    )
    parser.add_argument(
        "--selected-settings-json",
        default=None,
        help=_ADVANCED_HELP,
    )
    parser.add_argument(
        "--heldout-results-csv",
        default=None,
        help=_ADVANCED_HELP,
    )
    parser.add_argument(
        "--output-dir",
        default="results/comparisons",
        help="Directory where held-out class-only report artifacts are written (default: results/comparisons).",
    )
    parser.add_argument(
        "--output-csv-path",
        default=None,
        help=_ADVANCED_HELP,
    )
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        action="store_true",
        help="Force-enable progress bars for the held-out class-only run stage.",
    )
    progress_group.add_argument(
        "--no-progress",
        action="store_true",
        help=_ADVANCED_HELP,
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

    heldout_selection_parser = subparsers.add_parser(
        "select-heldout-settings",
        help="Freeze held-out TF-IDF and BM25 settings from development results.",
    )
    _build_heldout_selection_parser(heldout_selection_parser)

    depth_analysis_parser = subparsers.add_parser(
        "depth-analysis",
        help="Generate retrieval-depth behavior artifacts from results CSV.",
    )
    _build_depth_analysis_parser(depth_analysis_parser)

    report_heldout_kg_parser = subparsers.add_parser(
        "report-heldout-kg",
        help="Generate held-out KG reporting artifacts by entity type.",
    )
    _build_report_heldout_kg_parser(report_heldout_kg_parser)

    report_heldout_class_parser = subparsers.add_parser(
        "report-heldout-class",
        help="Generate held-out class-only reporting artifacts (secondary benchmark).",
    )
    _build_report_heldout_class_parser(report_heldout_class_parser)

    heldout_run_parser = subparsers.add_parser(
        "run-heldout-kg",
        help="Run frozen held-out KG evaluation by entity type.",
    )
    _build_run_heldout_kg_parser(heldout_run_parser)

    heldout_class_run_parser = subparsers.add_parser(
        "run-heldout-class",
        help="Run frozen held-out class-only evaluation.",
    )
    _build_run_heldout_class_parser(heldout_class_run_parser)

    heldout_full_run_parser = subparsers.add_parser(
        "heldout-full-run",
        help="Run the full held-out KG workflow: selection, held-out execution, and reporting.",
    )
    _build_heldout_full_run_parser(heldout_full_run_parser)

    heldout_class_full_run_parser = subparsers.add_parser(
        "heldout-class-full-run",
        help="Run the full secondary class-only held-out workflow: selection, execution, and reporting.",
    )
    _build_heldout_class_full_run_parser(heldout_class_full_run_parser)

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


def _collect_existing_heldout_result_files(
    results_dir: Path = Path("results"),
) -> set[Path]:
    if not results_dir.exists():
        return set()
    return {
        path.resolve() for path in results_dir.glob("heldout_result_*.csv")
    }


def _resolve_new_heldout_result_file(
    existing_files: set[Path], results_dir: Path = Path("results")
) -> Path:
    current_files = {
        path.resolve() for path in results_dir.glob("heldout_result_*.csv")
    }
    new_files = sorted(current_files - existing_files, key=lambda p: p.stat().st_mtime)
    if not new_files:
        raise FileNotFoundError(
            "No new heldout_result_*.csv file was created in 'results/' for this run."
        )
    return new_files[-1]


def _collect_existing_heldout_class_result_files(
    results_dir: Path = Path("results"),
) -> set[Path]:
    if not results_dir.exists():
        return set()
    return {
        path.resolve() for path in results_dir.glob("heldout_class_result_*.csv")
    }


def _resolve_new_heldout_class_result_file(
    existing_files: set[Path], results_dir: Path = Path("results")
) -> Path:
    current_files = {
        path.resolve() for path in results_dir.glob("heldout_class_result_*.csv")
    }
    new_files = sorted(current_files - existing_files, key=lambda p: p.stat().st_mtime)
    if not new_files:
        raise FileNotFoundError(
            "No new heldout_class_result_*.csv file was created in 'results/' for this run."
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


def _run_heldout_selection_cli(args: argparse.Namespace) -> int:
    artifacts = generate_heldout_selection(
        results_csv_path=args.results_csv,
        config_path=args.config_path,
        output_dir=args.output_dir,
    )
    print(f"Held-Out Selection Dir: {artifacts['output_dir']}")
    return 0


def _run_depth_analysis_cli(args: argparse.Namespace) -> int:
    artifacts = generate_depth_analysis(
        results_csv_path=args.results_csv,
        output_dir=args.output_dir,
    )
    print(f"Depth Analysis Dir: {artifacts['output_dir']}")
    return 0


def _run_report_heldout_kg_cli(args: argparse.Namespace) -> int:
    artifacts = generate_kg_heldout_reporting(
        results_csv_path=args.results_csv,
        output_dir=args.output_dir,
        selected_settings_path=args.selected_settings_json,
        query_level_csv_path=args.query_level_csv_path,
    )
    print(f"Held-Out KG Report Dir: {artifacts['output_dir']}")
    return 0


def _run_report_heldout_class_cli(args: argparse.Namespace) -> int:
    artifacts = generate_class_heldout_reporting(
        results_csv_path=args.results_csv,
        output_dir=args.output_dir,
        selected_settings_path=args.selected_settings_json,
    )
    print(f"Held-Out Class-Only Report Dir: {artifacts['output_dir']}")
    return 0


def _run_heldout_kg_cli(args: argparse.Namespace) -> int:
    show_progress = _resolve_show_progress(args)
    existing_files = (
        _collect_existing_heldout_result_files()
        if args.output_csv_path is None
        else set()
    )
    run_heldout_kg_experiments(
        config_path=args.config_path,
        selected_settings_path=args.selected_settings_json,
        output_csv_path=args.output_csv_path,
        show_progress=show_progress,
    )
    final_output_path = (
        Path(args.output_csv_path)
        if args.output_csv_path
        else _resolve_new_heldout_result_file(existing_files)
    )
    if not final_output_path.exists():
        raise FileNotFoundError(
            f"Expected held-out output CSV was not created: {final_output_path}"
        )
    query_csv_path = _infer_query_level_csv_path(final_output_path)
    print(f"Held-Out KG Results CSV: {final_output_path}")
    print(f"Held-Out KG Query CSV: {query_csv_path}")
    return 0


def _infer_query_level_csv_path(results_csv_path: Path) -> Path:
    stem = results_csv_path.stem
    if stem.startswith("heldout_result_"):
        suffix = stem.removeprefix("heldout_result_")
        query_name = f"heldout_query_result_{suffix}.csv"
    else:
        query_name = f"{stem}_query.csv"
    return results_csv_path.with_name(query_name)


def _run_heldout_class_cli(args: argparse.Namespace) -> int:
    show_progress = _resolve_show_progress(args)
    existing_files = (
        _collect_existing_heldout_class_result_files()
        if args.output_csv_path is None
        else set()
    )
    run_heldout_class_experiments(
        config_path=args.config_path,
        selected_settings_path=args.selected_settings_json,
        output_csv_path=args.output_csv_path,
        show_progress=show_progress,
    )
    final_output_path = (
        Path(args.output_csv_path)
        if args.output_csv_path
        else _resolve_new_heldout_class_result_file(existing_files)
    )
    if not final_output_path.exists():
        raise FileNotFoundError(
            f"Expected held-out class-only output CSV was not created: {final_output_path}"
        )
    print(f"Held-Out Class-Only Results CSV: {final_output_path}")
    return 0


def _resolve_existing_file(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def _load_selected_settings_source_csv(selected_settings_path: Path) -> Path | None:
    try:
        payload = json.loads(selected_settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    source_csv = payload.get("source_csv")
    if not isinstance(source_csv, str) or not source_csv:
        return None
    return Path(source_csv).resolve()


def _resolve_heldout_manifest_dir(
    *,
    output_dir: str | Path,
    report_output_dir: Path | None,
    heldout_results_csv: Path | None,
    heldout_results_csv_arg: str | Path | None,
    output_csv_path_arg: str | Path | None,
) -> Path:
    if report_output_dir is not None:
        return report_output_dir.resolve()
    if heldout_results_csv is not None:
        return Path(output_dir).resolve() / heldout_results_csv.stem
    if heldout_results_csv_arg is not None:
        return Path(output_dir).resolve() / Path(heldout_results_csv_arg).stem
    if output_csv_path_arg is not None:
        return Path(output_dir).resolve() / Path(output_csv_path_arg).stem
    return Path(output_dir).resolve() / "heldout_full_run_partial"


def _resolve_heldout_class_manifest_dir(
    *,
    output_dir: str | Path,
    report_output_dir: Path | None,
    heldout_results_csv: Path | None,
    heldout_results_csv_arg: str | Path | None,
    output_csv_path_arg: str | Path | None,
) -> Path:
    if report_output_dir is not None:
        return report_output_dir.resolve()
    if heldout_results_csv is not None:
        return Path(output_dir).resolve() / heldout_results_csv.stem
    if heldout_results_csv_arg is not None:
        return Path(output_dir).resolve() / Path(heldout_results_csv_arg).stem
    if output_csv_path_arg is not None:
        return Path(output_dir).resolve() / Path(output_csv_path_arg).stem
    return Path(output_dir).resolve() / "heldout_class_full_run_partial"


def _run_heldout_full_run_cli(args: argparse.Namespace) -> int:
    started_at = datetime.now()
    start_perf = time.perf_counter()
    show_progress = _resolve_show_progress(args)
    stage_status: dict[str, str] = {
        "select_heldout_settings": "not_run",
        "run_heldout_kg": "not_run",
        "report_heldout_kg": "not_run",
    }

    selected_settings_path: Path
    heldout_results_csv: Path
    development_results_csv: Path | None = None
    report_output_dir: Path | None = None
    caught_exc: Exception | None = None

    try:
        if args.selected_settings_json:
            selected_settings_path = _resolve_existing_file(
                args.selected_settings_json,
                label="Selected settings JSON",
            )
            development_results_csv = _load_selected_settings_source_csv(selected_settings_path)
            stage_status["select_heldout_settings"] = "skipped"
        else:
            logger.info(
                "Starting heldout-full-run stage=%s results_csv=%s config_path=%s output_dir=%s",
                "select_heldout_settings",
                args.dev_results_csv,
                args.config_path,
                args.output_dir,
            )
            selection_artifacts = generate_heldout_selection(
                results_csv_path=args.dev_results_csv,
                config_path=args.config_path,
                output_dir=args.output_dir,
            )
            selected_settings_path = Path(selection_artifacts["heldout_selected_settings"]).resolve()
            development_results_csv = Path(selection_artifacts["source_csv"]).resolve()
            stage_status["select_heldout_settings"] = "success"
            logger.info("Completed heldout-full-run stage=%s", "select_heldout_settings")

        if args.heldout_results_csv:
            heldout_results_csv = _resolve_existing_file(
                args.heldout_results_csv,
                label="Held-out results CSV",
            )
            stage_status["run_heldout_kg"] = "skipped"
        else:
            logger.info(
                "Starting heldout-full-run stage=%s config_path=%s selected_settings=%s output_csv_path=%s progress=%s",
                "run_heldout_kg",
                args.config_path,
                selected_settings_path,
                args.output_csv_path,
                show_progress,
            )
            existing_files = (
                _collect_existing_heldout_result_files()
                if args.output_csv_path is None
                else set()
            )
            run_heldout_kg_experiments(
                config_path=args.config_path,
                selected_settings_path=selected_settings_path,
                output_csv_path=args.output_csv_path,
                show_progress=show_progress,
            )
            heldout_results_csv = (
                Path(args.output_csv_path).resolve()
                if args.output_csv_path
                else _resolve_new_heldout_result_file(existing_files)
            )
            if not heldout_results_csv.exists():
                raise FileNotFoundError(
                    f"Expected held-out output CSV was not created: {heldout_results_csv}"
                )
            stage_status["run_heldout_kg"] = "success"
            logger.info("Completed heldout-full-run stage=%s", "run_heldout_kg")

        logger.info(
            "Starting heldout-full-run stage=%s heldout_results_csv=%s selected_settings=%s output_dir=%s",
            "report_heldout_kg",
            heldout_results_csv,
            selected_settings_path,
            args.output_dir,
        )
        report_artifacts = generate_kg_heldout_reporting(
            results_csv_path=heldout_results_csv,
            output_dir=args.output_dir,
            selected_settings_path=selected_settings_path,
            query_level_csv_path=_infer_query_level_csv_path(heldout_results_csv),
        )
        report_output_dir = Path(report_artifacts["output_dir"]).resolve()
        stage_status["report_heldout_kg"] = "success"
        logger.info("Completed heldout-full-run stage=%s", "report_heldout_kg")
    except Exception as exc:
        caught_exc = exc
        failed_stage = next(
            (
                stage
                for stage in ("select_heldout_settings", "run_heldout_kg", "report_heldout_kg")
                if stage_status[stage] == "not_run"
            ),
            "unknown",
        )
        if failed_stage != "unknown":
            stage_status[failed_stage] = "failed"
        logger.exception(
            "Heldout-full-run stage failed stage=%s dev_results_csv=%s config_path=%s selected_settings_json=%s heldout_results_csv=%s output_dir=%s error=%s: %s",
            failed_stage,
            args.dev_results_csv,
            args.config_path,
            args.selected_settings_json,
            args.heldout_results_csv,
            args.output_dir,
            type(exc).__name__,
            exc,
        )
    ended_at = datetime.now()
    elapsed_seconds = time.perf_counter() - start_perf
    manifest_dir = _resolve_heldout_manifest_dir(
        output_dir=args.output_dir,
        report_output_dir=report_output_dir,
        heldout_results_csv=heldout_results_csv if "heldout_results_csv" in locals() else None,
        heldout_results_csv_arg=args.heldout_results_csv,
        output_csv_path_arg=args.output_csv_path,
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "heldout_full_run_manifest.txt"
    manifest_path.write_text(
        "\n".join(
            [
                f"generated_at_start={started_at.isoformat(timespec='seconds')}",
                f"generated_at_end={ended_at.isoformat(timespec='seconds')}",
                f"elapsed_seconds={elapsed_seconds:.6f}",
                f"git_sha={_get_git_short_sha()}",
                f"config_path={Path(args.config_path).resolve()}",
                f"development_results_csv={development_results_csv if development_results_csv is not None else ''}",
                f"selected_settings_json={selected_settings_path.resolve() if 'selected_settings_path' in locals() else ''}",
                f"heldout_results_csv={heldout_results_csv.resolve() if 'heldout_results_csv' in locals() else ''}",
                f"report_output_dir={report_output_dir if report_output_dir is not None else ''}",
                f"stage_select_heldout_settings={stage_status['select_heldout_settings']}",
                f"stage_run_heldout_kg={stage_status['run_heldout_kg']}",
                f"stage_report_heldout_kg={stage_status['report_heldout_kg']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if caught_exc is not None:
        raise caught_exc

    print(f"Selected Settings JSON: {selected_settings_path}")
    print(f"Held-Out KG Results CSV: {heldout_results_csv}")
    print(f"Held-Out KG Report Dir: {report_output_dir}")
    print(f"Held-Out Full Run Manifest: {manifest_path}")
    return 0


def _run_heldout_class_full_run_cli(args: argparse.Namespace) -> int:
    started_at = datetime.now()
    start_perf = time.perf_counter()
    show_progress = _resolve_show_progress(args)
    stage_status: dict[str, str] = {
        "select_heldout_settings": "not_run",
        "run_heldout_class": "not_run",
        "report_heldout_class": "not_run",
    }

    selected_settings_path: Path
    heldout_results_csv: Path
    development_results_csv: Path | None = None
    report_output_dir: Path | None = None
    caught_exc: Exception | None = None

    try:
        if args.selected_settings_json:
            selected_settings_path = _resolve_existing_file(
                args.selected_settings_json,
                label="Selected settings JSON",
            )
            development_results_csv = _load_selected_settings_source_csv(selected_settings_path)
            stage_status["select_heldout_settings"] = "skipped"
        else:
            selection_artifacts = generate_heldout_selection(
                results_csv_path=args.dev_results_csv,
                config_path=args.config_path,
                output_dir=args.output_dir,
            )
            selected_settings_path = Path(selection_artifacts["heldout_selected_settings"]).resolve()
            development_results_csv = Path(selection_artifacts["source_csv"]).resolve()
            stage_status["select_heldout_settings"] = "success"

        if args.heldout_results_csv:
            heldout_results_csv = _resolve_existing_file(
                args.heldout_results_csv,
                label="Held-out class-only results CSV",
            )
            stage_status["run_heldout_class"] = "skipped"
        else:
            existing_files = (
                _collect_existing_heldout_class_result_files()
                if args.output_csv_path is None
                else set()
            )
            run_heldout_class_experiments(
                config_path=args.config_path,
                selected_settings_path=selected_settings_path,
                output_csv_path=args.output_csv_path,
                show_progress=show_progress,
            )
            heldout_results_csv = (
                Path(args.output_csv_path).resolve()
                if args.output_csv_path
                else _resolve_new_heldout_class_result_file(existing_files)
            )
            if not heldout_results_csv.exists():
                raise FileNotFoundError(
                    f"Expected held-out class-only output CSV was not created: {heldout_results_csv}"
                )
            stage_status["run_heldout_class"] = "success"

        report_artifacts = generate_class_heldout_reporting(
            results_csv_path=heldout_results_csv,
            output_dir=args.output_dir,
            selected_settings_path=selected_settings_path,
        )
        report_output_dir = Path(report_artifacts["output_dir"]).resolve()
        stage_status["report_heldout_class"] = "success"
    except Exception as exc:
        caught_exc = exc
        failed_stage = next(
            (
                stage
                for stage in (
                    "select_heldout_settings",
                    "run_heldout_class",
                    "report_heldout_class",
                )
                if stage_status[stage] == "not_run"
            ),
            "unknown",
        )
        if failed_stage != "unknown":
            stage_status[failed_stage] = "failed"
        logger.exception(
            "Heldout-class-full-run stage failed stage=%s dev_results_csv=%s config_path=%s selected_settings_json=%s heldout_results_csv=%s output_dir=%s error=%s: %s",
            failed_stage,
            args.dev_results_csv,
            args.config_path,
            args.selected_settings_json,
            args.heldout_results_csv,
            args.output_dir,
            type(exc).__name__,
            exc,
        )
    ended_at = datetime.now()
    elapsed_seconds = time.perf_counter() - start_perf
    manifest_dir = _resolve_heldout_class_manifest_dir(
        output_dir=args.output_dir,
        report_output_dir=report_output_dir,
        heldout_results_csv=heldout_results_csv if "heldout_results_csv" in locals() else None,
        heldout_results_csv_arg=args.heldout_results_csv,
        output_csv_path_arg=args.output_csv_path,
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "heldout_class_full_run_manifest.txt"
    manifest_path.write_text(
        "\n".join(
            [
                f"generated_at_start={started_at.isoformat(timespec='seconds')}",
                f"generated_at_end={ended_at.isoformat(timespec='seconds')}",
                f"elapsed_seconds={elapsed_seconds:.6f}",
                f"git_sha={_get_git_short_sha()}",
                f"config_path={Path(args.config_path).resolve()}",
                f"development_results_csv={development_results_csv if development_results_csv is not None else ''}",
                f"selected_settings_json={selected_settings_path.resolve() if 'selected_settings_path' in locals() else ''}",
                f"heldout_results_csv={heldout_results_csv.resolve() if 'heldout_results_csv' in locals() else ''}",
                f"report_output_dir={report_output_dir if report_output_dir is not None else ''}",
                f"stage_select_heldout_settings={stage_status['select_heldout_settings']}",
                f"stage_run_heldout_class={stage_status['run_heldout_class']}",
                f"stage_report_heldout_class={stage_status['report_heldout_class']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if caught_exc is not None:
        raise caught_exc

    print(f"Selected Settings JSON: {selected_settings_path}")
    print(f"Held-Out Class-Only Results CSV: {heldout_results_csv}")
    print(f"Held-Out Class-Only Report Dir: {report_output_dir}")
    print(f"Held-Out Class-Only Full Run Manifest: {manifest_path}")
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
        if args.command == "select-heldout-settings":
            return _run_heldout_selection_cli(args)
        if args.command == "depth-analysis":
            return _run_depth_analysis_cli(args)
        if args.command == "report-heldout-kg":
            return _run_report_heldout_kg_cli(args)
        if args.command == "report-heldout-class":
            return _run_report_heldout_class_cli(args)
        if args.command == "run-heldout-kg":
            return _run_heldout_kg_cli(args)
        if args.command == "run-heldout-class":
            return _run_heldout_class_cli(args)
        if args.command == "heldout-full-run":
            return _run_heldout_full_run_cli(args)
        if args.command == "heldout-class-full-run":
            return _run_heldout_class_full_run_cli(args)
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
