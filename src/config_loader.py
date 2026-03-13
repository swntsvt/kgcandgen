"""Dataset configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetConfig:
    """Structured dataset configuration."""

    name: str
    track: str
    version: str
    source_rdf: Path
    target_rdf: Path
    alignment_rdf: Path


@dataclass(frozen=True)
class TfidfGridEntry:
    """Validated TF-IDF hyperparameter entry."""

    ngram_range: tuple[int, int]
    min_df: int | float
    max_df: int | float
    sublinear_tf: bool


@dataclass(frozen=True)
class Bm25GridEntry:
    """Validated BM25 hyperparameter entry."""

    k1: float
    b: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment runtime configuration."""

    evaluation_ks: list[int]
    tfidf_grid: list[TfidfGridEntry]
    bm25_grid: list[Bm25GridEntry]


@dataclass(frozen=True)
class RuntimeConfig:
    """Top-level runtime configuration with datasets and experiments."""

    datasets: dict[str, DatasetConfig]
    experiments: ExperimentConfig


def _ensure_mapping(value: Any, field_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_path} must be a mapping.")
    return value


def _ensure_number(value: Any, field_path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_path} must be a number.")
    return value


def _validate_df_threshold(value: Any, field_path: str) -> int | float:
    numeric = _ensure_number(value, field_path)
    if isinstance(numeric, int):
        if numeric < 1:
            raise ValueError(f"{field_path} must be >= 1 when provided as an integer.")
        return numeric

    threshold = float(numeric)
    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError(
            f"{field_path} must be in (0.0, 1.0] when provided as a float."
        )
    return threshold


def _load_config_content(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.error("Config file not found: %s", path)
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        logger.exception("Invalid YAML in config file: %s", path)
        raise ValueError(f"Invalid YAML in config file: {path}") from exc

    if not isinstance(content, dict):
        logger.error("Config must be a YAML mapping at the top level: %s", path)
        raise ValueError("Config must be a YAML mapping at the top level.")
    return content


def _validate_datasets(raw_datasets: Any, path: Path) -> dict[str, DatasetConfig]:
    datasets = _ensure_mapping(raw_datasets, "datasets")
    logger.info("Discovered %d dataset entry(ies) in %s", len(datasets), path)

    loaded: dict[str, DatasetConfig] = {}
    required_fields = ("track", "version", "source_rdf", "target_rdf", "alignment_rdf")

    for dataset_name, raw_config in datasets.items():
        if not isinstance(raw_config, dict):
            logger.error("Dataset '%s' must be a mapping", dataset_name)
            raise ValueError(f"Dataset '{dataset_name}' must be a mapping.")

        missing = [field for field in required_fields if field not in raw_config]
        if missing:
            missing_str = ", ".join(missing)
            logger.error(
                "Dataset '%s' missing required fields: %s", dataset_name, missing_str
            )
            raise ValueError(f"Dataset '{dataset_name}' missing required fields: {missing_str}")

        source_rdf = Path(raw_config["source_rdf"])
        target_rdf = Path(raw_config["target_rdf"])
        alignment_rdf = Path(raw_config["alignment_rdf"])

        for label, rdf_path in (
            ("source_rdf", source_rdf),
            ("target_rdf", target_rdf),
            ("alignment_rdf", alignment_rdf),
        ):
            if not rdf_path.exists():
                logger.error(
                    "Dataset '%s' has missing file for '%s': %s",
                    dataset_name,
                    label,
                    rdf_path,
                )
                raise FileNotFoundError(
                    f"Dataset '{dataset_name}' has missing file for '{label}': {rdf_path}"
                )

        loaded[dataset_name] = DatasetConfig(
            name=dataset_name,
            track=str(raw_config["track"]),
            version=str(raw_config["version"]),
            source_rdf=source_rdf,
            target_rdf=target_rdf,
            alignment_rdf=alignment_rdf,
        )
        logger.info(
            "Validated dataset '%s' (track=%s, version=%s)",
            dataset_name,
            loaded[dataset_name].track,
            loaded[dataset_name].version,
        )
    return loaded


def _validate_experiment_config(raw_experiments: Any) -> ExperimentConfig:
    experiments = _ensure_mapping(raw_experiments, "experiments")

    raw_ks = experiments.get("evaluation_ks")
    if not isinstance(raw_ks, list) or not raw_ks:
        raise ValueError("experiments.evaluation_ks must be a non-empty list.")

    evaluation_ks: list[int] = []
    seen_ks: set[int] = set()
    for index, raw_k in enumerate(raw_ks):
        field = f"experiments.evaluation_ks[{index}]"
        if isinstance(raw_k, bool) or not isinstance(raw_k, int):
            raise ValueError(f"{field} must be a positive integer.")
        if raw_k <= 0:
            raise ValueError(f"{field} must be a positive integer.")
        if raw_k in seen_ks:
            raise ValueError(f"{field} duplicates k={raw_k}; values must be unique.")
        evaluation_ks.append(raw_k)
        seen_ks.add(raw_k)

    raw_tfidf_grid = experiments.get("tfidf_grid")
    if not isinstance(raw_tfidf_grid, list) or not raw_tfidf_grid:
        raise ValueError("experiments.tfidf_grid must be a non-empty list.")
    tfidf_grid: list[TfidfGridEntry] = []
    for index, raw_entry in enumerate(raw_tfidf_grid):
        field = f"experiments.tfidf_grid[{index}]"
        entry = _ensure_mapping(raw_entry, field)
        required = {"ngram_range", "min_df", "max_df", "sublinear_tf"}
        missing = sorted(required - set(entry.keys()))
        if missing:
            raise ValueError(f"{field} missing required keys: {', '.join(missing)}")

        raw_ngram = entry["ngram_range"]
        ngram_field = f"{field}.ngram_range"
        if (
            not isinstance(raw_ngram, (list, tuple))
            or len(raw_ngram) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in raw_ngram)
        ):
            raise ValueError(f"{ngram_field} must be [min_n, max_n] with two integers.")
        ngram_min = int(raw_ngram[0])
        ngram_max = int(raw_ngram[1])
        if ngram_min <= 0 or ngram_max <= 0:
            raise ValueError(f"{ngram_field} values must be positive integers.")
        if ngram_min > ngram_max:
            raise ValueError(f"{ngram_field} must satisfy min_n <= max_n.")

        min_df = _validate_df_threshold(entry["min_df"], f"{field}.min_df")
        max_df = _validate_df_threshold(entry["max_df"], f"{field}.max_df")
        if isinstance(min_df, int) and isinstance(max_df, int) and min_df > max_df:
            raise ValueError(
                f"{field} must satisfy min_df <= max_df when both are integers."
            )
        if isinstance(min_df, float) and isinstance(max_df, float) and min_df > max_df:
            raise ValueError(
                f"{field} must satisfy min_df <= max_df when both are floats."
            )

        sublinear_tf = entry["sublinear_tf"]
        if not isinstance(sublinear_tf, bool):
            raise ValueError(f"{field}.sublinear_tf must be a boolean.")

        tfidf_grid.append(
            TfidfGridEntry(
                ngram_range=(ngram_min, ngram_max),
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=sublinear_tf,
            )
        )

    raw_bm25_grid = experiments.get("bm25_grid")
    if not isinstance(raw_bm25_grid, list) or not raw_bm25_grid:
        raise ValueError("experiments.bm25_grid must be a non-empty list.")
    bm25_grid: list[Bm25GridEntry] = []
    for index, raw_entry in enumerate(raw_bm25_grid):
        field = f"experiments.bm25_grid[{index}]"
        entry = _ensure_mapping(raw_entry, field)
        required = {"k1", "b"}
        missing = sorted(required - set(entry.keys()))
        if missing:
            raise ValueError(f"{field} missing required keys: {', '.join(missing)}")

        k1 = float(_ensure_number(entry["k1"], f"{field}.k1"))
        b = float(_ensure_number(entry["b"], f"{field}.b"))
        if k1 < 0:
            raise ValueError(f"{field}.k1 must be greater than or equal to 0.")
        if b < 0 or b > 1:
            raise ValueError(f"{field}.b must be between 0 and 1.")

        bm25_grid.append(Bm25GridEntry(k1=k1, b=b))

    logger.info(
        "Validated experiment config (evaluation_ks=%s, tfidf_grid=%d, bm25_grid=%d)",
        evaluation_ks,
        len(tfidf_grid),
        len(bm25_grid),
    )
    return ExperimentConfig(
        evaluation_ks=evaluation_ks,
        tfidf_grid=tfidf_grid,
        bm25_grid=bm25_grid,
    )


def load_runtime_config(config_path: str | Path = "config/datasets.yaml") -> RuntimeConfig:
    """Load and validate dataset + experiment configuration from YAML."""
    path = Path(config_path)
    logger.info("Loading datasets config from %s", path)
    content = _load_config_content(path)

    if "datasets" not in content:
        logger.error("Config missing 'datasets' mapping: %s", path)
        raise ValueError("Config must contain a 'datasets' mapping.")
    if "experiments" not in content:
        logger.error("Config missing 'experiments' mapping: %s", path)
        raise ValueError("Config must contain an 'experiments' mapping.")

    datasets = _validate_datasets(content["datasets"], path)
    experiments = _validate_experiment_config(content["experiments"])
    logger.info(
        "Loaded %d validated dataset config(s) from %s", len(datasets), path
    )
    return RuntimeConfig(datasets=datasets, experiments=experiments)


def get_dataset_config(name: str, config_path: str | Path = "config/datasets.yaml") -> DatasetConfig:
    """Return one dataset configuration by name."""
    datasets = load_runtime_config(config_path=config_path).datasets
    if name not in datasets:
        raise KeyError(f"Dataset '{name}' not found in config.")
    return datasets[name]
