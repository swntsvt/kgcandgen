"""Runtime configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from src.rdf_utils.alignment_parser import (
    ALIGNMENT_POLICIES,
    DEFAULT_ALIGNMENT_POLICY,
)
from src.retrieval.topk import CANDIDATE_POLICIES, DEFAULT_CANDIDATE_POLICY

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/runtime.yaml"
DATA_ROOT_ENV_VAR = "KGCANDGEN_DATA_ROOT"

REQUIRED_HELDOUT_SELECTION_POLICY = {
    "metric": "mrr",
    "lambda": 0.5,
    "weighting": "equal_track_weight",
    "ranking": "per_track_normalized_rank",
}


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
class SecondaryHeldoutDatasetConfig(DatasetConfig):
    """Secondary held-out dataset configuration."""


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
class CharNgramGridEntry:
    """Validated character n-gram sensitivity entry (not used for selection)."""

    analyzer: str
    ngram_range: tuple[int, int]
    min_df: int
    sublinear_tf: bool


DENSE_TIERS: tuple[str, ...] = ("development", "kg", "class")


@dataclass(frozen=True)
class DenseModelEntry:
    """A pinned sentence-transformer baseline, applied with fixed settings (never tuned)."""

    name: str
    model: str
    revision: str
    tiers: tuple[str, ...]
    device: str = "cpu"
    batch_size: int = 256

    def hyperparameters(self) -> dict[str, object]:
        # Device and batch size are runtime knobs, not part of the method's identity.
        return {
            "model": self.model,
            "revision": self.revision,
            "normalization": "light_v1",
            "similarity": "cosine_exact",
        }


@dataclass(frozen=True)
class HybridModelEntry:
    """Reciprocal-rank fusion of the default char n-gram retriever and one dense model."""

    name: str
    dense: DenseModelEntry
    tiers: tuple[str, ...]
    rrf_k: int = 60
    depth: int = 50

    @property
    def device(self) -> str:
        return self.dense.device

    @property
    def batch_size(self) -> int:
        return self.dense.batch_size

    def hyperparameters(self) -> dict[str, object]:
        return {
            "fusion": "rrf",
            "rrf_k": self.rrf_k,
            "depth": self.depth,
            "lexical": "char_ngram",
            "lexical_analyzer": "char_wb",
            "lexical_ngram_range": [3, 5],
            "model": self.dense.model,
            "revision": self.dense.revision,
            "normalization": "light_v1",
        }


FixedBaselineEntry = DenseModelEntry | HybridModelEntry


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment runtime configuration."""

    evaluation_ks: list[int]
    tfidf_grid: list[TfidfGridEntry]
    bm25_grid: list[Bm25GridEntry]
    candidate_policy: str = DEFAULT_CANDIDATE_POLICY
    alignment_policy: str = DEFAULT_ALIGNMENT_POLICY
    char_ngram_grid: tuple[CharNgramGridEntry, ...] = ()
    dense_models: tuple[DenseModelEntry, ...] = ()
    hybrid_models: tuple[HybridModelEntry, ...] = ()

    def dense_models_for(self, tier: str) -> list[DenseModelEntry]:
        return [entry for entry in self.dense_models if tier in entry.tiers]

    def fixed_baselines_for(self, tier: str) -> list[FixedBaselineEntry]:
        """Dense, then hybrid baselines that run on ``tier`` (pinned, never tuned)."""
        hybrids = [entry for entry in self.hybrid_models if tier in entry.tiers]
        return [*self.dense_models_for(tier), *hybrids]


@dataclass(frozen=True)
class HeldoutSelectionPolicy:
    """Held-out setting selection policy."""

    metric: str
    lambda_penalty: float
    weighting: str
    ranking: str

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "lambda": self.lambda_penalty,
            "weighting": self.weighting,
            "ranking": self.ranking,
        }


@dataclass(frozen=True)
class HeldoutConfig:
    """Top-level held-out workflow configuration."""

    selection: HeldoutSelectionPolicy


@dataclass(frozen=True)
class RuntimeConfig:
    """Top-level runtime configuration with dataset splits and experiments."""

    development_datasets: dict[str, DatasetConfig]
    heldout_datasets: dict[str, DatasetConfig]
    heldout_secondary_datasets: dict[str, SecondaryHeldoutDatasetConfig]
    experiments: ExperimentConfig
    heldout: HeldoutConfig | None

    @property
    def datasets(self) -> dict[str, DatasetConfig]:
        """Backward-compatible alias for development datasets."""
        return self.development_datasets


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


def _resolve_data_root(content: dict[str, Any], config_path: Path) -> Path | None:
    """Return the root for relative dataset paths: env var, else ``data_root``, else None.

    A relative ``data_root`` is resolved against the config file's directory.
    """
    raw = os.environ.get(DATA_ROOT_ENV_VAR) or content.get("data_root")
    if raw is None:
        return None
    root = Path(str(raw)).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data_root does not exist or is not a directory: {root}")
    return root


def _resolve_dataset_path(raw: Any, data_root: Path | None) -> Path:
    path = Path(str(raw)).expanduser()
    if data_root is not None and not path.is_absolute():
        return data_root / path
    return path


def _validate_dataset_group(
    raw_datasets: Any,
    *,
    field_name: str,
    path: Path,
    data_root: Path | None = None,
) -> dict[str, DatasetConfig]:
    datasets = _ensure_mapping(raw_datasets, field_name)
    logger.info("Discovered %d dataset entry(ies) in %s.%s", len(datasets), path, field_name)

    loaded: dict[str, DatasetConfig] = {}
    required_fields = ("track", "version", "source_rdf", "target_rdf", "alignment_rdf")

    for dataset_name, raw_config in datasets.items():
        if not isinstance(raw_config, dict):
            logger.error("%s '%s' must be a mapping", field_name, dataset_name)
            raise ValueError(f"{field_name} '{dataset_name}' must be a mapping.")

        missing = [field for field in required_fields if field not in raw_config]
        if missing:
            missing_str = ", ".join(missing)
            logger.error(
                "%s '%s' missing required fields: %s",
                field_name,
                dataset_name,
                missing_str,
            )
            raise ValueError(f"{field_name} '{dataset_name}' missing required fields: {missing_str}")

        source_rdf = _resolve_dataset_path(raw_config["source_rdf"], data_root)
        target_rdf = _resolve_dataset_path(raw_config["target_rdf"], data_root)
        alignment_rdf = _resolve_dataset_path(raw_config["alignment_rdf"], data_root)

        for label, rdf_path in (
            ("source_rdf", source_rdf),
            ("target_rdf", target_rdf),
            ("alignment_rdf", alignment_rdf),
        ):
            if not rdf_path.exists():
                logger.error(
                    "%s '%s' has missing file for '%s': %s",
                    field_name,
                    dataset_name,
                    label,
                    rdf_path,
                )
                raise FileNotFoundError(
                    f"{field_name} '{dataset_name}' has missing file for '{label}': {rdf_path}"
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
            "Validated %s '%s' (track=%s, version=%s)",
            field_name,
            dataset_name,
            loaded[dataset_name].track,
            loaded[dataset_name].version,
        )
    return loaded


def _validate_secondary_heldout_dataset_group(
    raw_datasets: Any,
    *,
    path: Path,
    data_root: Path | None = None,
) -> dict[str, SecondaryHeldoutDatasetConfig]:
    datasets = _ensure_mapping(raw_datasets, "heldout_secondary_datasets")
    logger.info(
        "Discovered %d dataset entry(ies) in %s.heldout_secondary_datasets",
        len(datasets),
        path,
    )

    loaded: dict[str, SecondaryHeldoutDatasetConfig] = {}
    required_fields = (
        "track",
        "version",
        "source_rdf",
        "target_rdf",
        "alignment_rdf",
    )

    for dataset_name, raw_config in datasets.items():
        if not isinstance(raw_config, dict):
            raise ValueError(
                f"heldout_secondary_datasets '{dataset_name}' must be a mapping."
            )
        missing = [field for field in required_fields if field not in raw_config]
        if missing:
            raise ValueError(
                "heldout_secondary_datasets "
                f"'{dataset_name}' missing required fields: {', '.join(missing)}"
            )

        source_rdf = _resolve_dataset_path(raw_config["source_rdf"], data_root)
        target_rdf = _resolve_dataset_path(raw_config["target_rdf"], data_root)
        alignment_rdf = _resolve_dataset_path(raw_config["alignment_rdf"], data_root)
        for label, rdf_path in (
            ("source_rdf", source_rdf),
            ("target_rdf", target_rdf),
            ("alignment_rdf", alignment_rdf),
        ):
            if not rdf_path.exists():
                raise FileNotFoundError(
                    "heldout_secondary_datasets "
                    f"'{dataset_name}' has missing file for '{label}': {rdf_path}"
                )

        loaded[dataset_name] = SecondaryHeldoutDatasetConfig(
            name=dataset_name,
            track=str(raw_config["track"]),
            version=str(raw_config["version"]),
            source_rdf=source_rdf,
            target_rdf=target_rdf,
            alignment_rdf=alignment_rdf,
        )
        logger.info(
            "Validated heldout_secondary_datasets '%s' (track=%s, version=%s)",
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

    char_ngram_grid: list[CharNgramGridEntry] = []
    raw_char_grid = experiments.get("char_ngram_grid", [])
    if not isinstance(raw_char_grid, list):
        raise ValueError("experiments.char_ngram_grid must be a list when provided.")
    for index, raw_entry in enumerate(raw_char_grid):
        field = f"experiments.char_ngram_grid[{index}]"
        entry = _ensure_mapping(raw_entry, field)
        analyzer = entry.get("analyzer", "char_wb")
        if analyzer not in ("char_wb", "char"):
            raise ValueError(f"{field}.analyzer must be 'char_wb' or 'char'.")
        raw_range = entry.get("ngram_range")
        if (
            not isinstance(raw_range, list)
            or len(raw_range) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in raw_range)
            or raw_range[0] < 1
            or raw_range[0] > raw_range[1]
        ):
            raise ValueError(f"{field}.ngram_range must be [min, max] with 1 <= min <= max.")
        min_df = entry.get("min_df", 1)
        if not isinstance(min_df, int) or isinstance(min_df, bool) or min_df < 1:
            raise ValueError(f"{field}.min_df must be an integer >= 1.")
        sublinear_tf = entry.get("sublinear_tf", False)
        if not isinstance(sublinear_tf, bool):
            raise ValueError(f"{field}.sublinear_tf must be a boolean.")
        char_ngram_grid.append(
            CharNgramGridEntry(
                analyzer=analyzer,
                ngram_range=(raw_range[0], raw_range[1]),
                min_df=min_df,
                sublinear_tf=sublinear_tf,
            )
        )

    dense_models: list[DenseModelEntry] = []
    raw_dense = experiments.get("dense_models", [])
    if not isinstance(raw_dense, list):
        raise ValueError("experiments.dense_models must be a list when provided.")
    for index, raw_entry in enumerate(raw_dense):
        field = f"experiments.dense_models[{index}]"
        entry = _ensure_mapping(raw_entry, field)
        name = entry.get("name")
        if not isinstance(name, str) or not name.startswith("dense_"):
            raise ValueError(f"{field}.name must be a string starting with 'dense_'.")
        if any(existing.name == name for existing in dense_models):
            raise ValueError(f"{field}.name '{name}' is duplicated.")
        for key in ("model", "revision"):
            if not isinstance(entry.get(key), str) or not entry.get(key):
                raise ValueError(f"{field}.{key} must be a non-empty string (pin the revision).")
        tiers = entry.get("tiers", list(DENSE_TIERS))
        if not isinstance(tiers, list) or not tiers or any(t not in DENSE_TIERS for t in tiers):
            raise ValueError(f"{field}.tiers must be a non-empty subset of {list(DENSE_TIERS)}.")
        batch_size = entry.get("batch_size", 256)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"{field}.batch_size must be a positive integer.")
        dense_models.append(
            DenseModelEntry(
                name=name,
                model=entry["model"],
                revision=entry["revision"],
                tiers=tuple(tiers),
                device=str(entry.get("device", "cpu")),
                batch_size=batch_size,
            )
        )

    hybrid_models: list[HybridModelEntry] = []
    raw_hybrid = experiments.get("hybrid_models", [])
    if not isinstance(raw_hybrid, list):
        raise ValueError("experiments.hybrid_models must be a list when provided.")
    dense_by_name = {entry.name: entry for entry in dense_models}
    for index, raw_entry in enumerate(raw_hybrid):
        field = f"experiments.hybrid_models[{index}]"
        entry = _ensure_mapping(raw_entry, field)
        name = entry.get("name")
        if not isinstance(name, str) or not name.startswith("hybrid_"):
            raise ValueError(f"{field}.name must be a string starting with 'hybrid_'.")
        if any(existing.name == name for existing in hybrid_models):
            raise ValueError(f"{field}.name '{name}' is duplicated.")
        dense_entry = dense_by_name.get(entry.get("dense"))  # type: ignore[arg-type]
        if dense_entry is None:
            raise ValueError(f"{field}.dense must name an entry of experiments.dense_models.")
        tiers = entry.get("tiers", list(dense_entry.tiers))
        if not isinstance(tiers, list) or not tiers or any(t not in dense_entry.tiers for t in tiers):
            raise ValueError(
                f"{field}.tiers must be a non-empty subset of the tiers of '{dense_entry.name}'."
            )
        for key, default in (("rrf_k", 60), ("depth", 50)):
            value = entry.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field}.{key} must be a positive integer.")
        hybrid_models.append(
            HybridModelEntry(
                name=name,
                dense=dense_entry,
                tiers=tuple(tiers),
                rrf_k=entry.get("rrf_k", 60),
                depth=entry.get("depth", 50),
            )
        )

    candidate_policy = experiments.get("candidate_policy", DEFAULT_CANDIDATE_POLICY)
    if candidate_policy not in CANDIDATE_POLICIES:
        raise ValueError(
            f"experiments.candidate_policy must be one of {list(CANDIDATE_POLICIES)}."
        )
    alignment_policy = experiments.get("alignment_policy", DEFAULT_ALIGNMENT_POLICY)
    if alignment_policy not in ALIGNMENT_POLICIES:
        raise ValueError(
            f"experiments.alignment_policy must be one of {list(ALIGNMENT_POLICIES)}."
        )

    logger.info(
        "Validated experiment config (evaluation_ks=%s, tfidf_grid=%d, bm25_grid=%d, "
        "candidate_policy=%s, alignment_policy=%s)",
        evaluation_ks,
        len(tfidf_grid),
        len(bm25_grid),
        candidate_policy,
        alignment_policy,
    )
    return ExperimentConfig(
        evaluation_ks=evaluation_ks,
        tfidf_grid=tfidf_grid,
        bm25_grid=bm25_grid,
        candidate_policy=candidate_policy,
        alignment_policy=alignment_policy,
        char_ngram_grid=tuple(char_ngram_grid),
        dense_models=tuple(dense_models),
        hybrid_models=tuple(hybrid_models),
    )


def _validate_heldout_config(raw_heldout: Any) -> HeldoutConfig:
    heldout = _ensure_mapping(raw_heldout, "heldout")
    selection = _ensure_mapping(heldout.get("selection"), "heldout.selection")

    metric = str(selection.get("metric"))
    lambda_penalty = float(_ensure_number(selection.get("lambda"), "heldout.selection.lambda"))
    weighting = str(selection.get("weighting"))
    ranking = str(selection.get("ranking"))

    expected = REQUIRED_HELDOUT_SELECTION_POLICY
    mismatches: list[str] = []
    if metric != expected["metric"]:
        mismatches.append(f"metric={metric!r}")
    if lambda_penalty != expected["lambda"]:
        mismatches.append(f"lambda={lambda_penalty!r}")
    if weighting != expected["weighting"]:
        mismatches.append(f"weighting={weighting!r}")
    if ranking != expected["ranking"]:
        mismatches.append(f"ranking={ranking!r}")
    if mismatches:
        raise ValueError(
            "heldout.selection must match the fixed supported policy: "
            + ", ".join(f"{key}={value!r}" for key, value in expected.items())
            + f". Got: {', '.join(mismatches)}"
        )

    logger.info(
        "Validated heldout selection policy (metric=%s, lambda=%s, weighting=%s, ranking=%s)",
        metric,
        lambda_penalty,
        weighting,
        ranking,
    )
    return HeldoutConfig(
        selection=HeldoutSelectionPolicy(
            metric=metric,
            lambda_penalty=lambda_penalty,
            weighting=weighting,
            ranking=ranking,
        )
    )


def load_runtime_config(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    require_heldout: bool = False,
) -> RuntimeConfig:
    """Load and validate runtime configuration from YAML."""
    path = Path(config_path)
    logger.info("Loading runtime config from %s", path)
    content = _load_config_content(path)

    if "development_datasets" not in content:
        logger.error("Config missing 'development_datasets' mapping: %s", path)
        raise ValueError("Config must contain a 'development_datasets' mapping.")
    if "experiments" not in content:
        logger.error("Config missing 'experiments' mapping: %s", path)
        raise ValueError("Config must contain an 'experiments' mapping.")
    if require_heldout and "heldout_datasets" not in content:
        logger.error("Config missing 'heldout_datasets' mapping: %s", path)
        raise ValueError("Config must contain a 'heldout_datasets' mapping.")
    if require_heldout and "heldout" not in content:
        logger.error("Config missing 'heldout' mapping: %s", path)
        raise ValueError("Config must contain a 'heldout' mapping.")

    data_root = _resolve_data_root(content, path)
    if data_root is not None:
        logger.info("Resolving relative dataset paths against data_root=%s", data_root)
    development_datasets = _validate_dataset_group(
        content["development_datasets"],
        field_name="development_datasets",
        path=path,
        data_root=data_root,
    )
    heldout_datasets = (
        _validate_dataset_group(
            content["heldout_datasets"],
            field_name="heldout_datasets",
            path=path,
            data_root=data_root,
        )
        if "heldout_datasets" in content
        else {}
    )
    heldout_secondary_datasets = (
        _validate_secondary_heldout_dataset_group(
            content["heldout_secondary_datasets"],
            path=path,
            data_root=data_root,
        )
        if "heldout_secondary_datasets" in content
        else {}
    )
    experiments = _validate_experiment_config(content["experiments"])
    heldout = _validate_heldout_config(content["heldout"]) if "heldout" in content else None
    logger.info(
        "Loaded %d validated development dataset(s) and %d heldout dataset(s) from %s",
        len(development_datasets),
        len(heldout_datasets),
        path,
    )
    return RuntimeConfig(
        development_datasets=development_datasets,
        heldout_datasets=heldout_datasets,
        heldout_secondary_datasets=heldout_secondary_datasets,
        experiments=experiments,
        heldout=heldout,
    )


def get_dataset_config(name: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> DatasetConfig:
    """Return one development dataset configuration by name."""
    datasets = load_runtime_config(config_path=config_path).development_datasets
    if name not in datasets:
        raise KeyError(f"Dataset '{name}' not found in config.")
    return datasets[name]
