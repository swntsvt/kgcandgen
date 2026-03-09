"""Dataset configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

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


def load_datasets_config(config_path: str | Path = "config/datasets.yaml") -> dict[str, DatasetConfig]:
    """Load dataset configuration entries from YAML."""
    path = Path(config_path)
    logger.info("Loading datasets config from %s", path)
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

    datasets = content.get("datasets")
    if not isinstance(datasets, dict):
        logger.error("Config missing 'datasets' mapping: %s", path)
        raise ValueError("Config must contain a 'datasets' mapping.")
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

    logger.info("Loaded %d validated dataset config(s) from %s", len(loaded), path)
    return loaded


def get_dataset_config(name: str, config_path: str | Path = "config/datasets.yaml") -> DatasetConfig:
    """Return one dataset configuration by name."""
    datasets = load_datasets_config(config_path=config_path)
    if name not in datasets:
        raise KeyError(f"Dataset '{name}' not found in config.")
    return datasets[name]
