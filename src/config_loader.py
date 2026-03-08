"""Dataset configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


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
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file: {path}") from exc

    if not isinstance(content, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")

    datasets = content.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Config must contain a 'datasets' mapping.")

    loaded: dict[str, DatasetConfig] = {}
    required_fields = ("track", "version", "source_rdf", "target_rdf", "alignment_rdf")

    for dataset_name, raw_config in datasets.items():
        if not isinstance(raw_config, dict):
            raise ValueError(f"Dataset '{dataset_name}' must be a mapping.")

        missing = [field for field in required_fields if field not in raw_config]
        if missing:
            missing_str = ", ".join(missing)
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

    return loaded


def get_dataset_config(name: str, config_path: str | Path = "config/datasets.yaml") -> DatasetConfig:
    """Return one dataset configuration by name."""
    datasets = load_datasets_config(config_path=config_path)
    if name not in datasets:
        raise KeyError(f"Dataset '{name}' not found in config.")
    return datasets[name]
