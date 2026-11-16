"""Failed-run records for grid runs, written next to the results CSV."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ERROR_CSV_COLUMNS: tuple[str, ...] = ("scope", "dataset", "entity_type", "method", "hyperparameters", "error")


def errors_csv_path(results_csv_path: Path) -> Path:
    return results_csv_path.with_name(f"{results_csv_path.stem}_errors.csv")


def error_record(
    *, dataset: str, entity_type: str, method: str, hyperparameters: dict[str, object], exc: Exception
) -> dict[str, str]:
    logger.exception(
        "Grid run failed dataset=%s entity_type=%s method=%s hyperparameters=%s",
        dataset,
        entity_type,
        method,
        hyperparameters,
    )
    return {
        "scope": "model_run",
        "dataset": dataset,
        "entity_type": entity_type,
        "method": method,
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True, separators=(",", ":")),
        "error": f"{type(exc).__name__}: {exc}",
    }


def write_errors_csv(records: list[dict[str, str]], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ERROR_CSV_COLUMNS))
        writer.writeheader()
        writer.writerows(records)
