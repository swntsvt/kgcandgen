"""Shared analysis helpers for coercion and results CSV resolution."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any, cast

_INTEGER_TEXT_PATTERN = re.compile(r"^[+-]?\d+$")


def _ensure_finite_float(value: float, *, original: object) -> float:
    if not math.isfinite(value):
        raise TypeError(f"Unsupported float value: {original!r}")
    return value


def coerce_count(value: object) -> int:
    """Coerce count-like values to int with strict validation.

    Strict behavior:
    - reject booleans
    - reject non-integral numeric values (for example 1.2)
    - accept integer-like strings / numpy scalar wrappers
    """
    if isinstance(value, bool):
        raise TypeError(f"Unsupported count value: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise TypeError(f"Unsupported count value: {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            raise TypeError(f"Unsupported count value: {value!r}")
        if not _INTEGER_TEXT_PATTERN.fullmatch(text):
            raise TypeError(f"Unsupported count value: {value!r}")
        return int(text)
    if hasattr(value, "item"):
        try:
            scalar_value = value.item()  # type: ignore[call-arg]
        except (TypeError, ValueError, AttributeError) as exc:
            raise TypeError(f"Unsupported count value: {value!r}") from exc
        return coerce_count(scalar_value)

    try:
        as_float = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Unsupported count value: {value!r}") from exc
    if as_float.is_integer():
        return int(as_float)
    raise TypeError(f"Unsupported count value: {value!r}")


def coerce_float(value: object) -> float:
    """Coerce numeric-like values to float with strict validation.

    Strict behavior:
    - reject booleans
    - accept numeric strings / numpy scalar wrappers
    """
    if isinstance(value, bool):
        raise TypeError(f"Unsupported float value: {value!r}")
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            raise TypeError(f"Unsupported float value: {value!r}")
        try:
            return _ensure_finite_float(float(text), original=value)
        except ValueError as exc:
            raise TypeError(f"Unsupported float value: {value!r}") from exc
    if isinstance(value, (int, float)):
        return _ensure_finite_float(float(value), original=value)
    if hasattr(value, "item"):
        try:
            scalar_value = value.item()  # type: ignore[call-arg]
        except (TypeError, ValueError, AttributeError) as exc:
            raise TypeError(f"Unsupported float value: {value!r}") from exc
        return coerce_float(scalar_value)
    try:
        return _ensure_finite_float(float(cast(Any, value)), original=value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Unsupported float value: {value!r}") from exc


def resolve_results_csv(
    results_csv_path: str | Path | None,
    *,
    default_glob: str,
    explicit_label: str,
    latest_not_found_message: str,
) -> Path:
    """Resolve explicit CSV path or latest matching CSV under `results/`."""
    if results_csv_path is not None:
        path = Path(results_csv_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{explicit_label} not found: {path}")
        return path

    candidates = sorted(
        Path("results").glob(default_glob),
        key=lambda candidate: candidate.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(latest_not_found_message)
    return candidates[-1].resolve()
