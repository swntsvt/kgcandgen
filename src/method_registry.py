"""Shared method registry metadata for candidate-generation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.preprocessing.exact_match_normalizer import (
    EXACT_MATCH_NORMALIZATION_VERSION,
)


@dataclass(frozen=True)
class MethodDescriptor:
    """Static metadata about a supported candidate-generation method."""

    name: str
    tunable: bool
    fixed: bool
    supports_development: bool = True
    supports_heldout: bool = True
    selected_settings_required: bool = False
    fixed_hyperparameters: dict[str, object] | None = None


PRIMARY_COMPARISON_METHODS: tuple[str, str] = ("tfidf", "bm25")

REGISTERED_METHODS: tuple[MethodDescriptor, ...] = (
    MethodDescriptor(
        name="tfidf",
        tunable=True,
        fixed=False,
        selected_settings_required=True,
    ),
    MethodDescriptor(
        name="bm25",
        tunable=True,
        fixed=False,
        selected_settings_required=True,
    ),
    MethodDescriptor(
        name="exact_match",
        tunable=False,
        fixed=True,
        selected_settings_required=False,
        fixed_hyperparameters={"normalization": EXACT_MATCH_NORMALIZATION_VERSION},
    ),
)

REGISTERED_METHODS_BY_NAME = {descriptor.name: descriptor for descriptor in REGISTERED_METHODS}


def registered_method_names() -> list[str]:
    return [descriptor.name for descriptor in REGISTERED_METHODS]


def tunable_method_names() -> list[str]:
    return [descriptor.name for descriptor in REGISTERED_METHODS if descriptor.tunable]


def development_method_names() -> list[str]:
    return [descriptor.name for descriptor in REGISTERED_METHODS if descriptor.supports_development]


def heldout_method_names() -> list[str]:
    return [descriptor.name for descriptor in REGISTERED_METHODS if descriptor.supports_heldout]


def heldout_selected_method_names() -> list[str]:
    return [
        descriptor.name
        for descriptor in REGISTERED_METHODS
        if descriptor.supports_heldout and descriptor.selected_settings_required
    ]


def fixed_method_names(*, development: bool = False, heldout: bool = False) -> list[str]:
    names: list[str] = []
    for descriptor in REGISTERED_METHODS:
        if not descriptor.fixed:
            continue
        if development and not descriptor.supports_development:
            continue
        if heldout and not descriptor.supports_heldout:
            continue
        names.append(descriptor.name)
    return names


def fixed_method_hyperparameters(method_name: str) -> dict[str, object]:
    descriptor = REGISTERED_METHODS_BY_NAME.get(str(method_name))
    if descriptor is None or descriptor.fixed_hyperparameters is None:
        raise KeyError(f"No fixed hyperparameters registered for method '{method_name}'.")
    return dict(descriptor.fixed_hyperparameters)


def supports_primary_comparison(method_names: Iterable[str]) -> bool:
    observed = {str(name) for name in method_names}
    return all(method in observed for method in PRIMARY_COMPARISON_METHODS)


def ordered_method_names(method_names: Iterable[str]) -> list[str]:
    deduplicated = list(dict.fromkeys(str(name) for name in method_names))
    known = [name for name in registered_method_names() if name in deduplicated]
    unknown = sorted(name for name in deduplicated if name not in REGISTERED_METHODS_BY_NAME)
    return known + unknown
