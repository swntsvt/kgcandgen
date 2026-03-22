"""Shared method registry metadata for candidate-generation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MethodDescriptor:
    """Static metadata about a supported candidate-generation method."""

    name: str
    tunable: bool
    fixed: bool
    supports_development: bool = True
    supports_heldout: bool = True
    selected_settings_required: bool = False


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
)

REGISTERED_METHODS_BY_NAME = {descriptor.name: descriptor for descriptor in REGISTERED_METHODS}


def registered_method_names() -> list[str]:
    return [descriptor.name for descriptor in REGISTERED_METHODS]


def tunable_method_names() -> list[str]:
    return [descriptor.name for descriptor in REGISTERED_METHODS if descriptor.tunable]


def heldout_selected_method_names() -> list[str]:
    return [
        descriptor.name
        for descriptor in REGISTERED_METHODS
        if descriptor.supports_heldout and descriptor.selected_settings_required
    ]


def supports_primary_comparison(method_names: Iterable[str]) -> bool:
    observed = {str(name) for name in method_names}
    return all(method in observed for method in PRIMARY_COMPARISON_METHODS)


def ordered_method_names(method_names: Iterable[str]) -> list[str]:
    deduplicated = list(dict.fromkeys(str(name) for name in method_names))
    known = [name for name in registered_method_names() if name in deduplicated]
    unknown = sorted(name for name in deduplicated if name not in REGISTERED_METHODS_BY_NAME)
    return known + unknown
