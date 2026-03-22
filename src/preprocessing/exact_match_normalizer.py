"""Normalization utilities for the exact-match lexical baseline."""

from __future__ import annotations

import re
import string

EXACT_MATCH_NORMALIZATION_VERSION = "light_v1"


def _split_camel_case(text: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z][a-z])", r"\1 \2", text)


def normalize_exact_match_text(text: str) -> str:
    """Apply conservative normalization for the exact-match baseline.

    Policy:
    - lowercase
    - split camel case
    - normalize '-', '_' and '/' to spaces
    - strip punctuation
    - collapse repeated whitespace
    - preserve stopwords
    """

    camel_split = _split_camel_case(text).lower()
    separator_normalized = re.sub(r"[-_/]+", " ", camel_split)
    punctuation_as_space = separator_normalized.translate(
        str.maketrans({char: " " for char in string.punctuation})
    )
    return re.sub(r"\s+", " ", punctuation_as_space).strip()
