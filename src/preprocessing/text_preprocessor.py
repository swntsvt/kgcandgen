"""Text preprocessing utilities for lexical retrieval."""

from __future__ import annotations

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

_NLTK_RESOURCES_READY = False


def _ensure_nltk_resources() -> None:
    global _NLTK_RESOURCES_READY
    if _NLTK_RESOURCES_READY:
        return

    resources = (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    )
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

    _NLTK_RESOURCES_READY = True


def _split_camel_case(text: str) -> str:
    # Split only true camel boundaries (lowercase -> Uppercase+lowercase).
    text = re.sub(r"([a-z0-9])([A-Z][a-z])", r"\1 \2", text)
    return text


def preprocess_text(text: str) -> list[str]:
    """Normalize and tokenize text for lexical retrieval models."""
    _ensure_nltk_resources()

    camel_split = _split_camel_case(text).lower()
    separator_normalized = re.sub(r"[-_/]+", " ", camel_split)

    tokens = word_tokenize(separator_normalized)
    stop_words = set(stopwords.words("english"))

    cleaned: list[str] = []
    for token in tokens:
        if token in stop_words:
            continue

        stripped = token.translate(str.maketrans("", "", string.punctuation))
        if not stripped:
            continue

        # Keep tokenization stable across separators like "-", "/" and "_".
        parts = re.findall(r"[a-z0-9]+", stripped)
        cleaned.extend(part for part in parts if part)

    return cleaned
