"""Text preprocessing utilities for lexical retrieval."""

from __future__ import annotations

from pathlib import Path
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

_NLTK_RESOURCES_READY = False
_PROJECT_NLTK_DATA_DIR = Path(__file__).resolve().parents[2] / "resources" / "nltk_data"
_REQUIRED_NLTK_RESOURCES: tuple[tuple[str, str], ...] = (
    ("tokenizers/punkt/english.pickle", "punkt"),
    ("tokenizers/punkt_tab/english", "punkt_tab"),
    ("corpora/stopwords/english", "stopwords"),
)


def _ensure_project_nltk_path() -> None:
    project_path = str(_PROJECT_NLTK_DATA_DIR)
    remaining_paths = [path for path in nltk.data.path if path != project_path]
    nltk.data.path[:] = [project_path, *remaining_paths]


def validate_nltk_assets() -> None:
    """Validate that required bundled NLTK assets are available locally."""
    global _NLTK_RESOURCES_READY
    _ensure_project_nltk_path()

    if _NLTK_RESOURCES_READY:
        return

    missing_entries: list[str] = []
    for resource_path, package_name in _REQUIRED_NLTK_RESOURCES:
        try:
            nltk.data.find(resource_path)
        except (LookupError, OSError) as exc:
            expected_path = _PROJECT_NLTK_DATA_DIR / resource_path
            missing_entries.append(
                f"- package='{package_name}', resource='{resource_path}', expected_at='{expected_path}' ({type(exc).__name__})"
            )

    if missing_entries:
        details = "\n".join(missing_entries)
        raise RuntimeError(
            "Required bundled NLTK resources are missing or unreadable.\n"
            f"Checked project data directory: {_PROJECT_NLTK_DATA_DIR}\n"
            "Missing resources:\n"
            f"{details}\n"
            "Restore the bundled files under resources/nltk_data before running preprocessing."
        )

    # Verify assets are loadable, not just present on disk.
    try:
        _ = stopwords.words("english")
        _ = word_tokenize("validation")
    except (LookupError, OSError, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "Bundled NLTK resources are present but unreadable/corrupt.\n"
            f"Checked project data directory: {_PROJECT_NLTK_DATA_DIR}\n"
            f"Validation failure: {type(exc).__name__}: {exc}\n"
            "Restore the bundled files under resources/nltk_data before running preprocessing."
        ) from exc

    _NLTK_RESOURCES_READY = True


def _split_camel_case(text: str) -> str:
    # Split only true camel boundaries (lowercase -> Uppercase+lowercase).
    text = re.sub(r"([a-z0-9])([A-Z][a-z])", r"\1 \2", text)
    return text


def preprocess_text(text: str) -> list[str]:
    """Normalize and tokenize text for lexical retrieval models."""
    validate_nltk_assets()

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
