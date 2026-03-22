"""Exact normalized string-match retrieval baseline."""

from __future__ import annotations

from collections import defaultdict

from src.preprocessing.exact_match_normalizer import normalize_exact_match_text


class ExactMatchRetriever:
    """Retrieve candidates whose normalized labels match the normalized query exactly."""

    def __init__(self) -> None:
        self._index: dict[str, list[str]] = {}
        self._is_fitted = False

    def fit(self, entity_ids: list[str], labels: list[str]) -> None:
        """Build a deterministic exact-match index over raw labels."""
        if len(entity_ids) != len(labels):
            raise ValueError("entity_ids and labels must have the same length.")
        if not entity_ids:
            raise ValueError("entity_ids and labels must not be empty.")

        grouped: dict[str, list[str]] = defaultdict(list)
        for entity_id, label in zip(entity_ids, labels, strict=True):
            grouped[normalize_exact_match_text(label)].append(entity_id)

        self._index = {
            normalized: sorted(entity_group)
            for normalized, entity_group in grouped.items()
        }
        self._is_fitted = True

    def retrieve(self, query_text: str, k: int) -> list[tuple[str, float]]:
        """Return exact normalized matches, ranked deterministically by target URI."""
        if not self._is_fitted:
            raise ValueError("Retriever is not fitted. Call fit(...) before retrieve(...).")
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        normalized_query = normalize_exact_match_text(query_text)
        if normalized_query not in self._index:
            return []

        ranked_entity_ids = self._index[normalized_query][:k]
        return [(entity_id, 1.0) for entity_id in ranked_entity_ids]
