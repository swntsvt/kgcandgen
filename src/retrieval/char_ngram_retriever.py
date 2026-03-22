"""Character n-gram TF-IDF retrieval baseline."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from typing import Literal

from src.preprocessing.exact_match_normalizer import normalize_exact_match_text

CharAnalyzer = Literal["word", "char", "char_wb"]


class CharNgramRetriever:
    """Retrieve candidates using TF-IDF over normalized character n-grams."""

    def __init__(
        self,
        *,
        analyzer: CharAnalyzer = "char_wb",
        ngram_range: tuple[int, int] = (3, 5),
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
        )
        self._entity_ids: list[str] = []
        self._matrix = None

    def fit(self, entity_ids: list[str], labels: list[str]) -> None:
        """Fit the retriever over normalized labels."""
        if len(entity_ids) != len(labels):
            raise ValueError("entity_ids and labels must have the same length.")
        if not entity_ids:
            raise ValueError("entity_ids and labels must not be empty.")

        sorted_pairs = sorted(zip(entity_ids, labels, strict=True), key=lambda pair: pair[0])
        sorted_entity_ids = [entity_id for entity_id, _label in sorted_pairs]
        normalized_labels = [
            normalize_exact_match_text(label) for _entity_id, label in sorted_pairs
        ]
        self._matrix = self._vectorizer.fit_transform(normalized_labels)
        self._entity_ids = sorted_entity_ids

    def retrieve(self, query_text: str, k: int) -> list[tuple[str, float]]:
        """Retrieve top-k candidates as (entity_id, similarity_score)."""
        if self._matrix is None or not self._entity_ids:
            raise ValueError("Retriever is not fitted. Call fit(...) before retrieve(...).")
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        normalized_query = normalize_exact_match_text(query_text)
        query_vector = self._vectorizer.transform([normalized_query])
        scores = linear_kernel(query_vector, self._matrix).ravel()

        candidate_count = min(k, len(self._entity_ids))
        order = np.lexsort((np.arange(len(scores)), -scores))[:candidate_count]
        return [(self._entity_ids[idx], float(scores[idx])) for idx in order]
