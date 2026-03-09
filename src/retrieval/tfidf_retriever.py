"""TF-IDF retrieval utilities for candidate generation."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.preprocessing.text_preprocessor import preprocess_text


class TfidfRetriever:
    """Reusable TF-IDF retriever for top-k lexical candidate generation."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        sublinear_tf: bool = False,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
        )
        self._entity_ids: list[str] = []
        self._matrix = None

    def fit(self, entity_ids: list[str], labels: list[str]) -> None:
        """Fit TF-IDF index over entity labels."""
        if len(entity_ids) != len(labels):
            raise ValueError("entity_ids and labels must have the same length.")
        if not entity_ids:
            raise ValueError("entity_ids and labels must not be empty.")

        processed_labels = [" ".join(preprocess_text(label)) for label in labels]
        self._matrix = self._vectorizer.fit_transform(processed_labels)
        self._entity_ids = list(entity_ids)

    def retrieve(self, query_text: str, k: int) -> list[tuple[str, float]]:
        """Retrieve top-k candidates as (entity_id, similarity_score)."""
        if self._matrix is None or not self._entity_ids:
            raise ValueError(
                "Retriever is not fitted. Call fit(...) before retrieve(...)."
            )
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        query_processed = " ".join(preprocess_text(query_text))
        query_vector = self._vectorizer.transform([query_processed])

        # L2-normalized TF-IDF vectors make linear kernel equivalent to cosine similarity.
        scores = linear_kernel(query_vector, self._matrix).ravel()

        candidate_count = min(k, len(self._entity_ids))
        order = np.lexsort((np.arange(len(scores)), -scores))[:candidate_count]
        return [(self._entity_ids[idx], float(scores[idx])) for idx in order]
