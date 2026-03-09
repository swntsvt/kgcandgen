"""BM25 retrieval utilities for candidate generation."""

from __future__ import annotations

import bm25s

from src.preprocessing.text_preprocessor import preprocess_text


class Bm25Retriever:
    """Reusable BM25 retriever for top-k lexical candidate generation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._model = bm25s.BM25(k1=k1, b=b)
        self._entity_ids: list[str] = []
        self._is_fitted = False

    def fit(self, entity_ids: list[str], labels: list[str]) -> None:
        """Fit BM25 index over entity labels."""
        if len(entity_ids) != len(labels):
            raise ValueError("entity_ids and labels must have the same length.")
        if not entity_ids:
            raise ValueError("entity_ids and labels must not be empty.")

        tokenized_corpus = [preprocess_text(label) for label in labels]
        self._model.index(tokenized_corpus, show_progress=False)
        self._entity_ids = list(entity_ids)
        self._is_fitted = True

    def retrieve(self, query_text: str, k: int) -> list[tuple[str, float]]:
        """Retrieve top-k candidates as (entity_id, similarity_score)."""
        if not self._is_fitted or not self._entity_ids:
            raise ValueError(
                "Retriever is not fitted. Call fit(...) before retrieve(...)."
            )
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        candidate_count = min(k, len(self._entity_ids))
        query_tokens = preprocess_text(query_text)
        documents, scores = self._model.retrieve(
            [query_tokens], k=candidate_count, show_progress=False, return_as="tuple"
        )

        doc_ids = documents[0]
        score_values = scores[0]
        ranked = sorted(
            zip(doc_ids, score_values, strict=True),
            key=lambda pair: (-float(pair[1]), int(pair[0])),
        )
        return [(self._entity_ids[int(doc_id)], float(score)) for doc_id, score in ranked]
