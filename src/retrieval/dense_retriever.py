"""Dense (bi-encoder) retrieval baseline with exact cosine search.

Labels get the same light normalisation as the exact-match and char n-gram baselines
(camel-case split, separators to spaces, lowercase). Embeddings are L2-normalised, so the
dot product is cosine similarity, and search is exact: every target is scored, with no
approximate index. Under ``candidate_policy="nonzero"`` a target is a candidate only if its
cosine similarity is > 0; ties are broken by target URI order as for every other method.

Models are pinned to a Hugging Face revision and cached per process.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from src.preprocessing.exact_match_normalizer import normalize_exact_match_text
from src.retrieval.topk import (
    DEFAULT_CANDIDATE_POLICY,
    CandidatePolicy,
    top_k_indices,
    validate_candidate_policy,
)

QUERY_SCORE_CHUNK = 256


@lru_cache(maxsize=4)
def _load_model(model_name: str, revision: str | None, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    return SentenceTransformer(model_name, revision=revision, device=device)


class DenseRetriever:
    """Exact cosine-similarity retrieval over sentence-transformer label embeddings."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None = None,
        device: str = "cpu",
        batch_size: int = 256,
        candidate_policy: CandidatePolicy = DEFAULT_CANDIDATE_POLICY,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self._candidate_policy = validate_candidate_policy(candidate_policy)
        self._model_name = model_name
        self._revision = revision
        self._device = device
        self._batch_size = batch_size
        self._entity_ids: list[str] = []
        self._embeddings: np.ndarray | None = None

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        model = _load_model(self._model_name, self._revision, self._device)
        normalized = [normalize_exact_match_text(text) for text in texts]
        embeddings = model.encode(
            normalized,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def fit(self, entity_ids: list[str], labels: list[str]) -> None:
        if len(entity_ids) != len(labels):
            raise ValueError("entity_ids and labels must have the same length.")
        if not entity_ids:
            raise ValueError("entity_ids and labels must not be empty.")
        order = sorted(range(len(entity_ids)), key=lambda index: entity_ids[index])
        self._entity_ids = [entity_ids[index] for index in order]
        self._embeddings = self._encode([labels[index] for index in order])

    def _check_ready(self, k: int) -> np.ndarray:
        if self._embeddings is None or not self._entity_ids:
            raise ValueError("Retriever is not fitted. Call fit(...) before retrieve(...).")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        return self._embeddings

    def _rank(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        order = top_k_indices(scores, k, require_positive=self._candidate_policy == "nonzero")
        return [(self._entity_ids[index], float(scores[index])) for index in order]

    def retrieve(self, query_text: str, k: int) -> list[tuple[str, float]]:
        return self.retrieve_many([query_text], k)[0]

    def retrieve_many(self, query_texts: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Batch-encode queries, then rank exactly; results are identical to ``retrieve``."""
        embeddings = self._check_ready(k)
        if not query_texts:
            return []
        queries = self._encode(query_texts)
        results: list[list[tuple[str, float]]] = []
        for start in range(0, len(queries), QUERY_SCORE_CHUNK):
            block = queries[start : start + QUERY_SCORE_CHUNK] @ embeddings.T
            results.extend(self._rank(row, k) for row in block)
        return results
