"""Hybrid lexical-semantic baseline: reciprocal-rank fusion of char n-grams and a dense encoder.

Each component retrieves its own top ``depth`` candidates for a query: the default
character n-gram retriever (``char_wb``, 3-5) and an exact-cosine dense retriever. The
fused score of a target is

    sum over components c that returned it of 1 / (rrf_k + rank_c(target)),

with 1-based ranks (Cormack et al., 2009; ``rrf_k = 60`` is their constant). Fusion uses
ranks only, so the two score scales never need calibrating, and nothing is tuned. The
candidate set is the union of the two lists; ties in the fused score are broken by target
URI, as for every other method. Under ``candidate_policy="nonzero"`` each component
contributes only targets with a positive score, so the hybrid inherits that policy.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.retrieval.char_ngram_retriever import CharNgramRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.topk import DEFAULT_CANDIDATE_POLICY, CandidatePolicy, validate_candidate_policy

RankedList = list[tuple[str, float]]


def reciprocal_rank_fusion(ranked_lists: Sequence[RankedList], *, rrf_k: int, k: int) -> RankedList:
    """Fuse ranked lists by reciprocal rank; ties broken by ascending target id."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (target, _score) in enumerate(ranked, start=1):
            fused[target] = fused.get(target, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:k]


class HybridRetriever:
    """Rank-fused char n-gram + dense retrieval, with fixed settings."""

    def __init__(
        self,
        *,
        model_name: str,
        revision: str | None = None,
        device: str = "cpu",
        batch_size: int = 256,
        rrf_k: int = 60,
        depth: int = 50,
        candidate_policy: CandidatePolicy = DEFAULT_CANDIDATE_POLICY,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")
        if depth <= 0:
            raise ValueError("depth must be positive.")
        policy = validate_candidate_policy(candidate_policy)
        self._rrf_k = rrf_k
        self._depth = depth
        self._lexical = CharNgramRetriever(candidate_policy=policy)
        self._dense = DenseRetriever(
            model_name=model_name,
            revision=revision,
            device=device,
            batch_size=batch_size,
            candidate_policy=policy,
        )

    def fit(self, entity_ids: list[str], labels: list[str]) -> None:
        self._lexical.fit(entity_ids, labels)
        self._dense.fit(entity_ids, labels)

    def retrieve(self, query_text: str, k: int) -> RankedList:
        return self.retrieve_many([query_text], k)[0]

    def retrieve_many(self, query_texts: Sequence[str], k: int) -> list[RankedList]:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if not query_texts:
            return []
        dense_lists = self._dense.retrieve_many(query_texts, self._depth)
        return [
            reciprocal_rank_fusion(
                [self._lexical.retrieve(text, self._depth), dense_list], rrf_k=self._rrf_k, k=k
            )
            for text, dense_list in zip(query_texts, dense_lists, strict=True)
        ]
