"""Evaluation metrics for candidate generation.

All reciprocal ranks are computed over the returned candidate list, which retrievers cap
at ``k_max`` (50 in the paper), so every ``mrr`` here is MRR@k_max: a gold target ranked
below ``k_max`` contributes zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _validate_gold(gold: dict[str, str]) -> None:
    if not gold:
        raise ValueError("gold must not be empty.")


def _find_gold_rank(
    ranked_candidates: list[tuple[str, float]], gold_target: str, max_rank: int | None = None
) -> int:
    for rank, (target_id, _score) in enumerate(ranked_candidates, start=1):
        if max_rank is not None and rank > max_rank:
            break
        if target_id == gold_target:
            return rank
    return 0


def compute_recall_at_k(
    predictions: dict[str, list[tuple[str, float]]], gold: dict[str, str], k: int
) -> float:
    """Compute Recall@k over all gold source entities."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    _validate_gold(gold)

    hits = 0
    for source, gold_target in gold.items():
        ranked_candidates = predictions.get(source, [])
        if _find_gold_rank(ranked_candidates, gold_target, max_rank=k) > 0:
            hits += 1

    return hits / len(gold)


def compute_mrr(predictions: dict[str, list[tuple[str, float]]], gold: dict[str, str]) -> float:
    """Compute Mean Reciprocal Rank over all gold source entities."""
    _validate_gold(gold)

    reciprocal_sum = 0.0
    for source, gold_target in gold.items():
        ranked_candidates = predictions.get(source, [])
        rank = _find_gold_rank(ranked_candidates, gold_target)
        reciprocal_sum += (1.0 / rank) if rank > 0 else 0.0

    return reciprocal_sum / len(gold)


def compute_recall_at_k_and_mrr(
    predictions: dict[str, list[tuple[str, float]]], gold: dict[str, str], k: int
) -> tuple[float, float]:
    """Compute Recall@k and MRR in one pass over the gold mappings."""
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    _validate_gold(gold)

    hits = 0
    reciprocal_sum = 0.0
    for source, gold_target in gold.items():
        ranked_candidates = predictions.get(source, [])
        rank = _find_gold_rank(ranked_candidates, gold_target)
        if rank > 0:
            reciprocal_sum += 1.0 / rank
            if rank <= k:
                hits += 1

    denominator = len(gold)
    return hits / denominator, reciprocal_sum / denominator


def compute_recall_at_ks_and_mrr(
    predictions: dict[str, list[tuple[str, float]]], gold: dict[str, str], ks: Sequence[int]
) -> tuple[dict[int, float], float]:
    """Compute Recall@k for multiple cutoffs and MRR in one pass over gold mappings."""
    if not ks:
        raise ValueError("ks must not be empty.")
    if any(k <= 0 for k in ks):
        raise ValueError("all k values must be positive integers.")
    _validate_gold(gold)

    unique_ks = sorted(set(ks))
    hit_counts = {k: 0 for k in unique_ks}
    reciprocal_sum = 0.0

    for source, gold_target in gold.items():
        ranked_candidates = predictions.get(source, [])
        rank = _find_gold_rank(ranked_candidates, gold_target)
        if rank > 0:
            reciprocal_sum += 1.0 / rank
            for k in unique_ks:
                if rank <= k:
                    hit_counts[k] += 1

    denominator = len(gold)
    recalls = {k: hit_counts[k] / denominator for k in unique_ks}
    return recalls, reciprocal_sum / denominator


@dataclass(frozen=True)
class RetrievalMetrics:
    """Pair-level and any-gold (source-level) Recall@k and MRR@k_max."""

    recalls: dict[int, float]
    mrr: float
    any_gold_recalls: dict[int, float]
    any_gold_mrr: float
    pair_count: int
    query_count: int


def gold_target_ranks(
    ranked_candidates: list[tuple[str, float]], gold_targets: Sequence[str]
) -> dict[str, int]:
    """Return the 1-based rank of each gold target in the candidate list (0 = absent)."""
    position = {}
    for rank, (target_id, _score) in enumerate(ranked_candidates, start=1):
        position.setdefault(target_id, rank)
    return {target: position.get(target, 0) for target in gold_targets}


def compute_retrieval_metrics(
    predictions: Mapping[str, list[tuple[str, float]]],
    gold: Mapping[str, Sequence[str]],
    ks: Sequence[int],
) -> RetrievalMetrics:
    """Compute pair-level and any-gold Recall@k and MRR@k_max.

    Pair-level: every (source, target) gold pair is one unit, as in the paper's
    Recall@k definition; a source with two gold targets contributes two units.
    Any-gold: every source is one unit and counts as a hit at the best rank of any of
    its gold targets. With one target per source the two coincide.
    """
    if not ks:
        raise ValueError("ks must not be empty.")
    if any(k <= 0 for k in ks):
        raise ValueError("all k values must be positive integers.")
    if not gold:
        raise ValueError("gold must not be empty.")

    unique_ks = sorted(set(ks))
    pair_hits = {k: 0 for k in unique_ks}
    any_hits = {k: 0 for k in unique_ks}
    pair_rr = 0.0
    any_rr = 0.0
    pair_count = 0

    for source, targets in gold.items():
        if not targets:
            raise ValueError(f"gold source {source!r} has no targets.")
        ranks = gold_target_ranks(list(predictions.get(source, [])), targets)
        for rank in ranks.values():
            pair_count += 1
            if rank > 0:
                pair_rr += 1.0 / rank
                for k in unique_ks:
                    if rank <= k:
                        pair_hits[k] += 1
        found = [rank for rank in ranks.values() if rank > 0]
        if found:
            best = min(found)
            any_rr += 1.0 / best
            for k in unique_ks:
                if best <= k:
                    any_hits[k] += 1

    query_count = len(gold)
    return RetrievalMetrics(
        recalls={k: pair_hits[k] / pair_count for k in unique_ks},
        mrr=pair_rr / pair_count,
        any_gold_recalls={k: any_hits[k] / query_count for k in unique_ks},
        any_gold_mrr=any_rr / query_count,
        pair_count=pair_count,
        query_count=query_count,
    )


def mean_candidate_count(predictions: Mapping[str, list[tuple[str, float]]], sources: Sequence[str]) -> float:
    """Mean number of candidates actually returned per evaluated source."""
    if not sources:
        raise ValueError("sources must not be empty.")
    return sum(len(predictions.get(source, [])) for source in sources) / len(sources)
