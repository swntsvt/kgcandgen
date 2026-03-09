"""Evaluation metrics for candidate generation."""

from __future__ import annotations

from collections.abc import Sequence


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
