"""Deterministic top-k selection and candidate-policy helpers shared by retrievers."""

from __future__ import annotations

from typing import Literal

import numpy as np

CandidatePolicy = Literal["nonzero", "padded"]

CANDIDATE_POLICIES: tuple[str, ...] = ("nonzero", "padded")
DEFAULT_CANDIDATE_POLICY: CandidatePolicy = "nonzero"


def validate_candidate_policy(policy: str) -> CandidatePolicy:
    """Return ``policy`` if it is a supported candidate policy, else raise ``ValueError``.

    - ``nonzero``: a target is a candidate only if its similarity score is > 0, so a
      method may return fewer than ``k`` candidates.
    - ``padded``: the submitted-version behaviour; the list is always filled to
      ``min(k, |T|)`` targets, including zero-score targets in index order.
    """
    if policy not in CANDIDATE_POLICIES:
        raise ValueError(
            f"candidate_policy must be one of {list(CANDIDATE_POLICIES)}; got {policy!r}."
        )
    return policy  # type: ignore[return-value]


def top_k_indices(scores: np.ndarray, k: int, *, require_positive: bool) -> np.ndarray:
    """Return indices of the top ``k`` scores ordered by (-score, index).

    The result is identical to ``np.lexsort((np.arange(n), -scores))[:k]`` (restricted
    to positive scores when ``require_positive``) but runs in O(n + k log k) instead
    of a full O(n log n) sort. Ties are broken by ascending index, including ties that
    straddle the k-th position, so membership and order are fully deterministic.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    scores = np.asarray(scores, dtype=float).ravel()

    if require_positive:
        pool = np.flatnonzero(scores > 0.0)
    else:
        pool = np.arange(scores.size)
    if pool.size == 0:
        return pool

    pool_scores = scores[pool]
    if pool.size > k:
        # Score of the k-th best candidate; everything strictly above it is in,
        # and ties at the threshold are filled by ascending index.
        threshold = np.partition(-pool_scores, k - 1)[k - 1]
        above = pool[-pool_scores < threshold]
        at_threshold = pool[-pool_scores == threshold]
        selected = np.concatenate((above, at_threshold[: k - above.size]))
    else:
        selected = pool

    selected_scores = scores[selected]
    order = np.lexsort((selected, -selected_scores))
    return selected[order]
