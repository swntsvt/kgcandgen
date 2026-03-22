"""Deterministic inferential helpers for held-out pairwise comparisons.

Sign-flip p-value scalability note
---------------------------------
Problem:
- Exact paired sign-flip testing enumerates ``2^n`` sign assignments for ``n``
  non-zero paired deltas. This grows exponentially and can become impractical
  for larger held-out comparisons.

Alternatives considered:
- Keep full exact enumeration always (simple but not scalable).
- Use asymptotic approximations only (fast but less faithful for small n).
- Use randomization/permutation sampling for large n.

Chosen approach:
- Hybrid exact + deterministic randomization:
  - For small ``n``, use exact enumeration.
  - Above a fixed threshold, use a deterministic Monte Carlo sign-flip
    approximation with a fixed seed and fixed number of draws.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np

BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1729
SIGN_FLIP_EXACT_MAX_NONZERO = 20
SIGN_FLIP_MONTE_CARLO_DRAWS = 100_000


def coerce_deltas(values: Iterable[float]) -> np.ndarray:
    """Convert paired deltas to a stable float array."""
    deltas = np.asarray(list(values), dtype=float)
    if deltas.ndim != 1:
        raise ValueError("Paired deltas must be a one-dimensional sequence.")
    if deltas.size == 0:
        raise ValueError("Paired deltas must not be empty.")
    return deltas


def paired_bootstrap_confidence_interval(
    deltas: Iterable[float],
    *,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap CI over paired mean deltas."""
    observed = coerce_deltas(deltas)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    if resamples <= 0:
        raise ValueError("resamples must be a positive integer.")

    rng = np.random.default_rng(seed)
    sample_size = observed.size
    sample_indices = rng.integers(0, sample_size, size=(resamples, sample_size))
    resampled_means = observed[sample_indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.quantile(resampled_means, alpha))
    upper = float(np.quantile(resampled_means, 1.0 - alpha))
    return lower, upper


def exact_paired_sign_flip_p_value(deltas: Iterable[float]) -> float:
    """Return a two-sided paired sign-flip p-value for mean delta.

    Uses exact enumeration for small sample sizes and deterministic Monte Carlo
    randomization above ``SIGN_FLIP_EXACT_MAX_NONZERO``.
    """
    return paired_sign_flip_p_value(deltas)


def paired_sign_flip_p_value(
    deltas: Iterable[float],
    *,
    exact_max_nonzero: int = SIGN_FLIP_EXACT_MAX_NONZERO,
    monte_carlo_draws: int = SIGN_FLIP_MONTE_CARLO_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> float:
    """Return a two-sided paired sign-flip p-value for mean delta.

    - Exact enumeration when non-zero paired deltas <= ``exact_max_nonzero``.
    - Deterministic Monte Carlo approximation otherwise.
    """
    observed = coerce_deltas(deltas)
    nonzero = observed[observed != 0.0]
    if nonzero.size == 0:
        return 1.0
    if exact_max_nonzero < 0:
        raise ValueError("exact_max_nonzero must be non-negative.")
    if monte_carlo_draws <= 0:
        raise ValueError("monte_carlo_draws must be positive.")

    absolute = np.abs(nonzero)
    observed_statistic = float(abs(nonzero.sum()))
    tolerance = 1e-12

    if absolute.size <= exact_max_nonzero:
        extreme_assignments = 0
        total_assignments = 2 ** absolute.size
        for sign_pattern in product((-1.0, 1.0), repeat=absolute.size):
            signed_sum = float(np.dot(np.asarray(sign_pattern, dtype=float), absolute))
            if abs(signed_sum) + tolerance >= observed_statistic:
                extreme_assignments += 1
        return float(extreme_assignments / total_assignments)

    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(monte_carlo_draws, absolute.size))
    sampled_sums = signs @ absolute
    extreme = np.abs(sampled_sums) + tolerance >= observed_statistic
    # Plus-one correction avoids pathological zero p-values under sampling.
    return float((extreme.sum() + 1) / (monte_carlo_draws + 1))
