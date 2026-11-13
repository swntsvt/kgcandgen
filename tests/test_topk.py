import unittest

import numpy as np

from src.retrieval.topk import top_k_indices, validate_candidate_policy


def _reference(scores: np.ndarray, k: int, require_positive: bool) -> list[int]:
    order = np.lexsort((np.arange(len(scores)), -scores))
    if require_positive:
        order = [idx for idx in order if scores[idx] > 0]
    return [int(idx) for idx in list(order)[:k]]


class TopKTests(unittest.TestCase):
    def test_matches_full_lexsort_on_random_scores_with_ties(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(300):
            n = int(rng.integers(1, 80))
            # Coarse values force many ties, including at the k-th position.
            scores = rng.integers(0, 4, size=n).astype(float) / 3.0
            k = int(rng.integers(1, 60))
            for require_positive in (False, True):
                self.assertEqual(
                    top_k_indices(scores, k, require_positive=require_positive).tolist(),
                    _reference(scores, k, require_positive),
                )

    def test_ties_at_threshold_are_filled_by_ascending_index(self) -> None:
        scores = np.array([0.5, 0.9, 0.5, 0.5, 0.1])
        self.assertEqual(top_k_indices(scores, 3, require_positive=True).tolist(), [1, 0, 2])

    def test_require_positive_drops_zero_scores(self) -> None:
        scores = np.array([0.0, 0.2, 0.0])
        self.assertEqual(top_k_indices(scores, 3, require_positive=True).tolist(), [1])
        self.assertEqual(top_k_indices(scores, 3, require_positive=False).tolist(), [1, 0, 2])

    def test_all_zero_scores_with_require_positive_returns_empty(self) -> None:
        self.assertEqual(top_k_indices(np.zeros(5), 3, require_positive=True).tolist(), [])

    def test_invalid_k_raises(self) -> None:
        with self.assertRaises(ValueError):
            top_k_indices(np.ones(3), 0, require_positive=False)

    def test_validate_candidate_policy(self) -> None:
        self.assertEqual(validate_candidate_policy("nonzero"), "nonzero")
        self.assertEqual(validate_candidate_policy("padded"), "padded")
        with self.assertRaises(ValueError):
            validate_candidate_policy("all")


if __name__ == "__main__":
    unittest.main()
