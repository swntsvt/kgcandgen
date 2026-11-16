import unittest

from src.evaluation.metrics import (
    compute_recall_at_ks_and_mrr,
    compute_retrieval_metrics,
    gold_target_ranks,
    mean_candidate_count,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_single_target_matches_legacy_metrics(self) -> None:
        predictions = {"s1": [("t1", 0.9), ("t2", 0.1)], "s2": [("t3", 0.5), ("t4", 0.4)]}
        gold = {"s1": "t2", "s2": "t5"}

        legacy_recalls, legacy_mrr = compute_recall_at_ks_and_mrr(predictions, gold, [1, 2])
        metrics = compute_retrieval_metrics(
            predictions, {source: (target,) for source, target in gold.items()}, [1, 2]
        )

        self.assertEqual(metrics.recalls, legacy_recalls)
        self.assertAlmostEqual(metrics.mrr, legacy_mrr)
        self.assertEqual(metrics.any_gold_recalls, legacy_recalls)
        self.assertAlmostEqual(metrics.any_gold_mrr, legacy_mrr)
        self.assertEqual((metrics.pair_count, metrics.query_count), (2, 2))

    def test_pair_level_and_any_gold_differ_for_multi_target_sources(self) -> None:
        predictions = {"s1": [("t1", 0.9), ("t2", 0.5)], "s2": []}
        gold = {"s1": ("t1", "t2", "t3"), "s2": ("t4",)}

        metrics = compute_retrieval_metrics(predictions, gold, [1, 2])

        # Pair level: 4 pairs; t1 at rank 1, t2 at rank 2, t3 and t4 missed.
        self.assertEqual(metrics.recalls, {1: 0.25, 2: 0.5})
        self.assertAlmostEqual(metrics.mrr, (1.0 + 0.5) / 4)
        # Any gold: 2 queries; s1 hit at rank 1, s2 missed.
        self.assertEqual(metrics.any_gold_recalls, {1: 0.5, 2: 0.5})
        self.assertAlmostEqual(metrics.any_gold_mrr, 0.5)
        self.assertEqual((metrics.pair_count, metrics.query_count), (4, 2))

    def test_gold_target_ranks_uses_first_occurrence(self) -> None:
        ranks = gold_target_ranks([("a", 1.0), ("b", 0.5), ("a", 0.1)], ("a", "c"))
        self.assertEqual(ranks, {"a": 1, "c": 0})

    def test_mean_candidate_count_counts_missing_sources_as_zero(self) -> None:
        predictions = {"s1": [("t1", 0.9), ("t2", 0.5)]}
        self.assertEqual(mean_candidate_count(predictions, ["s1", "s2"]), 1.0)

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            compute_retrieval_metrics({}, {}, [1])
        with self.assertRaises(ValueError):
            compute_retrieval_metrics({}, {"s1": ()}, [1])
        with self.assertRaises(ValueError):
            compute_retrieval_metrics({}, {"s1": ("t1",)}, [0])


if __name__ == "__main__":
    unittest.main()
