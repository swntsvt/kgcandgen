import unittest

from src.evaluation.metrics import (
    compute_mrr,
    compute_recall_at_k,
    compute_recall_at_k_and_mrr,
    compute_recall_at_ks_and_mrr,
)


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = {
            "s1": [("t1", 0.9), ("t2", 0.8), ("t3", 0.7)],
            "s2": [("t3", 0.95), ("t2", 0.6)],
            "s3": [("t5", 0.5)],
        }
        self.gold = {
            "s1": "t2",  # rank 2
            "s2": "t1",  # miss
            "s3": "t5",  # rank 1
            "s4": "t9",  # missing in predictions -> zero
        }

    def test_recall_at_k_mixed_hits_and_misses(self) -> None:
        recall_at_2 = compute_recall_at_k(self.predictions, self.gold, k=2)
        # hits: s1 (yes), s2 (no), s3 (yes), s4 (no) => 2/4
        self.assertAlmostEqual(recall_at_2, 0.5)

    def test_mrr_known_value(self) -> None:
        mrr = compute_mrr(self.predictions, self.gold)
        # s1: 1/2, s2: 0, s3: 1, s4: 0 => (1.5)/4
        self.assertAlmostEqual(mrr, 0.375)

    def test_missing_prediction_entry_counts_as_zero(self) -> None:
        gold = {"missing_source": "target"}
        mrr = compute_mrr(self.predictions, gold)
        recall = compute_recall_at_k(self.predictions, gold, k=3)
        self.assertEqual(mrr, 0.0)
        self.assertEqual(recall, 0.0)

    def test_gold_target_absent_counts_as_zero(self) -> None:
        predictions = {"s1": [("x", 1.0), ("y", 0.5)]}
        gold = {"s1": "z"}
        self.assertEqual(compute_recall_at_k(predictions, gold, k=2), 0.0)
        self.assertEqual(compute_mrr(predictions, gold), 0.0)

    def test_recall_at_k_cutoff_behavior(self) -> None:
        predictions = {"s1": [("t1", 0.9), ("t2", 0.8)]}
        gold = {"s1": "t2"}
        self.assertEqual(compute_recall_at_k(predictions, gold, k=1), 0.0)
        self.assertEqual(compute_recall_at_k(predictions, gold, k=2), 1.0)

    def test_invalid_k_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_recall_at_k(self.predictions, self.gold, k=0)
        with self.assertRaises(ValueError):
            compute_recall_at_k(self.predictions, self.gold, k=-1)

    def test_empty_gold_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_recall_at_k(self.predictions, {}, k=1)
        with self.assertRaises(ValueError):
            compute_mrr(self.predictions, {})

    def test_deterministic_repeated_calls(self) -> None:
        first_recall = compute_recall_at_k(self.predictions, self.gold, k=2)
        second_recall = compute_recall_at_k(self.predictions, self.gold, k=2)
        first_mrr = compute_mrr(self.predictions, self.gold)
        second_mrr = compute_mrr(self.predictions, self.gold)

        self.assertEqual(first_recall, second_recall)
        self.assertEqual(first_mrr, second_mrr)

    def test_combined_metric_function_matches_individual_results(self) -> None:
        recall = compute_recall_at_k(self.predictions, self.gold, k=2)
        mrr = compute_mrr(self.predictions, self.gold)
        combined_recall, combined_mrr = compute_recall_at_k_and_mrr(
            self.predictions, self.gold, k=2
        )

        self.assertEqual(recall, combined_recall)
        self.assertEqual(mrr, combined_mrr)

    def test_combined_metric_validation(self) -> None:
        with self.assertRaises(ValueError):
            compute_recall_at_k_and_mrr(self.predictions, self.gold, k=0)
        with self.assertRaises(ValueError):
            compute_recall_at_k_and_mrr(self.predictions, {}, k=1)

    def test_multi_k_combined_function_matches_individual_results(self) -> None:
        ks = [1, 2, 5]
        recalls, mrr = compute_recall_at_ks_and_mrr(self.predictions, self.gold, ks)
        self.assertEqual(mrr, compute_mrr(self.predictions, self.gold))
        for k in ks:
            self.assertEqual(recalls[k], compute_recall_at_k(self.predictions, self.gold, k))

    def test_multi_k_combined_validation(self) -> None:
        with self.assertRaises(ValueError):
            compute_recall_at_ks_and_mrr(self.predictions, self.gold, [])
        with self.assertRaises(ValueError):
            compute_recall_at_ks_and_mrr(self.predictions, self.gold, [0, 1])
        with self.assertRaises(ValueError):
            compute_recall_at_ks_and_mrr(self.predictions, {}, [1])


if __name__ == "__main__":
    unittest.main()
