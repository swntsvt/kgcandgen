import unittest

from src.experiments.efficiency_benchmark import (
    _subsample_targets,
    make_retriever,
    measure_method,
    measure_method_isolated,
)

TARGET_IDS = [f"urn:t:{i}" for i in range(6)]
TARGET_LABELS = ["red apple", "green apple", "blue car", "fast car", "tall tree", "old tree"]
QUERIES = ["apple", "car", "tree", "banana"]


class EfficiencyBenchmarkTests(unittest.TestCase):
    def test_measure_method_reports_all_fields(self) -> None:
        for method, params in (
            ("bm25", {"k1": 1.2, "b": 0.75}),
            ("tfidf", {"ngram_range": [1, 1], "min_df": 1, "max_df": 1.0, "sublinear_tf": False}),
            ("exact_match", {"normalization": "light_v1"}),
            ("char_ngram", {"analyzer": "char_wb", "ngram_range": [3, 5]}),
        ):
            with self.subTest(method=method):
                result = measure_method(
                    method=method,
                    hyperparameters=params,
                    candidate_policy="nonzero",
                    target_ids=TARGET_IDS,
                    target_labels=TARGET_LABELS,
                    query_labels=QUERIES,
                    k=3,
                    warmup=2,
                    repeats=2,
                )
                self.assertGreater(result["index_bytes"], 0)
                self.assertGreaterEqual(result["build_seconds"], 0.0)
                self.assertLessEqual(result["latency_median_ms"], result["latency_p95_ms"])
                self.assertLessEqual(result["mean_candidate_count"], 3)
                self.assertGreaterEqual(result["peak_rss_delta_mb"], 0.0)

    def test_isolated_measurement_runs_in_child_process(self) -> None:
        result = measure_method_isolated(
            method="bm25",
            hyperparameters={"k1": 1.2, "b": 0.75},
            candidate_policy="nonzero",
            target_ids=TARGET_IDS,
            target_labels=TARGET_LABELS,
            query_labels=QUERIES,
            k=3,
            warmup=1,
            repeats=1,
        )
        self.assertGreater(result["peak_rss_mb"], 0.0)

    def test_subsample_keeps_required_targets_and_is_seeded(self) -> None:
        ids = [f"t{i}" for i in range(100)]
        labels = [f"label {i}" for i in range(100)]
        first = _subsample_targets(ids, labels, {"t5", "t99"}, 10, seed=1)
        second = _subsample_targets(ids, labels, {"t5", "t99"}, 10, seed=1)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 10)
        self.assertIn("t5", first[0])
        self.assertIn("t99", first[0])
        self.assertEqual(_subsample_targets(ids, labels, set(), None, seed=1)[0], ids)

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_retriever("dense", {}, "nonzero")


if __name__ == "__main__":
    unittest.main()
