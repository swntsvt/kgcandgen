import json
import unittest

import numpy as np
import pandas as pd

from src.analysis.paired_inference import (
    cluster_sign_flip_p,
    filter_to_selected,
    holm_adjust,
    paired_inference,
)


def _queries(ranks_by_method: dict[str, dict[tuple[str, str], int]]) -> pd.DataFrame:
    rows = []
    for method, ranks in ranks_by_method.items():
        for (dataset, source), rank in ranks.items():
            rows.append(
                {
                    "track": "t",
                    "dataset": dataset,
                    "entity_type": "class",
                    "method": method,
                    "hyperparameters": "{}",
                    "source_entity": source,
                    "gold_target": source + "_gold",
                    "gold_rank": rank,
                }
            )
    return pd.DataFrame(rows)


class PairedInferenceTests(unittest.TestCase):
    def test_holm_adjust(self) -> None:
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm_adjust([0.5]), [0.5])

    def test_micro_and_macro_deltas_and_sign(self) -> None:
        # d1: 3 queries where A wins (rank 1 vs 2); d2: 1 query where B wins.
        a = {("d1", "q1"): 1, ("d1", "q2"): 1, ("d1", "q3"): 1, ("d2", "q4"): 0}
        b = {("d1", "q1"): 2, ("d1", "q2"): 2, ("d1", "q3"): 2, ("d2", "q4"): 1}
        out = paired_inference(
            _queries({"a": a, "b": b}), scope="s", methods=["a", "b"], metrics=["mrr"], resamples=200
        ).iloc[0]

        self.assertEqual(out["delta_definition"], "method_a - method_b")
        self.assertAlmostEqual(out["p_floor"], 0.5)
        self.assertEqual(out["p_method"], "exact cluster sign-flip")
        self.assertAlmostEqual(out["delta_micro"], (3 * 0.5 - 1.0) / 4)
        self.assertAlmostEqual(out["delta_macro"], (0.5 + -1.0) / 2)
        self.assertEqual((out["n_datasets"], out["n_units"], out["n_pairs"]), (2, 2, 4))
        self.assertLessEqual(out["ci_low_micro"], out["delta_micro"])
        self.assertGreaterEqual(out["ci_high_micro"], out["delta_micro"])

    def test_sign_flip_floor_with_five_agreeing_clusters(self) -> None:
        rng = np.random.default_rng(0)
        p = cluster_sign_flip_p(np.array([1.0, 2.0, 1.0, 3.0, 1.0]), np.ones(5), rng=rng)
        self.assertAlmostEqual(p, 2 / 32)  # only all-plus and all-minus are as extreme

    def test_sign_flip_with_mixed_signs_is_not_significant(self) -> None:
        rng = np.random.default_rng(0)
        p = cluster_sign_flip_p(np.array([1.0, -1.0, 1.0, -1.0]), np.ones(4), rng=rng)
        self.assertEqual(p, 1.0)

    def test_identical_methods_give_zero_delta_and_p_one(self) -> None:
        ranks = {("d1", "q1"): 1, ("d2", "q2"): 3}
        out = paired_inference(
            _queries({"a": ranks, "b": ranks}), scope="s", metrics=["recall_at_1"], resamples=100
        ).iloc[0]
        self.assertEqual(out["delta_micro"], 0.0)
        self.assertEqual(out["p_micro"], 1.0)

    def test_bootstrap_is_seeded(self) -> None:
        rng = np.random.default_rng(0)
        a = {("d%d" % i, "q%d" % i): int(rng.integers(0, 5)) for i in range(8)}
        b = {key: int(rng.integers(0, 5)) for key in a}
        frame = _queries({"a": a, "b": b})
        first = paired_inference(frame, scope="s", metrics=["mrr"], resamples=300, seed=1)
        second = paired_inference(frame, scope="s", metrics=["mrr"], resamples=300, seed=1)
        pd.testing.assert_frame_equal(first, second)

    def test_filter_to_selected_requires_settings_for_grids(self) -> None:
        frame = _queries({"bm25": {("d1", "q1"): 1}})
        grid = pd.concat([frame, frame.assign(hyperparameters='{"k1":0.0}')])
        with self.assertRaises(ValueError):
            filter_to_selected(grid, None)
        kept = filter_to_selected(grid, {"bm25": {"k1": 0.0}})
        self.assertEqual(kept["hyperparameters"].tolist(), [json.dumps({"k1": 0.0}, separators=(",", ":"))])

    def test_dense_and_hybrid_baselines_are_kept_as_fixed_methods(self) -> None:
        frame = _queries({"dense_minilm": {("d1", "q1"): 1}, "hybrid_char_minilm": {("d1", "q1"): 2}})
        kept = filter_to_selected(frame, {"bm25": {"k1": 0.0}})
        self.assertEqual(sorted(kept["method"].unique()), ["dense_minilm", "hybrid_char_minilm"])


if __name__ == "__main__":
    unittest.main()
