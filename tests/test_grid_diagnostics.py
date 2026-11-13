import json
import unittest

import pandas as pd

from src.analysis.grid_diagnostics import diagnose_grid


def _row(method, params, dataset, mrr, zero):
    return {
        "method": method,
        "hyperparameters": json.dumps(params, sort_keys=True, separators=(",", ":")),
        "dataset": dataset,
        "mrr": mrr,
        "recall_at_50": mrr,
        "zero_candidate_query_share": zero,
    }


class GridDiagnosticsTests(unittest.TestCase):
    def test_categories(self) -> None:
        uni = {"ngram_range": [1, 1], "min_df": 1}
        bi = {"ngram_range": [2, 2], "min_df": 1}
        bi2 = {"ngram_range": [2, 2], "min_df": 2}
        results = pd.DataFrame(
            [
                _row("tfidf", uni, "d1", 0.8, 0.1),
                _row("tfidf", bi, "d1", 0.1, 0.8),
                _row("tfidf", bi2, "d1", 0.0, 0.9),
                _row("bm25", {"k1": 1.2, "b": 0.75}, "d1", 0.8, 0.1),
                _row("bm25", {"k1": 0.0, "b": 0.75}, "d1", 0.7, 0.1),
                _row("exact_match", {}, "d1", 0.5, 0.4),
            ]
        )
        errors = pd.DataFrame(
            [{"scope": "model_run", "dataset": "d2", "method": "tfidf",
              "hyperparameters": json.dumps(bi2, sort_keys=True, separators=(",", ":")), "error": "x"}]
        )

        out = diagnose_grid(results, errors).set_index(["method", "hyperparameters"])["category"]

        key = lambda m, p: (m, json.dumps(p, sort_keys=True, separators=(",", ":")))
        self.assertEqual(out[key("tfidf", uni)], "valid")
        self.assertEqual(out[key("tfidf", bi)], "feature_loss")
        self.assertEqual(out[key("tfidf", bi2)], "failed")
        self.assertEqual(out[key("bm25", {"k1": 1.2, "b": 0.75})], "valid")
        self.assertEqual(out[key("bm25", {"k1": 0.0, "b": 0.75})], "reduced_scoring")
        self.assertNotIn("exact_match", out.index.get_level_values(0))

    def test_missing_columns_raise(self) -> None:
        with self.assertRaises(ValueError):
            diagnose_grid(pd.DataFrame({"method": ["tfidf"]}))


if __name__ == "__main__":
    unittest.main()
