import unittest

import pandas as pd

from src.analysis.selection_regret import (
    development_rule_picks,
    heldout_config_scores,
    parse_tier_csv_arguments,
    selection_regret,
)

TF_A = '{"max_df":1.0,"min_df":1,"ngram_range":[1,1],"sublinear_tf":false}'
TF_B = '{"max_df":1.0,"min_df":1,"ngram_range":[1,2],"sublinear_tf":false}'
BM_A = '{"b":0.75,"k1":1.2}'
BM_B = '{"b":0.75,"k1":2.4}'


def _dev() -> pd.DataFrame:
    rows = []
    for track, datasets in (("t1", ["d1", "d2"]), ("t2", ["d3"]), ("t3", ["d4"])):
        for dataset in datasets:
            rows += [
                {"track": track, "dataset": dataset, "method": "tfidf", "hyperparameters": TF_A, "mrr": 0.8},
                {"track": track, "dataset": dataset, "method": "tfidf", "hyperparameters": TF_B, "mrr": 0.7},
                {"track": track, "dataset": dataset, "method": "bm25", "hyperparameters": BM_A, "mrr": 0.6},
                {"track": track, "dataset": dataset, "method": "bm25", "hyperparameters": BM_B, "mrr": 0.9},
            ]
    return pd.DataFrame(rows)


def _heldout() -> pd.DataFrame:
    rows = []
    for dataset, gold in (("h1", 10), ("h2", 90)):
        rows += [
            {"dataset": dataset, "entity_type": "class", "method": "tfidf", "hyperparameters": TF_A, "mrr": 0.5, "gold_count": gold},
            {"dataset": dataset, "entity_type": "class", "method": "tfidf", "hyperparameters": TF_B, "mrr": 0.6, "gold_count": gold},
            {"dataset": dataset, "entity_type": "class", "method": "bm25", "hyperparameters": BM_A, "mrr": 0.4, "gold_count": gold},
            {"dataset": dataset, "entity_type": "class", "method": "bm25", "hyperparameters": BM_B, "mrr": 0.7, "gold_count": gold},
            {"dataset": dataset, "entity_type": "class", "method": "char_ngram", "hyperparameters": "{}", "mrr": 0.9, "gold_count": gold},
        ]
    return pd.DataFrame(rows)


class SelectionRegretTests(unittest.TestCase):
    def test_rules_pick_dominant_configurations(self) -> None:
        picks = development_rule_picks(_dev(), lambdas=(0.0, 0.5))
        by_rule = picks.set_index(["rule", "method"])["hyperparameters"]
        for rule in ("lambda=0", "lambda=0.5", "mean_mrr_dataset_weighted", "worst_track_mrr"):
            self.assertEqual(by_rule[(rule, "tfidf")], TF_A)
            self.assertEqual(by_rule[(rule, "bm25")], BM_B)
        self.assertEqual(len(picks[picks["rule"].str.startswith("leave_out_")]), 6)

    def test_regret_against_post_hoc_best(self) -> None:
        result = selection_regret(_dev(), {"kg": _heldout()}, lambdas=(0.5,))
        row = result[(result["rule"] == "lambda=0.5") & (result["method"] == "tfidf")].iloc[0]
        self.assertAlmostEqual(row["regret_macro"], 0.1)
        self.assertEqual(row["best_macro_hyperparameters"], TF_B)
        self.assertEqual(row["rank_macro"], 2)
        bm = result[(result["rule"] == "lambda=0.5") & (result["method"] == "bm25")].iloc[0]
        self.assertAlmostEqual(bm["regret_macro"], 0.0)
        self.assertEqual(bm["configurations"], 2)

    def test_micro_weights_by_gold_pairs(self) -> None:
        scores = heldout_config_scores(_heldout()).set_index(["method", "hyperparameters"])
        self.assertAlmostEqual(scores.loc[("tfidf", TF_A), "micro_mrr"], 0.5)
        self.assertNotIn("char_ngram", scores.index.get_level_values(0))

    def test_incomplete_configurations_are_not_post_hoc_best(self) -> None:
        heldout = _heldout()
        # TF_B "wins" on h2 only because it failed on h1.
        heldout = heldout[~((heldout["hyperparameters"] == TF_B) & (heldout["dataset"] == "h1"))]
        heldout.loc[heldout["hyperparameters"] == TF_B, "mrr"] = 0.99
        result = selection_regret(_dev(), {"kg": heldout}, lambdas=(0.5,))
        row = result[(result["rule"] == "lambda=0.5") & (result["method"] == "tfidf")].iloc[0]
        self.assertEqual(row["best_macro_hyperparameters"], TF_A)
        self.assertAlmostEqual(row["regret_macro"], 0.0)
        self.assertEqual(row["configurations"], 1)

    def test_missing_configuration_raises(self) -> None:
        heldout = _heldout()
        heldout = heldout[heldout["hyperparameters"] != BM_B]
        with self.assertRaises(ValueError):
            selection_regret(_dev(), {"kg": heldout}, lambdas=(0.5,))

    def test_parse_tier_arguments(self) -> None:
        self.assertEqual(parse_tier_csv_arguments(["kg=a.csv", "bioml=b.csv"]), {"kg": "a.csv", "bioml": "b.csv"})
        with self.assertRaises(ValueError):
            parse_tier_csv_arguments(["kg"])


if __name__ == "__main__":
    unittest.main()
