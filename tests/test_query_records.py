import csv
import tempfile
import unittest
from pathlib import Path

from src.experiments.query_records import (
    QUERY_CSV_COLUMNS,
    build_query_records,
    query_output_csv_path,
    retrieval_band,
    write_query_csv,
)


class QueryRecordsTests(unittest.TestCase):
    def test_query_path_mapping_per_tier(self) -> None:
        cases = {
            "result_20260101_000000_abc.csv": "query_result_20260101_000000_abc.csv",
            "heldout_result_x.csv": "heldout_query_result_x.csv",
            "heldout_class_result_x.csv": "heldout_class_query_result_x.csv",
            "custom.csv": "custom_query.csv",
        }
        for results_name, query_name in cases.items():
            self.assertEqual(query_output_csv_path(Path("r") / results_name).name, query_name)

    def test_one_record_per_gold_pair_with_scores(self) -> None:
        records = build_query_records(
            track="t",
            version="v",
            dataset_name="d",
            entity_type="class",
            method_name="bm25",
            hyperparameters={"k1": 1.2},
            eval_sources=["s1", "s2"],
            gold={"s1": ("a", "c"), "s2": ("z",)},
            predictions={"s1": [("a", 2.0), ("b", 1.0)], "s2": []},
            source_label_map={"s1": "S1", "s2": "S2"},
            target_label_map={"a": "A", "c": "C"},
            k_max=50,
        )

        by_pair = {(r["source_entity"], r["gold_target"]): r for r in records}
        self.assertEqual(len(records), 3)
        self.assertEqual(by_pair[("s1", "a")]["gold_rank"], 1)
        self.assertEqual(by_pair[("s1", "a")]["gold_score"], 2.0)
        self.assertEqual(by_pair[("s1", "a")]["kth_score"], 1.0)
        self.assertEqual(by_pair[("s1", "c")]["gold_rank"], 0)
        self.assertIsNone(by_pair[("s1", "c")]["gold_score"])
        self.assertEqual(by_pair[("s2", "z")]["candidate_count"], 0)
        self.assertIsNone(by_pair[("s2", "z")]["kth_score"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "q.csv"
            write_query_csv(records, path)
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(QUERY_CSV_COLUMNS))
                rows = list(reader)
        self.assertEqual([row["gold_target"] for row in rows], ["a", "c", "z"])
        self.assertEqual(rows[1]["gold_score"], "")

    def test_retrieval_band(self) -> None:
        self.assertEqual(
            [retrieval_band(r) for r in (0, 1, 10, 11, 50, 51)],
            ["missed", "strong", "strong", "weak", "weak", "missed"],
        )


if __name__ == "__main__":
    unittest.main()
