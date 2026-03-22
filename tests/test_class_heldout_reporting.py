import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.class_heldout_reporting import (
    ClassHeldoutReportingValidationError,
    generate_class_heldout_reporting,
)


class ClassHeldoutReportingTests(unittest.TestCase):
    def _write_results_csv(self, path: Path) -> None:
        rows = [
            {
                "track": "bioml-supervised",
                "version": "2022",
                "dataset": "bio1",
                "entity_type": "class",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"tfidf"}',
                "gold_count": 10,
                "target_pool_size": 100,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.50,
                "recall_at_50": 0.90,
                "mrr": 0.80,
                "runtime_seconds": 0.1,
            },
            {
                "track": "bioml-supervised",
                "version": "2022",
                "dataset": "bio1",
                "entity_type": "class",
                "method": "bm25",
                "hyperparameters": '{"cfg":"bm25"}',
                "gold_count": 10,
                "target_pool_size": 100,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.50,
                "recall_at_50": 0.85,
                "mrr": 0.75,
                "runtime_seconds": 0.1,
            },
            {
                "track": "bioml-supervised",
                "version": "2022",
                "dataset": "bio1",
                "entity_type": "class",
                "method": "exact_match",
                "hyperparameters": '{"normalization":"light_v1"}',
                "gold_count": 10,
                "target_pool_size": 100,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.50,
                "recall_at_50": 0.60,
                "mrr": 0.55,
                "runtime_seconds": 0.1,
            },
            {
                "track": "bioml-supervised",
                "version": "2022",
                "dataset": "bio1",
                "entity_type": "class",
                "method": "char_ngram",
                "hyperparameters": '{"normalization":"light_v1","analyzer":"char_wb","ngram_range":[3,5]}',
                "gold_count": 10,
                "target_pool_size": 100,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.50,
                "recall_at_50": 0.70,
                "mrr": 0.65,
                "runtime_seconds": 0.1,
            },
            {
                "track": "bioml-unsupervised",
                "version": "2022",
                "dataset": "bio2",
                "entity_type": "class",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"tfidf"}',
                "gold_count": 20,
                "target_pool_size": 80,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.375,
                "recall_at_50": 0.88,
                "mrr": 0.78,
                "runtime_seconds": 0.1,
            },
            {
                "track": "bioml-unsupervised",
                "version": "2022",
                "dataset": "bio2",
                "entity_type": "class",
                "method": "bm25",
                "hyperparameters": '{"cfg":"bm25"}',
                "gold_count": 20,
                "target_pool_size": 80,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.375,
                "recall_at_50": 0.84,
                "mrr": 0.73,
                "runtime_seconds": 0.1,
            },
            {
                "track": "bioml-unsupervised",
                "version": "2022",
                "dataset": "bio2",
                "entity_type": "class",
                "method": "exact_match",
                "hyperparameters": '{"normalization":"light_v1"}',
                "gold_count": 20,
                "target_pool_size": 80,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.375,
                "recall_at_50": 0.50,
                "mrr": 0.45,
                "runtime_seconds": 0.1,
            },
            {
                "track": "bioml-unsupervised",
                "version": "2022",
                "dataset": "bio2",
                "entity_type": "class",
                "method": "char_ngram",
                "hyperparameters": '{"normalization":"light_v1","analyzer":"char_wb","ngram_range":[3,5]}',
                "gold_count": 20,
                "target_pool_size": 80,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.375,
                "recall_at_50": 0.65,
                "mrr": 0.60,
                "runtime_seconds": 0.1,
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_selected_settings(self, path: Path) -> None:
        payload = {
            "heldout_datasets": ["bio1", "bio2"],
            "selected_settings": {
                "tfidf": {"heldout_score": 0.8, "mu": 0.78, "sigma": 0.04},
                "bm25": {"heldout_score": 0.74, "mu": 0.73, "sigma": 0.03},
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    def test_generates_separate_class_only_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "heldout_class_result_fixture.csv"
            selected_json = tmp / "heldout_selected_settings.json"
            self._write_results_csv(results_csv)
            self._write_selected_settings(selected_json)

            artifacts = generate_class_heldout_reporting(
                results_csv_path=results_csv,
                output_dir=tmp / "comparisons",
                selected_settings_path=selected_json,
            )

            expected = {
                "source_csv",
                "output_dir",
                "class_heldout_by_method_summary",
                "class_heldout_macro_summary",
                "class_heldout_micro_summary",
                "class_heldout_reduction_effectiveness",
                "class_heldout_pairwise_overall_inference",
                "class_heldout_interpretation_scaffold",
                "class_heldout_transfer_summary",
                "manifest",
            }
            self.assertEqual(set(artifacts.keys()), expected)

            with Path(artifacts["class_heldout_by_method_summary"]).open(
                encoding="utf-8"
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual({row["method"] for row in rows}, {"tfidf", "bm25", "exact_match", "char_ngram"})

    def test_fails_when_non_class_rows_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "heldout_class_result_fixture.csv"
            self._write_results_csv(results_csv)
            text = results_csv.read_text(encoding="utf-8").replace(",class,", ",instance,", 1)
            results_csv.write_text(text, encoding="utf-8")
            with self.assertRaises(ClassHeldoutReportingValidationError):
                generate_class_heldout_reporting(
                    results_csv_path=results_csv,
                    output_dir=tmp / "comparisons",
                )


if __name__ == "__main__":
    unittest.main()
