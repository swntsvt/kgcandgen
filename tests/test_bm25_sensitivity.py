import csv
import tempfile
import unittest
from pathlib import Path

from src.analysis.bm25_sensitivity import (
    Bm25SensitivityValidationError,
    generate_bm25_sensitivity,
)


class Bm25SensitivityTests(unittest.TestCase):
    def _write_config(self, path: Path, source: Path, target: Path, alignment: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "experiments:",
                    "  evaluation_ks: [1, 10, 50]",
                    "  tfidf_grid:",
                    "    - ngram_range: [1, 1]",
                    "      min_df: 1",
                    "      max_df: 1.0",
                    "      sublinear_tf: false",
                    "  bm25_grid:",
                    "    - k1: 0.6",
                    "      b: 0.3",
                    "    - k1: 1.2",
                    "      b: 0.75",
                    "    - k1: 1.8",
                    "      b: 1.0",
                    "datasets:",
                    "  d1:",
                    "    track: biodiv",
                    '    version: "v1"',
                    f"    source_rdf: {source}",
                    f"    target_rdf: {target}",
                    f"    alignment_rdf: {alignment}",
                    "  d2:",
                    "    track: conference",
                    '    version: "v1"',
                    f"    source_rdf: {source}",
                    f"    target_rdf: {target}",
                    f"    alignment_rdf: {alignment}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_results(self, path: Path) -> None:
        rows = [
            {
                "track": "biodiv",
                "version": "v1",
                "dataset": "d1",
                "method": "bm25",
                "hyperparameters": '{"k1":0.6,"b":0.3}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.1,
                "recall_at_10": 0.5,
                "recall_at_50": 0.8,
                "mrr": 0.4,
                "runtime_seconds": 0.01,
            },
            {
                "track": "conference",
                "version": "v1",
                "dataset": "d2",
                "method": "bm25",
                "hyperparameters": '{"k1":0.6,"b":0.3}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.1,
                "recall_at_10": 0.6,
                "recall_at_50": 0.85,
                "mrr": 0.5,
                "runtime_seconds": 0.01,
            },
            {
                "track": "biodiv",
                "version": "v1",
                "dataset": "d1",
                "method": "bm25",
                "hyperparameters": '{"k1":1.2,"b":0.75}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.1,
                "recall_at_10": 0.7,
                "recall_at_50": 0.9,
                "mrr": 0.6,
                "runtime_seconds": 0.01,
            },
            # d2 missing for (1.2, 0.75) and both datasets missing for (1.8, 1.0)
            {
                "track": "biodiv",
                "version": "v1",
                "dataset": "d1",
                "method": "tfidf",
                "hyperparameters": '{"ngram_range":[1,1],"min_df":1,"max_df":1.0,"sublinear_tf":false}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.1,
                "recall_at_10": 0.5,
                "recall_at_50": 0.8,
                "mrr": 0.4,
                "runtime_seconds": 0.01,
            },
        ]
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_generates_bm25_sensitivity_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.rdf"
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            source.write_text("", encoding="utf-8")
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config = tmp / "datasets.yaml"
            results = tmp / "result_fixture.csv"
            output_dir = tmp / "comparisons"
            self._write_config(config, source, target, alignment)
            self._write_results(results)

            artifacts = generate_bm25_sensitivity(
                results_csv_path=results,
                config_path=config,
                output_dir=output_dir,
            )

            expected_keys = {
                "source_csv",
                "output_dir",
                "bm25_sensitivity_summary",
                "bm25_sensitivity_by_track",
                "bm25_dataset_stability",
                "bm25_track_stability",
                "bm25_top_settings_with_ci",
                "bm25_failure_records",
                "bm25_mrr_heatmap_overall_png",
                "bm25_mrr_heatmap_overall_pdf",
                "bm25_mrr_heatmap_by_track_png",
                "bm25_mrr_heatmap_by_track_pdf",
                "bm25_mrr_profiles_by_b_png",
                "bm25_mrr_profiles_by_b_pdf",
                "bm25_failure_rate_heatmap_png",
                "bm25_failure_rate_heatmap_pdf",
                "bm25_sensitivity_interpretation",
                "manifest",
            }
            self.assertEqual(set(artifacts.keys()), expected_keys)
            for path in artifacts.values():
                self.assertTrue(Path(path).exists(), f"Missing artifact: {path}")

    def test_failure_inference_and_top_settings_ci(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.rdf"
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            source.write_text("", encoding="utf-8")
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config = tmp / "datasets.yaml"
            results = tmp / "result_fixture.csv"
            output_dir = tmp / "comparisons"
            self._write_config(config, source, target, alignment)
            self._write_results(results)

            artifacts = generate_bm25_sensitivity(results, config_path=config, output_dir=output_dir)

            with Path(artifacts["bm25_sensitivity_summary"]).open(encoding="utf-8") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))
            row = next(
                item for item in summary_rows if item["k1"] == "1.2" and item["b"] == "0.75"
            )
            self.assertEqual(int(row["success_count"]), 1)
            self.assertEqual(int(row["failure_count"]), 1)
            self.assertAlmostEqual(float(row["failure_rate"]), 0.5)

            with Path(artifacts["bm25_failure_records"]).open(encoding="utf-8") as failures_file:
                failures = list(csv.DictReader(failures_file))
            self.assertTrue(any(item["failure_reason"] == "missing_in_results" for item in failures))

            with Path(artifacts["bm25_top_settings_with_ci"]).open(encoding="utf-8") as top_file:
                top_rows = list(csv.DictReader(top_file))
            self.assertGreaterEqual(len(top_rows), 1)
            self.assertIn("mrr_ci_lower", top_rows[0])
            self.assertIn("mrr_ci_upper", top_rows[0])
            float(top_rows[0]["mrr_ci_lower"])
            float(top_rows[0]["mrr_ci_upper"])

    def test_invalid_hyperparameter_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.rdf"
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            source.write_text("", encoding="utf-8")
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config = tmp / "datasets.yaml"
            self._write_config(config, source, target, alignment)

            bad_results = tmp / "result_bad.csv"
            bad_results.write_text(
                "dataset,track,method,hyperparameters,mrr,recall_at_10,recall_at_50\n"
                'd1,biodiv,bm25,"{bad_json}",0.4,0.5,0.6\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(Bm25SensitivityValidationError, "Malformed BM25 hyperparameters JSON"):
                generate_bm25_sensitivity(bad_results, config_path=config, output_dir=tmp / "comparisons")


if __name__ == "__main__":
    unittest.main()
