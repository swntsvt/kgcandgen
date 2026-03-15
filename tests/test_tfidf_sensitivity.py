import csv
import tempfile
import unittest
from pathlib import Path

from src.analysis.tfidf_sensitivity import (
    TfidfSensitivityValidationError,
    generate_tfidf_sensitivity,
)


class TfidfSensitivityTests(unittest.TestCase):
    def _write_config(self, path: Path) -> None:
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
                    "    - ngram_range: [1, 2]",
                    "      min_df: 2",
                    "      max_df: 1.0",
                    "      sublinear_tf: false",
                    "    - ngram_range: [1, 2]",
                    "      min_df: 2",
                    "      max_df: 1.0",
                    "      sublinear_tf: true",
                    "  bm25_grid:",
                    "    - k1: 1.2",
                    "      b: 0.75",
                    "datasets:",
                    "  d1:",
                    "    track: biodiv",
                    '    version: "v1"',
                    "    source_rdf: /tmp/does_not_matter_source.rdf",
                    "    target_rdf: /tmp/does_not_matter_target.rdf",
                    "    alignment_rdf: /tmp/does_not_matter_alignment.rdf",
                    "  d2:",
                    "    track: conference",
                    '    version: "v1"',
                    "    source_rdf: /tmp/does_not_matter_source.rdf",
                    "    target_rdf: /tmp/does_not_matter_target.rdf",
                    "    alignment_rdf: /tmp/does_not_matter_alignment.rdf",
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
                "method": "tfidf",
                "hyperparameters": '{"ngram_range":[1,1],"min_df":1,"max_df":1.0,"sublinear_tf":false}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.1,
                "recall_at_10": 0.6,
                "recall_at_50": 0.9,
                "mrr": 0.5,
                "runtime_seconds": 0.01,
            },
            {
                "track": "conference",
                "version": "v1",
                "dataset": "d2",
                "method": "tfidf",
                "hyperparameters": '{"ngram_range":[1,1],"min_df":1,"max_df":1.0,"sublinear_tf":false}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.1,
                "recall_at_10": 0.8,
                "recall_at_50": 0.95,
                "mrr": 0.7,
                "runtime_seconds": 0.01,
            },
            {
                "track": "biodiv",
                "version": "v1",
                "dataset": "d1",
                "method": "tfidf",
                "hyperparameters": '{"ngram_range":[1,2],"min_df":2,"max_df":1.0,"sublinear_tf":false}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.1,
                "recall_at_10": 0.3,
                "recall_at_50": 0.6,
                "mrr": 0.2,
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
                "recall_at_10": 0.6,
                "recall_at_50": 0.9,
                "mrr": 0.5,
                "runtime_seconds": 0.01,
            },
        ]
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _create_dummy_rdf_files(self) -> tuple[Path, Path, Path]:
        source = Path("/tmp/does_not_matter_source.rdf")
        target = Path("/tmp/does_not_matter_target.rdf")
        alignment = Path("/tmp/does_not_matter_alignment.rdf")
        source.write_text("", encoding="utf-8")
        target.write_text("", encoding="utf-8")
        alignment.write_text("", encoding="utf-8")
        return source, target, alignment

    def test_generates_sensitivity_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = tmp / "datasets.yaml"
            results = tmp / "result_fixture.csv"
            output_dir = tmp / "comparisons"
            self._write_config(config)
            self._write_results(results)

            source, target, alignment = self._create_dummy_rdf_files()
            self.addCleanup(lambda: source.unlink(missing_ok=True))
            self.addCleanup(lambda: target.unlink(missing_ok=True))
            self.addCleanup(lambda: alignment.unlink(missing_ok=True))

            artifacts = generate_tfidf_sensitivity(
                results_csv_path=results,
                config_path=config,
                output_dir=output_dir,
            )

            expected_keys = {
                "source_csv",
                "output_dir",
                "tfidf_sensitivity_summary",
                "tfidf_sensitivity_by_track",
                "tfidf_interaction_summary",
                "tfidf_failure_records",
                "tfidf_mrr_heatmap_sublinear_false_png",
                "tfidf_mrr_heatmap_sublinear_false_pdf",
                "tfidf_mrr_heatmap_sublinear_true_png",
                "tfidf_mrr_heatmap_sublinear_true_pdf",
                "tfidf_failure_rate_heatmap_png",
                "tfidf_failure_rate_heatmap_pdf",
                "tfidf_sensitivity_interpretation",
                "manifest",
            }
            self.assertEqual(set(artifacts.keys()), expected_keys)
            for path in artifacts.values():
                self.assertTrue(Path(path).exists(), f"Missing artifact: {path}")

    def test_failure_inference_and_success_only_means(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = tmp / "datasets.yaml"
            results = tmp / "result_fixture.csv"
            output_dir = tmp / "comparisons"
            self._write_config(config)
            self._write_results(results)

            source, target, alignment = self._create_dummy_rdf_files()
            self.addCleanup(lambda: source.unlink(missing_ok=True))
            self.addCleanup(lambda: target.unlink(missing_ok=True))
            self.addCleanup(lambda: alignment.unlink(missing_ok=True))

            artifacts = generate_tfidf_sensitivity(results, config_path=config, output_dir=output_dir)

            with Path(artifacts["tfidf_sensitivity_summary"]).open(encoding="utf-8") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))

            # ngram 1-2 / min_df=2 / sublinear_tf=false appears for one dataset and is missing for one.
            target_row = next(
                row
                for row in summary_rows
                if row["ngram_range"] == "1-2" and row["min_df"] == "2" and row["sublinear_tf"] == "False"
            )
            self.assertEqual(int(target_row["success_count"]), 1)
            self.assertEqual(int(target_row["failure_count"]), 1)
            self.assertAlmostEqual(float(target_row["failure_rate"]), 0.5)
            self.assertAlmostEqual(float(target_row["mrr_mean"]), 0.2)

            with Path(artifacts["tfidf_failure_records"]).open(encoding="utf-8") as failures_file:
                failure_rows = list(csv.DictReader(failures_file))
            self.assertTrue(any(row["failure_reason"] == "missing_in_results" for row in failure_rows))

    def test_invalid_hyperparameter_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = tmp / "datasets.yaml"
            self._write_config(config)
            bad_results = tmp / "result_bad.csv"
            bad_results.write_text(
                "dataset,track,method,hyperparameters,mrr,recall_at_10,recall_at_50\n"
                'd1,biodiv,tfidf,"{bad_json}",0.4,0.5,0.6\n',
                encoding="utf-8",
            )

            source, target, alignment = self._create_dummy_rdf_files()
            self.addCleanup(lambda: source.unlink(missing_ok=True))
            self.addCleanup(lambda: target.unlink(missing_ok=True))
            self.addCleanup(lambda: alignment.unlink(missing_ok=True))

            with self.assertRaisesRegex(TfidfSensitivityValidationError, "Malformed TF-IDF hyperparameters JSON"):
                generate_tfidf_sensitivity(bad_results, config_path=config, output_dir=tmp / "comparisons")


if __name__ == "__main__":
    unittest.main()
