import csv
import os
import tempfile
import unittest
from pathlib import Path

from src.analysis.depth_analysis import (
    DepthAnalysisValidationError,
    generate_depth_analysis,
)


class DepthAnalysisTests(unittest.TestCase):
    def _write_results_csv(self, path: Path) -> None:
        rows = [
            # d1 tfidf: best row by mrr should be cfg=a
            {
                "track": "biodiv",
                "version": "v1",
                "dataset": "d1",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"a"}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.10,
                "recall_at_5": 0.40,
                "recall_at_10": 0.60,
                "recall_at_20": 0.75,
                "recall_at_50": 0.82,
                "mrr": 0.50,
                "runtime_seconds": 0.2,
            },
            {
                "track": "biodiv",
                "version": "v1",
                "dataset": "d1",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"b"}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.20,
                "recall_at_5": 0.50,
                "recall_at_10": 0.70,
                "recall_at_20": 0.80,
                "recall_at_50": 0.90,
                "mrr": 0.40,
                "runtime_seconds": 0.2,
            },
            # d1 bm25
            {
                "track": "biodiv",
                "version": "v1",
                "dataset": "d1",
                "method": "bm25",
                "hyperparameters": '{"k1":1.2,"b":0.75}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.12,
                "recall_at_5": 0.45,
                "recall_at_10": 0.62,
                "recall_at_20": 0.76,
                "recall_at_50": 0.85,
                "mrr": 0.55,
                "runtime_seconds": 0.2,
            },
            # d2 with candidate ceiling < 50 for both methods
            {
                "track": "conference",
                "version": "v1",
                "dataset": "d2",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"x"}',
                "gold_count": 10,
                "candidate_size": 38,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.20,
                "recall_at_5": 0.55,
                "recall_at_10": 0.72,
                "recall_at_20": 0.80,
                "recall_at_50": 0.84,
                "mrr": 0.60,
                "runtime_seconds": 0.2,
            },
            {
                "track": "conference",
                "version": "v1",
                "dataset": "d2",
                "method": "bm25",
                "hyperparameters": '{"k1":1.8,"b":0.75}',
                "gold_count": 10,
                "candidate_size": 49,
                "dataset_prep_seconds": 0.1,
                "recall_at_1": 0.21,
                "recall_at_5": 0.58,
                "recall_at_10": 0.73,
                "recall_at_20": 0.83,
                "recall_at_50": 0.86,
                "mrr": 0.62,
                "runtime_seconds": 0.2,
            },
        ]
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_generates_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_csv = tmp / "result_fixture.csv"
            self._write_results_csv(source_csv)

            artifacts = generate_depth_analysis(
                results_csv_path=source_csv,
                output_dir=tmp / "comparisons",
            )

            expected_keys = {
                "source_csv",
                "output_dir",
                "depth_best_settings",
                "depth_marginal_gains",
                "depth_gain_summary_overall",
                "depth_gain_summary_by_track",
                "depth_transition_coverage",
                "depth_recall_gain_curves_by_method_png",
                "depth_recall_gain_curves_by_method_pdf",
                "depth_marginal_gain_bars_png",
                "depth_marginal_gain_bars_pdf",
                "depth_transition_coverage_png",
                "depth_transition_coverage_pdf",
                "depth_analysis_interpretation",
                "manifest",
            }
            self.assertEqual(set(artifacts.keys()), expected_keys)
            for artifact_path in artifacts.values():
                self.assertTrue(Path(artifact_path).exists())

    def test_best_settings_and_cap_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_csv = tmp / "result_fixture.csv"
            self._write_results_csv(source_csv)

            artifacts = generate_depth_analysis(source_csv, output_dir=tmp / "comparisons")

            with Path(artifacts["depth_best_settings"]).open(encoding="utf-8") as best_file:
                best_rows = list(csv.DictReader(best_file))

            d1_tfidf = next(
                row
                for row in best_rows
                if row["dataset"] == "d1" and row["method"] == "tfidf"
            )
            self.assertEqual(float(d1_tfidf["mrr"]), 0.50)
            self.assertEqual(int(d1_tfidf["effective_k_50"]), 50)

            d2_tfidf = next(
                row
                for row in best_rows
                if row["dataset"] == "d2" and row["method"] == "tfidf"
            )
            self.assertEqual(int(d2_tfidf["candidate_size"]), 38)
            self.assertEqual(int(d2_tfidf["effective_k_50"]), 38)

            with Path(artifacts["depth_marginal_gains"]).open(encoding="utf-8") as gains_file:
                gains_rows = list(csv.DictReader(gains_file))

            d2_tfidf_20_50 = next(
                row
                for row in gains_rows
                if row["dataset"] == "d2"
                and row["method"] == "tfidf"
                and row["transition"] == "20->50"
            )
            self.assertEqual(d2_tfidf_20_50["is_capped_before_to"], "True")
            self.assertEqual(int(d2_tfidf_20_50["effective_to_k"]), 38)
            self.assertAlmostEqual(float(d2_tfidf_20_50["marginal_gain"]), 0.04, places=7)

            with Path(artifacts["depth_transition_coverage"]).open(encoding="utf-8") as coverage_file:
                coverage_rows = list(csv.DictReader(coverage_file))

            overall_20_50 = next(
                row
                for row in coverage_rows
                if row["method"] == "ALL" and row["transition"] == "20->50"
            )
            self.assertEqual(int(overall_20_50["total_count"]), 4)
            self.assertEqual(int(overall_20_50["capped_count"]), 2)
            self.assertAlmostEqual(float(overall_20_50["capped_rate"]), 0.5, places=7)

    def test_uses_latest_results_csv_when_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_dir = tmp / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            old_csv = results_dir / "result_old.csv"
            new_csv = results_dir / "result_new.csv"
            self._write_results_csv(old_csv)
            self._write_results_csv(new_csv)
            old_time = (new_csv.stat().st_mtime - 1000, new_csv.stat().st_mtime - 1000)
            os.utime(old_csv, old_time)

            old_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                artifacts = generate_depth_analysis(None, output_dir=tmp / "comparisons")
            finally:
                os.chdir(old_cwd)

            self.assertEqual(Path(artifacts["source_csv"]), new_csv.resolve())

    def test_missing_required_column_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv = Path(tmpdir) / "bad.csv"
            bad_csv.write_text(
                "dataset,track,method,mrr\n"
                "d1,biodiv,tfidf,0.9\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DepthAnalysisValidationError, "Missing required depth-analysis column"):
                generate_depth_analysis(bad_csv, output_dir=Path(tmpdir) / "comparisons")


if __name__ == "__main__":
    unittest.main()
