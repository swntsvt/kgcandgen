import csv
import os
import tempfile
import unittest
from pathlib import Path

from src.analysis.model_comparison import (
    ComparisonValidationError,
    generate_model_comparison,
)


class ModelComparisonTests(unittest.TestCase):
    def _write_results_csv(self, path: Path) -> None:
        rows = [
            # biodiv dataset: best MRR and best Recall@10 come from different rows.
            {
                "track": "biodiv",
                "dataset": "d1",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"a"}',
                "mrr": 0.80,
                "recall_at_10": 0.90,
                "recall_at_50": 0.99,
            },
            {
                "track": "biodiv",
                "dataset": "d1",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"b"}',
                "mrr": 0.70,
                "recall_at_10": 0.95,
                "recall_at_50": 1.00,
            },
            {
                "track": "biodiv",
                "dataset": "d1",
                "method": "bm25",
                "hyperparameters": '{"cfg":"c"}',
                "mrr": 0.85,
                "recall_at_10": 0.88,
                "recall_at_50": 0.97,
            },
            {
                "track": "biodiv",
                "dataset": "d1",
                "method": "bm25",
                "hyperparameters": '{"cfg":"d"}',
                "mrr": 0.82,
                "recall_at_10": 0.92,
                "recall_at_50": 0.98,
            },
            # conference dataset: exact tie for best MRR.
            {
                "track": "conference",
                "dataset": "d2",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"a"}',
                "mrr": 0.60,
                "recall_at_10": 0.70,
                "recall_at_50": 1.00,
            },
            {
                "track": "conference",
                "dataset": "d2",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"b"}',
                "mrr": 0.55,
                "recall_at_10": 0.80,
                "recall_at_50": 0.95,
            },
            {
                "track": "conference",
                "dataset": "d2",
                "method": "bm25",
                "hyperparameters": '{"cfg":"c"}',
                "mrr": 0.60,
                "recall_at_10": 0.70,
                "recall_at_50": 1.00,
            },
            {
                "track": "conference",
                "dataset": "d2",
                "method": "bm25",
                "hyperparameters": '{"cfg":"d"}',
                "mrr": 0.58,
                "recall_at_10": 0.75,
                "recall_at_50": 0.98,
            },
            # anatomy dataset: TF-IDF win.
            {
                "track": "anatomy_track",
                "dataset": "d3",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"a"}',
                "mrr": 0.90,
                "recall_at_10": 1.00,
                "recall_at_50": 1.00,
            },
            {
                "track": "anatomy_track",
                "dataset": "d3",
                "method": "bm25",
                "hyperparameters": '{"cfg":"c"}',
                "mrr": 0.85,
                "recall_at_10": 0.95,
                "recall_at_50": 0.99,
            },
        ]

        fieldnames = [
            "track",
            "dataset",
            "method",
            "hyperparameters",
            "mrr",
            "recall_at_10",
            "recall_at_50",
        ]
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_generates_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_csv = tmp / "result_fixture.csv"
            self._write_results_csv(source_csv)

            artifacts = generate_model_comparison(
                results_csv_path=source_csv,
                output_dir=tmp / "comparisons",
            )

            expected_keys = {
                "source_csv",
                "output_dir",
                "best_of_grid_summary",
                "aggregate_of_grid_summary",
                "wins_overall",
                "wins_by_track",
                "best_mrr_dumbbell_png",
                "best_mrr_dumbbell_pdf",
                "best_mrr_track_box_png",
                "best_mrr_track_box_pdf",
                "best_mrr_track_violin_png",
                "best_mrr_track_violin_pdf",
                "interpretation_scaffold",
                "manifest",
            }
            self.assertEqual(set(artifacts.keys()), expected_keys)
            for path in artifacts.values():
                self.assertTrue(Path(path).exists(), f"Missing artifact: {path}")

    def test_best_and_aggregate_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_csv = tmp / "result_fixture.csv"
            self._write_results_csv(source_csv)

            artifacts = generate_model_comparison(source_csv, output_dir=tmp / "comparisons")

            with Path(artifacts["best_of_grid_summary"]).open(encoding="utf-8") as best_file:
                best_rows = list(csv.DictReader(best_file))
            target = next(
                row
                for row in best_rows
                if row["track"] == "biodiv" and row["dataset"] == "d1" and row["method"] == "tfidf"
            )
            self.assertEqual(float(target["best_mrr"]), 0.80)
            self.assertEqual(float(target["best_recall_at_10"]), 0.95)
            self.assertEqual(float(target["best_recall_at_50"]), 1.00)
            self.assertEqual(target["best_mrr_hyperparameters"], '{"cfg":"a"}')
            self.assertEqual(target["best_recall_at_10_hyperparameters"], '{"cfg":"b"}')
            self.assertEqual(target["best_recall_at_50_hyperparameters"], '{"cfg":"b"}')

            with Path(artifacts["aggregate_of_grid_summary"]).open(
                encoding="utf-8"
            ) as aggregate_file:
                aggregate_rows = list(csv.DictReader(aggregate_file))
            agg_target = next(
                row
                for row in aggregate_rows
                if row["track"] == "biodiv" and row["dataset"] == "d1" and row["method"] == "bm25"
            )
            self.assertAlmostEqual(float(agg_target["mrr_mean"]), 0.835)
            self.assertAlmostEqual(float(agg_target["mrr_median"]), 0.835)
            self.assertAlmostEqual(float(agg_target["recall_at_10_mean"]), 0.90)
            self.assertAlmostEqual(float(agg_target["recall_at_50_mean"]), 0.975)

    def test_win_loss_tie_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_csv = tmp / "result_fixture.csv"
            self._write_results_csv(source_csv)

            artifacts = generate_model_comparison(source_csv, output_dir=tmp / "comparisons")

            with Path(artifacts["wins_overall"]).open(encoding="utf-8") as overall_file:
                wins_overall = list(csv.DictReader(overall_file))
            mrr_row = next(row for row in wins_overall if row["metric"] == "best_mrr")
            recall_row = next(row for row in wins_overall if row["metric"] == "best_recall_at_10")
            recall_50_row = next(row for row in wins_overall if row["metric"] == "best_recall_at_50")

            self.assertEqual(int(mrr_row["tfidf_wins"]), 1)
            self.assertEqual(int(mrr_row["bm25_wins"]), 1)
            self.assertEqual(int(mrr_row["ties"]), 1)

            self.assertEqual(int(recall_row["tfidf_wins"]), 3)
            self.assertEqual(int(recall_row["bm25_wins"]), 0)
            self.assertEqual(int(recall_row["ties"]), 0)

            self.assertEqual(int(recall_50_row["tfidf_wins"]), 2)
            self.assertEqual(int(recall_50_row["bm25_wins"]), 0)
            self.assertEqual(int(recall_50_row["ties"]), 1)

            with Path(artifacts["wins_by_track"]).open(
                encoding="utf-8"
            ) as by_track_file:
                wins_by_track = list(csv.DictReader(by_track_file))
            self.assertTrue(any(row["track"] == "conference" for row in wins_by_track))
            self.assertTrue(any(row["track"] == "biodiv" for row in wins_by_track))
            self.assertTrue(any(row["track"] == "anatomy_track" for row in wins_by_track))

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
                artifacts = generate_model_comparison(None, output_dir=tmp / "comparisons")
            finally:
                os.chdir(old_cwd)

            self.assertEqual(Path(artifacts["source_csv"]), new_csv.resolve())

    def test_missing_required_column_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad.csv"
            csv_path.write_text(
                "dataset,track,method,mrr\n"
                "d1,biodiv,tfidf,0.9\n"
                "d1,biodiv,bm25,0.8\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ComparisonValidationError, "Missing required comparison column"):
                generate_model_comparison(csv_path)

    def test_missing_method_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "bad.csv"
            csv_path.write_text(
                "dataset,track,method,hyperparameters,mrr,recall_at_10,recall_at_50\n"
                "d1,biodiv,tfidf,{},0.9,0.8,0.9\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ComparisonValidationError, "Missing: bm25"):
                generate_model_comparison(csv_path)

    def test_win_counts_ignore_unpaired_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "partial.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "dataset,track,method,hyperparameters,mrr,recall_at_10,recall_at_50",
                        "d1,biodiv,tfidf,{},0.9,0.9,0.9",
                        "d1,biodiv,bm25,{},0.8,0.8,0.8",
                        "d2,biodiv,tfidf,{},0.7,0.7,0.7",
                        "d3,biodiv,bm25,{},0.6,0.6,0.6",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            artifacts = generate_model_comparison(csv_path, output_dir=Path(tmpdir) / "comparisons")

            with Path(artifacts["wins_overall"]).open(encoding="utf-8") as overall_file:
                wins_overall = list(csv.DictReader(overall_file))

            mrr_row = next(row for row in wins_overall if row["metric"] == "best_mrr")
            self.assertEqual(int(mrr_row["tfidf_wins"]), 1)
            self.assertEqual(int(mrr_row["bm25_wins"]), 0)
            self.assertEqual(int(mrr_row["ties"]), 0)
            self.assertEqual(int(mrr_row["total_datasets"]), 1)


if __name__ == "__main__":
    unittest.main()
