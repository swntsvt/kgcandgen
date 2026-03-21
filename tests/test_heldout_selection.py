import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.heldout_selection import (
    HeldoutSelectionValidationError,
    generate_heldout_selection,
)


class HeldoutSelectionTests(unittest.TestCase):
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
                    "    - ngram_range: [1, 2]",
                    "      min_df: 1",
                    "      max_df: 1.0",
                    "      sublinear_tf: false",
                    "  bm25_grid:",
                    "    - k1: 0.6",
                    "      b: 0.3",
                    "    - k1: 1.2",
                    "      b: 0.75",
                    "development_datasets:",
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
                    "heldout_datasets:",
                    "  kg1:",
                    "    track: kg",
                    '    version: "heldout-v1"',
                    f"    source_rdf: {source}",
                    f"    target_rdf: {target}",
                    f"    alignment_rdf: {alignment}",
                    "heldout:",
                    "  selection:",
                    "    metric: mrr",
                    "    lambda: 0.5",
                    "    weighting: equal_track_weight",
                    "    ranking: per_track_normalized_rank",
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
                "hyperparameters": '{"sublinear_tf":false,"max_df":1.0,"min_df":1,"ngram_range":[1,1]}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_10": 0.8,
                "recall_at_50": 0.9,
                "mrr": 0.8,
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
                "recall_at_10": 0.7,
                "recall_at_50": 0.85,
                "mrr": 0.7,
                "runtime_seconds": 0.01,
            },
            {
                "track": "biodiv",
                "version": "v1",
                "dataset": "d1",
                "method": "tfidf",
                "hyperparameters": '{"ngram_range":[1,2],"min_df":1,"max_df":1.0,"sublinear_tf":false}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_10": 0.6,
                "recall_at_50": 0.75,
                "mrr": 0.6,
                "runtime_seconds": 0.01,
            },
            {
                "track": "conference",
                "version": "v1",
                "dataset": "d2",
                "method": "tfidf",
                "hyperparameters": '{"ngram_range":[1,2],"min_df":1,"max_df":1.0,"sublinear_tf":false}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_10": 0.9,
                "recall_at_50": 0.95,
                "mrr": 0.9,
                "runtime_seconds": 0.01,
            },
            {
                "track": "biodiv",
                "version": "v1",
                "dataset": "d1",
                "method": "bm25",
                "hyperparameters": '{"k1":0.6,"b":0.3}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_10": 0.6,
                "recall_at_50": 0.8,
                "mrr": 0.6,
                "runtime_seconds": 0.01,
            },
            {
                "track": "conference",
                "version": "v1",
                "dataset": "d2",
                "method": "bm25",
                "hyperparameters": '{"b":0.3,"k1":0.6}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_10": 0.6,
                "recall_at_50": 0.8,
                "mrr": 0.6,
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
                "recall_at_10": 0.4,
                "recall_at_50": 0.7,
                "mrr": 0.4,
                "runtime_seconds": 0.01,
            },
            {
                "track": "conference",
                "version": "v1",
                "dataset": "d2",
                "method": "bm25",
                "hyperparameters": '{"k1":1.2,"b":0.75}',
                "gold_count": 10,
                "candidate_size": 50,
                "dataset_prep_seconds": 0.1,
                "recall_at_10": 0.3,
                "recall_at_50": 0.6,
                "mrr": 0.3,
                "runtime_seconds": 0.01,
            },
        ]
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_generates_selection_artifacts_and_expected_winners(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.rdf"
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            source.write_text("", encoding="utf-8")
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config = tmp / "runtime.yaml"
            results = tmp / "result_fixture.csv"
            output_dir = tmp / "comparisons"
            self._write_config(config, source, target, alignment)
            self._write_results(results)

            artifacts = generate_heldout_selection(
                results_csv_path=results,
                config_path=config,
                output_dir=output_dir,
            )

            expected_keys = {
                "source_csv",
                "output_dir",
                "heldout_selection_summary",
                "heldout_selected_settings",
                "heldout_selection_manifest",
            }
            self.assertEqual(set(artifacts.keys()), expected_keys)
            for path in artifacts.values():
                self.assertTrue(Path(path).exists(), f"Missing artifact: {path}")

            with Path(artifacts["heldout_selection_summary"]).open(encoding="utf-8") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))
            self.assertEqual(len(summary_rows), 4)

            tfidf_selected = next(
                row
                for row in summary_rows
                if row["method"] == "tfidf" and row["selected"] == "True"
            )
            self.assertAlmostEqual(float(tfidf_selected["mu"]), 0.5)
            self.assertAlmostEqual(float(tfidf_selected["sigma"]), 0.5)
            self.assertAlmostEqual(float(tfidf_selected["heldout_score"]), 0.25)
            self.assertEqual(
                tfidf_selected["hyperparameters"],
                '{"max_df":1.0,"min_df":1,"ngram_range":[1,1],"sublinear_tf":false}',
            )

            bm25_selected = next(
                row
                for row in summary_rows
                if row["method"] == "bm25" and row["selected"] == "True"
            )
            self.assertAlmostEqual(float(bm25_selected["mu"]), 1.0)
            self.assertAlmostEqual(float(bm25_selected["sigma"]), 0.0)
            self.assertAlmostEqual(float(bm25_selected["heldout_score"]), 1.0)

            payload = json.loads(
                Path(artifacts["heldout_selected_settings"]).read_text(encoding="utf-8")
            )
            self.assertEqual(set(payload["selected_settings"].keys()), {"tfidf", "bm25"})
            self.assertEqual(payload["policy"]["metric"], "mrr")
            self.assertEqual(payload["policy"]["lambda"], 0.5)
            self.assertEqual(payload["heldout_datasets"], ["kg1"])
            self.assertNotIn("generated_at", payload)

            first_json = Path(artifacts["heldout_selected_settings"]).read_text(encoding="utf-8")
            second_artifacts = generate_heldout_selection(
                results_csv_path=results,
                config_path=config,
                output_dir=output_dir,
            )
            second_json = Path(second_artifacts["heldout_selected_settings"]).read_text(
                encoding="utf-8"
            )
            self.assertEqual(first_json, second_json)

    def test_malformed_hyperparameters_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.rdf"
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            source.write_text("", encoding="utf-8")
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config = tmp / "runtime.yaml"
            results = tmp / "result_bad.csv"
            self._write_config(config, source, target, alignment)
            results.write_text(
                "dataset,track,method,hyperparameters,mrr\n"
                'd1,biodiv,tfidf,"{bad_json}",0.4\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(HeldoutSelectionValidationError, "Malformed hyperparameters JSON"):
                generate_heldout_selection(results, config_path=config, output_dir=tmp / "comparisons")

    def test_missing_track_coverage_counts_against_setting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.rdf"
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            source.write_text("", encoding="utf-8")
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config = tmp / "runtime.yaml"
            results = tmp / "result_missing_track.csv"
            self._write_config(config, source, target, alignment)
            with results.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["track", "version", "dataset", "method", "hyperparameters", "mrr"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "track": "biodiv",
                            "version": "v1",
                            "dataset": "d1",
                            "method": "tfidf",
                            "hyperparameters": '{"ngram_range":[1,1],"min_df":1,"max_df":1.0,"sublinear_tf":false}',
                            "mrr": 1.0,
                        },
                        {
                            "track": "biodiv",
                            "version": "v1",
                            "dataset": "d1",
                            "method": "tfidf",
                            "hyperparameters": '{"ngram_range":[1,2],"min_df":1,"max_df":1.0,"sublinear_tf":false}',
                            "mrr": 0.9,
                        },
                        {
                            "track": "conference",
                            "version": "v1",
                            "dataset": "d2",
                            "method": "tfidf",
                            "hyperparameters": '{"ngram_range":[1,2],"min_df":1,"max_df":1.0,"sublinear_tf":false}',
                            "mrr": 0.8,
                        },
                        {
                            "track": "biodiv",
                            "version": "v1",
                            "dataset": "d1",
                            "method": "tfidf",
                            "hyperparameters": '{"ngram_range":[2,2],"min_df":1,"max_df":1.0,"sublinear_tf":false}',
                            "mrr": 0.7,
                        },
                        {
                            "track": "conference",
                            "version": "v1",
                            "dataset": "d2",
                            "method": "tfidf",
                            "hyperparameters": '{"ngram_range":[2,2],"min_df":1,"max_df":1.0,"sublinear_tf":false}',
                            "mrr": 0.6,
                        },
                        {
                            "track": "biodiv",
                            "version": "v1",
                            "dataset": "d1",
                            "method": "bm25",
                            "hyperparameters": '{"k1":0.6,"b":0.3}',
                            "mrr": 0.6,
                        },
                        {
                            "track": "conference",
                            "version": "v1",
                            "dataset": "d2",
                            "method": "bm25",
                            "hyperparameters": '{"k1":0.6,"b":0.3}',
                            "mrr": 0.6,
                        },
                    ]
                )

            artifacts = generate_heldout_selection(
                results_csv_path=results,
                config_path=config,
                output_dir=tmp / "comparisons",
            )

            with Path(artifacts["heldout_selection_summary"]).open(encoding="utf-8") as summary_file:
                summary_rows = list(csv.DictReader(summary_file))

            tfidf_rows = [row for row in summary_rows if row["method"] == "tfidf"]
            selected = next(row for row in tfidf_rows if row["selected"] == "True")
            penalized = next(
                row
                for row in tfidf_rows
                if row["hyperparameters"] == '{"max_df":1.0,"min_df":1,"ngram_range":[1,1],"sublinear_tf":false}'
            )

            self.assertEqual(
                selected["hyperparameters"],
                '{"max_df":1.0,"min_df":1,"ngram_range":[1,2],"sublinear_tf":false}',
            )
            self.assertAlmostEqual(float(penalized["mu"]), 0.5)
            self.assertAlmostEqual(float(penalized["sigma"]), 0.5)
            self.assertAlmostEqual(float(penalized["heldout_score"]), 0.25)
            self.assertEqual(int(penalized["tracks_observed"]), 1)
            self.assertEqual(int(penalized["tracks_total"]), 2)
            self.assertEqual(
                json.loads(penalized["track_statuses_json"]),
                {"biodiv": "observed", "conference": "missing"},
            )
            self.assertEqual(
                json.loads(penalized["track_normalized_scores_json"]),
                {"biodiv": 1.0, "conference": 0.0},
            )

    def test_unknown_dataset_in_results_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.rdf"
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            source.write_text("", encoding="utf-8")
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config = tmp / "runtime.yaml"
            results = tmp / "result_bad.csv"
            self._write_config(config, source, target, alignment)
            with results.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["dataset", "track", "method", "hyperparameters", "mrr"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset": "heldout_only",
                        "track": "kg",
                        "method": "tfidf",
                        "hyperparameters": '{"ngram_range":[1,1],"min_df":1,"max_df":1.0,"sublinear_tf":false}',
                        "mrr": 0.4,
                    }
                )

            with self.assertRaisesRegex(HeldoutSelectionValidationError, "outside development_datasets"):
                generate_heldout_selection(results, config_path=config, output_dir=tmp / "comparisons")


if __name__ == "__main__":
    unittest.main()
