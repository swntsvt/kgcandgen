import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.analysis.heldout_inference import (
    exact_paired_sign_flip_p_value,
    paired_sign_flip_p_value,
    paired_bootstrap_confidence_interval,
)
from src.analysis.kg_heldout_reporting import (
    KgHeldoutReportingValidationError,
    generate_kg_heldout_reporting,
)


class KgHeldoutReportingTests(unittest.TestCase):
    def _write_results_csv(self, path: Path, *, include_extra_method: bool = False) -> None:
        rows = [
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d1",
                "entity_type": "class",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"tfidf"}',
                "gold_count": 10,
                "target_pool_size": 100,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.50,
                "recall_at_1": 0.70,
                "recall_at_50": 0.95,
                "mrr": 0.90,
                "runtime_seconds": 0.10,
            },
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d1",
                "entity_type": "class",
                "method": "bm25",
                "hyperparameters": '{"cfg":"bm25"}',
                "gold_count": 10,
                "target_pool_size": 100,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.50,
                "recall_at_1": 0.60,
                "recall_at_50": 0.90,
                "mrr": 0.80,
                "runtime_seconds": 0.05,
            },
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d2",
                "entity_type": "class",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"tfidf"}',
                "gold_count": 1,
                "target_pool_size": 80,
                "retained_candidate_size": 40,
                "candidate_reduction_ratio": 0.50,
                "recall_at_1": 0.50,
                "recall_at_50": 0.80,
                "mrr": 0.70,
                "runtime_seconds": 0.11,
            },
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d2",
                "entity_type": "class",
                "method": "bm25",
                "hyperparameters": '{"cfg":"bm25"}',
                "gold_count": 1,
                "target_pool_size": 80,
                "retained_candidate_size": 40,
                "candidate_reduction_ratio": 0.50,
                "recall_at_1": 0.40,
                "recall_at_50": 0.70,
                "mrr": 0.60,
                "runtime_seconds": 0.06,
            },
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d1",
                "entity_type": "predicate",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"tfidf"}',
                "gold_count": 1,
                "target_pool_size": 20,
                "retained_candidate_size": 5,
                "candidate_reduction_ratio": 0.75,
                "recall_at_1": 0.20,
                "recall_at_50": 0.50,
                "mrr": 0.40,
                "runtime_seconds": 0.07,
            },
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d1",
                "entity_type": "predicate",
                "method": "bm25",
                "hyperparameters": '{"cfg":"bm25"}',
                "gold_count": 1,
                "target_pool_size": 20,
                "retained_candidate_size": 5,
                "candidate_reduction_ratio": 0.75,
                "recall_at_1": 0.30,
                "recall_at_50": 0.55,
                "mrr": 0.50,
                "runtime_seconds": 0.04,
            },
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d2",
                "entity_type": "predicate",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"tfidf"}',
                "gold_count": 1,
                "target_pool_size": 10,
                "retained_candidate_size": 4,
                "candidate_reduction_ratio": 0.60,
                "recall_at_1": 0.30,
                "recall_at_50": 0.70,
                "mrr": 0.60,
                "runtime_seconds": 0.08,
            },
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d2",
                "entity_type": "predicate",
                "method": "bm25",
                "hyperparameters": '{"cfg":"bm25"}',
                "gold_count": 1,
                "target_pool_size": 10,
                "retained_candidate_size": 4,
                "candidate_reduction_ratio": 0.60,
                "recall_at_1": 0.20,
                "recall_at_50": 0.65,
                "mrr": 0.55,
                "runtime_seconds": 0.03,
            },
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d1",
                "entity_type": "instance",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"tfidf"}',
                "gold_count": 1,
                "target_pool_size": 200,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.75,
                "recall_at_1": 0.10,
                "recall_at_50": 0.40,
                "mrr": 0.30,
                "runtime_seconds": 0.12,
            },
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d1",
                "entity_type": "instance",
                "method": "bm25",
                "hyperparameters": '{"cfg":"bm25"}',
                "gold_count": 1,
                "target_pool_size": 200,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 0.75,
                "recall_at_1": 0.15,
                "recall_at_50": 0.35,
                "mrr": 0.35,
                "runtime_seconds": 0.09,
            },
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d2",
                "entity_type": "instance",
                "method": "tfidf",
                "hyperparameters": '{"cfg":"tfidf"}',
                "gold_count": 20,
                "target_pool_size": 150,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 2.0 / 3.0,
                "recall_at_1": 0.40,
                "recall_at_50": 0.60,
                "mrr": 0.50,
                "runtime_seconds": 0.13,
            },
            {
                "track": "kg",
                "version": "heldout-v1",
                "dataset": "d2",
                "entity_type": "instance",
                "method": "bm25",
                "hyperparameters": '{"cfg":"bm25"}',
                "gold_count": 20,
                "target_pool_size": 150,
                "retained_candidate_size": 50,
                "candidate_reduction_ratio": 2.0 / 3.0,
                "recall_at_1": 0.45,
                "recall_at_50": 0.65,
                "mrr": 0.55,
                "runtime_seconds": 0.10,
            },
        ]

        if include_extra_method:
            extra_rows: list[dict[str, object]] = []
            for row in rows:
                if row["method"] != "tfidf":
                    continue

                exact_row = dict(row)
                exact_row.update(
                    {
                        "method": "exact_match",
                        "hyperparameters": '{"normalization":"light_v1"}',
                        "recall_at_1": max(0.0, float(row["recall_at_1"]) - 0.05),
                        "recall_at_50": max(0.0, float(row["recall_at_50"]) - 0.03),
                        "mrr": max(0.0, float(row["mrr"]) - 0.06),
                        "runtime_seconds": 0.08,
                    }
                )
                extra_rows.append(exact_row)

                char_ngram_row = dict(row)
                char_ngram_row.update(
                    {
                        "method": "char_ngram",
                        "hyperparameters": '{"normalization":"light_v1","analyzer":"char_wb","ngram_range":[3,5]}',
                        "recall_at_1": max(0.0, float(row["recall_at_1"]) - 0.02),
                        "recall_at_50": max(0.0, float(row["recall_at_50"]) - 0.01),
                        "mrr": max(0.0, float(row["mrr"]) - 0.03),
                        "runtime_seconds": 0.07,
                    }
                )
                extra_rows.append(char_ngram_row)

            rows.extend(extra_rows)

        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _write_selected_settings(self, path: Path) -> None:
        payload = {
            "heldout_datasets": ["d1", "d2"],
            "selected_settings": {
                "tfidf": {
                    "method": "tfidf",
                    "heldout_score": 0.90,
                    "mu": 0.88,
                    "sigma": 0.04,
                },
                "bm25": {
                    "method": "bm25",
                    "heldout_score": 0.75,
                    "mu": 0.78,
                    "sigma": 0.06,
                },
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def test_generates_expected_artifacts_and_stable_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "heldout_result_fixture.csv"
            selected_json = tmp / "comparisons" / "result_fixture" / "heldout_selected_settings.json"
            self._write_results_csv(results_csv)
            self._write_selected_settings(selected_json)

            artifacts = generate_kg_heldout_reporting(
                results_csv_path=results_csv,
                output_dir=tmp / "comparisons",
                selected_settings_path=selected_json,
            )
            second_artifacts = generate_kg_heldout_reporting(
                results_csv_path=results_csv,
                output_dir=tmp / "comparisons_second",
                selected_settings_path=selected_json,
            )

            expected_keys = {
                "source_csv",
                "output_dir",
                "kg_heldout_by_type_summary",
                "kg_heldout_macro_summary",
                "kg_heldout_micro_summary",
                "kg_heldout_reduction_effectiveness",
                "kg_heldout_pairwise_by_type_inference",
                "kg_heldout_pairwise_overall_inference",
                "kg_heldout_interpretation_scaffold",
                "kg_heldout_transfer_summary",
                "manifest",
            }
            self.assertEqual(set(artifacts.keys()), expected_keys)

            stable_keys = expected_keys - {"source_csv", "output_dir", "manifest"}
            for key in stable_keys:
                self.assertEqual(
                    Path(artifacts[key]).read_text(encoding="utf-8"),
                    Path(second_artifacts[key]).read_text(encoding="utf-8"),
                )

            manifest_text = Path(artifacts["manifest"]).read_text(encoding="utf-8")
            self.assertNotIn("generated_at=", manifest_text)

    def test_grouping_macro_micro_and_transfer_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "heldout_result_fixture.csv"
            selected_json = tmp / "heldout_selected_settings.json"
            self._write_results_csv(results_csv)
            self._write_selected_settings(selected_json)

            artifacts = generate_kg_heldout_reporting(
                results_csv_path=results_csv,
                output_dir=tmp / "comparisons",
                selected_settings_path=selected_json,
            )

            with Path(artifacts["kg_heldout_by_type_summary"]).open(encoding="utf-8") as csv_file:
                by_type_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(by_type_rows), 6)

            class_tfidf = next(
                row
                for row in by_type_rows
                if row["entity_type"] == "class" and row["method"] == "tfidf"
            )
            self.assertEqual(int(class_tfidf["dataset_count"]), 2)
            self.assertEqual(float(class_tfidf["gold_count_sum"]), 11.0)
            self.assertAlmostEqual(float(class_tfidf["mrr_mean"]), 0.8)
            self.assertAlmostEqual(float(class_tfidf["recall_at_50_mean"]), 0.875)

            with Path(artifacts["kg_heldout_macro_summary"]).open(encoding="utf-8") as csv_file:
                macro_rows = list(csv.DictReader(csv_file))
            macro_tfidf = next(row for row in macro_rows if row["method"] == "tfidf")
            macro_bm25 = next(row for row in macro_rows if row["method"] == "bm25")
            self.assertIn("dataset_count_macro_mean", macro_tfidf)
            self.assertIn("gold_count_sum_macro_mean", macro_tfidf)
            self.assertAlmostEqual(float(macro_tfidf["mrr_mean"]), (0.8 + 0.5 + 0.4) / 3.0)
            self.assertAlmostEqual(float(macro_bm25["mrr_mean"]), (0.7 + 0.525 + 0.45) / 3.0)

            with Path(artifacts["kg_heldout_micro_summary"]).open(encoding="utf-8") as csv_file:
                micro_rows = list(csv.DictReader(csv_file))
            micro_tfidf = next(row for row in micro_rows if row["method"] == "tfidf")
            micro_bm25 = next(row for row in micro_rows if row["method"] == "bm25")
            self.assertAlmostEqual(float(micro_tfidf["mrr"]), 21.0 / 34.0)
            self.assertAlmostEqual(float(micro_bm25["mrr"]), 21.0 / 34.0)

            with Path(artifacts["kg_heldout_transfer_summary"]).open(encoding="utf-8") as csv_file:
                transfer_rows = list(csv.DictReader(csv_file))
            transfer_tfidf = next(row for row in transfer_rows if row["method"] == "tfidf")
            self.assertAlmostEqual(float(transfer_tfidf["development_selection_score"]), 0.90)
            self.assertAlmostEqual(float(transfer_tfidf["development_mu"]), 0.88)
            self.assertAlmostEqual(float(transfer_tfidf["heldout_macro_mrr"]), (0.8 + 0.5 + 0.4) / 3.0)
            self.assertAlmostEqual(float(transfer_tfidf["heldout_micro_mrr"]), 21.0 / 34.0)

    def test_method_agnostic_summaries_allow_extra_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "heldout_result_fixture.csv"
            selected_json = tmp / "heldout_selected_settings.json"
            self._write_results_csv(results_csv, include_extra_method=True)
            self._write_selected_settings(selected_json)

            artifacts = generate_kg_heldout_reporting(
                results_csv_path=results_csv,
                output_dir=tmp / "comparisons",
                selected_settings_path=selected_json,
            )

            with Path(artifacts["kg_heldout_by_type_summary"]).open(encoding="utf-8") as csv_file:
                by_type_rows = list(csv.DictReader(csv_file))
            self.assertIn("exact_match", {row["method"] for row in by_type_rows})
            self.assertIn("char_ngram", {row["method"] for row in by_type_rows})

            with Path(artifacts["kg_heldout_macro_summary"]).open(encoding="utf-8") as csv_file:
                macro_rows = list(csv.DictReader(csv_file))
            self.assertIn("exact_match", {row["method"] for row in macro_rows})
            self.assertIn("char_ngram", {row["method"] for row in macro_rows})
            self.assertIn("delta_tfidf_minus_bm25", {row["method"] for row in macro_rows})

            with Path(artifacts["kg_heldout_micro_summary"]).open(encoding="utf-8") as csv_file:
                micro_rows = list(csv.DictReader(csv_file))
            self.assertIn("exact_match", {row["method"] for row in micro_rows})
            self.assertIn("char_ngram", {row["method"] for row in micro_rows})

            with Path(artifacts["kg_heldout_reduction_effectiveness"]).open(
                encoding="utf-8"
            ) as csv_file:
                reduction_rows = list(csv.DictReader(csv_file))
            char_ngram_row = next(row for row in reduction_rows if row["method"] == "char_ngram")
            self.assertEqual(char_ngram_row["other_method"], "")
            self.assertEqual(char_ngram_row["mrr_delta_vs_other_method"], "")

    def test_reduction_effectiveness_includes_per_pair_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "heldout_result_fixture.csv"
            self._write_results_csv(results_csv)

            artifacts = generate_kg_heldout_reporting(
                results_csv_path=results_csv,
                output_dir=tmp / "comparisons",
            )

            with Path(artifacts["kg_heldout_reduction_effectiveness"]).open(
                encoding="utf-8"
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))

            tfidf_class = next(
                row
                for row in rows
                if row["dataset"] == "d1" and row["entity_type"] == "class" and row["method"] == "tfidf"
            )
            self.assertEqual(tfidf_class["other_method"], "bm25")
            self.assertAlmostEqual(float(tfidf_class["mrr_delta_vs_other_method"]), 0.10)
            self.assertAlmostEqual(float(tfidf_class["recall_at_50_delta_vs_other_method"]), 0.05)
            self.assertAlmostEqual(
                float(tfidf_class["candidate_reduction_ratio_delta_vs_other_method"]), 0.0
            )

    def test_pairwise_inference_artifacts_include_expected_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "heldout_result_fixture.csv"
            self._write_results_csv(results_csv, include_extra_method=True)

            artifacts = generate_kg_heldout_reporting(
                results_csv_path=results_csv,
                output_dir=tmp / "comparisons",
            )

            with Path(artifacts["kg_heldout_pairwise_by_type_inference"]).open(
                encoding="utf-8"
            ) as csv_file:
                by_type_rows = list(csv.DictReader(csv_file))
            class_tfidf_vs_bm25 = next(
                row
                for row in by_type_rows
                if row["entity_type"] == "class"
                and row["metric"] == "mrr"
                and row["method_a"] == "tfidf"
                and row["method_b"] == "bm25"
            )
            self.assertAlmostEqual(float(class_tfidf_vs_bm25["paired_delta"]), 0.10)
            self.assertAlmostEqual(float(class_tfidf_vs_bm25["method_a_mean"]), 0.80)
            self.assertAlmostEqual(float(class_tfidf_vs_bm25["method_b_mean"]), 0.70)
            self.assertAlmostEqual(float(class_tfidf_vs_bm25["ci_lower"]), 0.10)
            self.assertAlmostEqual(float(class_tfidf_vs_bm25["ci_upper"]), 0.10)
            self.assertAlmostEqual(float(class_tfidf_vs_bm25["p_value"]), 0.50)

            metrics = {row["metric"] for row in by_type_rows}
            self.assertEqual(metrics, {"mrr", "recall_at_1", "recall_at_50"})

            with Path(artifacts["kg_heldout_pairwise_overall_inference"]).open(
                encoding="utf-8"
            ) as csv_file:
                overall_rows = list(csv.DictReader(csv_file))
            self.assertIn(
                "overall_dataset_type_rows",
                {row["aggregation_scope"] for row in overall_rows},
            )
            self.assertIn(
                "macro_entity_type_means",
                {row["aggregation_scope"] for row in overall_rows},
            )
            self.assertNotIn(
                "candidate_reduction_ratio",
                {row["metric"] for row in overall_rows},
            )

    def test_inference_helpers_are_deterministic_and_exact(self) -> None:
        ci_first = paired_bootstrap_confidence_interval([0.1, 0.1])
        ci_second = paired_bootstrap_confidence_interval([0.1, 0.1])
        self.assertEqual(ci_first, ci_second)
        self.assertAlmostEqual(ci_first[0], 0.1)
        self.assertAlmostEqual(ci_first[1], 0.1)
        self.assertAlmostEqual(exact_paired_sign_flip_p_value([0.1, 0.1]), 0.5)
        self.assertAlmostEqual(exact_paired_sign_flip_p_value([0.0, 0.0]), 1.0)

    def test_sign_flip_monte_carlo_fallback_is_deterministic(self) -> None:
        deltas = [0.5, -0.1, 0.3, -0.2, 0.4, -0.05]
        p_first = paired_sign_flip_p_value(
            deltas,
            exact_max_nonzero=2,
            monte_carlo_draws=50_000,
            seed=1729,
        )
        p_second = paired_sign_flip_p_value(
            deltas,
            exact_max_nonzero=2,
            monte_carlo_draws=50_000,
            seed=1729,
        )
        self.assertAlmostEqual(p_first, p_second)
        self.assertGreaterEqual(p_first, 0.0)
        self.assertLessEqual(p_first, 1.0)

    def test_single_method_reporting_keeps_pairwise_artifacts_empty_but_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "heldout_result_fixture.csv"
            self._write_results_csv(results_csv)

            with results_csv.open(encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            tfidf_only = [row for row in rows if row["method"] == "tfidf"]
            with results_csv.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(tfidf_only[0].keys()))
                writer.writeheader()
                writer.writerows(tfidf_only)

            artifacts = generate_kg_heldout_reporting(
                results_csv_path=results_csv,
                output_dir=tmp / "comparisons",
            )

            pairwise_by_type_path = Path(artifacts["kg_heldout_pairwise_by_type_inference"])
            with pairwise_by_type_path.open(encoding="utf-8") as csv_file:
                by_type_headers = next(csv.reader(csv_file))
            with pairwise_by_type_path.open(encoding="utf-8") as csv_file:
                by_type_rows = list(csv.DictReader(csv_file))
            self.assertEqual(by_type_rows, [])
            self.assertEqual(
                by_type_headers,
                [
                    "entity_type",
                    "metric",
                    "method_a",
                    "method_b",
                    "paired_unit_count",
                    "nonzero_delta_count",
                    "method_a_mean",
                    "method_b_mean",
                    "paired_delta",
                    "ci_lower",
                    "ci_upper",
                    "p_value",
                ],
            )

            pairwise_overall_path = Path(artifacts["kg_heldout_pairwise_overall_inference"])
            with pairwise_overall_path.open(encoding="utf-8") as csv_file:
                overall_headers = next(csv.reader(csv_file))
            with pairwise_overall_path.open(encoding="utf-8") as csv_file:
                overall_rows = list(csv.DictReader(csv_file))
            self.assertEqual(overall_rows, [])
            self.assertEqual(
                overall_headers,
                [
                    "aggregation_scope",
                    "metric",
                    "method_a",
                    "method_b",
                    "paired_unit_count",
                    "nonzero_delta_count",
                    "method_a_mean",
                    "method_b_mean",
                    "paired_delta",
                    "ci_lower",
                    "ci_upper",
                    "p_value",
                ],
            )

    def test_skips_transfer_summary_when_auto_detection_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "results" / "heldout_result_fixture.csv"
            results_csv.parent.mkdir(parents=True, exist_ok=True)
            self._write_results_csv(results_csv)

            old_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                artifacts = generate_kg_heldout_reporting(
                    results_csv_path=None,
                    output_dir=tmp / "comparisons",
                    selected_settings_path=None,
                )
            finally:
                os.chdir(old_cwd)

            self.assertNotIn("kg_heldout_transfer_summary", artifacts)
            interpretation = Path(artifacts["kg_heldout_interpretation_scaffold"]).read_text(
                encoding="utf-8"
            )
            self.assertIn("transfer summary skipped", interpretation.lower())

    def test_auto_detection_uses_matching_selected_settings_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "results" / "heldout_result_fixture.csv"
            results_csv.parent.mkdir(parents=True, exist_ok=True)
            self._write_results_csv(results_csv)

            matching = tmp / "results" / "comparisons" / "matching" / "heldout_selected_settings.json"
            mismatched = tmp / "results" / "comparisons" / "mismatched" / "heldout_selected_settings.json"
            self._write_selected_settings(matching)
            mismatched_payload = {
                "heldout_datasets": ["other_dataset"],
                "selected_settings": {
                    "tfidf": {"heldout_score": 0.1, "mu": 0.1, "sigma": 0.1},
                    "bm25": {"heldout_score": 0.1, "mu": 0.1, "sigma": 0.1},
                },
            }
            mismatched.parent.mkdir(parents=True, exist_ok=True)
            mismatched.write_text(
                json.dumps(mismatched_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            old_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                artifacts = generate_kg_heldout_reporting(
                    results_csv_path=None,
                    output_dir=tmp / "comparisons",
                    selected_settings_path=None,
                )
            finally:
                os.chdir(old_cwd)

            self.assertIn("kg_heldout_transfer_summary", artifacts)

    def test_explicit_malformed_selected_settings_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "heldout_result_fixture.csv"
            selected_json = tmp / "heldout_selected_settings.json"
            self._write_results_csv(results_csv)
            selected_json.write_text("{not-json}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                KgHeldoutReportingValidationError,
                "Malformed selected settings JSON",
            ):
                generate_kg_heldout_reporting(
                    results_csv_path=results_csv,
                    output_dir=tmp / "comparisons",
                    selected_settings_path=selected_json,
                )

    def test_missing_entity_type_or_method_rows_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bad_csv = tmp / "bad.csv"
            self._write_results_csv(bad_csv)
            with bad_csv.open(encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            filtered_rows = [row for row in rows if row["entity_type"] != "predicate"]

            with bad_csv.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(filtered_rows[0].keys()))
                writer.writeheader()
                writer.writerows(filtered_rows)

            with self.assertRaisesRegex(
                KgHeldoutReportingValidationError,
                "requires exactly these entity types",
            ):
                generate_kg_heldout_reporting(
                    results_csv_path=bad_csv,
                    output_dir=tmp / "comparisons",
                )
