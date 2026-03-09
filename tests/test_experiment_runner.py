import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.experiments.experiment_runner as experiment_runner_module
from src.experiments.experiment_runner import run_experiments
from src.retrieval.tfidf_retriever import TfidfRetriever


def _source_rdf() -> str:
    return """<?xml version="1.0"?>
<rdf:RDF
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
  xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://example.org/source#SClass1">
    <rdfs:label>PlantHeight</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/source#SClass2">
    <rdfs:label>LeafColor</rdfs:label>
  </owl:Class>
  <rdf:Description rdf:about="http://example.org/source#SNonClass">
    <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#NamedIndividual"/>
    <rdfs:label>IgnoreMe</rdfs:label>
  </rdf:Description>
</rdf:RDF>
"""


def _target_rdf() -> str:
    return """<?xml version="1.0"?>
<rdf:RDF
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
  xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://example.org/target#TClass1">
    <rdfs:label>plant height</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/target#TClass2">
    <rdfs:label>leaf color</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://example.org/target#TClass3">
    <rdfs:label>root length</rdfs:label>
  </owl:Class>
  <rdf:Description rdf:about="http://example.org/target#TNonClass">
    <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#NamedIndividual"/>
    <rdfs:label>IgnoreMeToo</rdfs:label>
  </rdf:Description>
</rdf:RDF>
"""


def _alignment_rdf() -> str:
    return """<?xml version="1.0"?>
<rdf:RDF
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#">
  <Alignment>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#SClass1"/>
        <entity2 rdf:resource="http://example.org/target#TClass1"/>
        <relation>=</relation>
        <measure>1.0</measure>
      </Cell>
    </map>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#SClass2"/>
        <entity2 rdf:resource="http://example.org/target#TClass2"/>
        <relation>=</relation>
        <measure>1.0</measure>
      </Cell>
    </map>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#SNonClass"/>
        <entity2 rdf:resource="http://example.org/target#TNonClass"/>
        <relation>=</relation>
        <measure>1.0</measure>
      </Cell>
    </map>
  </Alignment>
</rdf:RDF>
"""


class ExperimentRunnerTests(unittest.TestCase):
    def _write_fixture_dataset(self, tmp: Path) -> Path:
        source_path = tmp / "source.rdf"
        target_path = tmp / "target.rdf"
        alignment_path = tmp / "alignment.rdf"
        source_path.write_text(_source_rdf(), encoding="utf-8")
        target_path.write_text(_target_rdf(), encoding="utf-8")
        alignment_path.write_text(_alignment_rdf(), encoding="utf-8")

        config_path = tmp / "datasets.yaml"
        config_path.write_text(
            f"""
datasets:
  fixture_dataset:
    track: fixture_track
    version: "v1"
    source_rdf: {source_path}
    target_rdf: {target_path}
    alignment_rdf: {alignment_path}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        return config_path

    def test_runner_executes_and_returns_expected_shape_and_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            output_csv_path = Path(tmpdir) / "results" / "experiment_results.csv"
            results = run_experiments(
                config_path=config_path, output_csv_path=output_csv_path
            )
            self.assertTrue(output_csv_path.exists())

        self.assertTrue(results)
        self.assertEqual(len(results), 4)  # 2 TF-IDF configs + 2 BM25 configs

        models = {row["model"] for row in results}
        self.assertEqual(models, {"tfidf", "bm25"})

        for row in results:
            for key in (
                "dataset_name",
                "track",
                "version",
                "model",
                "hyperparameters",
                "num_source_entities",
                "num_target_entities",
                "num_gold_pairs",
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "recall_at_20",
                "recall_at_50",
                "mrr",
                "runtime_seconds",
            ):
                self.assertIn(key, row)

            self.assertEqual(row["num_source_entities"], 2)  # owl:Class only
            self.assertEqual(row["num_target_entities"], 3)  # owl:Class only
            self.assertEqual(row["num_gold_pairs"], 2)  # non-class mapping filtered out

            for metric_key in (
                "recall_at_1",
                "recall_at_5",
                "recall_at_10",
                "recall_at_20",
                "recall_at_50",
                "mrr",
            ):
                metric_value = row[metric_key]
                self.assertIsInstance(metric_value, float)
                self.assertGreaterEqual(metric_value, 0.0)
                self.assertLessEqual(metric_value, 1.0)
            self.assertGreaterEqual(row["runtime_seconds"], 0.0)

    def test_csv_schema_and_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            output_csv_path = Path(tmpdir) / "results" / "experiment_results.csv"
            results = run_experiments(
                config_path=config_path, output_csv_path=output_csv_path
            )

            with output_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                self.assertEqual(
                    reader.fieldnames,
                    [
                        "track",
                        "version",
                        "dataset",
                        "method",
                        "hyperparameters",
                        "candidate_size",
                        "recall_at_1",
                        "recall_at_5",
                        "recall_at_10",
                        "recall_at_20",
                        "recall_at_50",
                        "mrr",
                        "runtime_seconds",
                    ],
                )
                rows = list(reader)

        self.assertEqual(len(rows), len(results))
        for row in rows:
            self.assertEqual(row["dataset"], "fixture_dataset")
            self.assertIn(row["method"], {"tfidf", "bm25"})
            self.assertEqual(row["candidate_size"], "50")
            json.loads(row["hyperparameters"])
            runtime_seconds = float(row["runtime_seconds"])
            self.assertGreaterEqual(runtime_seconds, 0.0)

    def test_best_effort_writes_successful_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            output_csv_path = Path(tmpdir) / "results" / "experiment_results.csv"
            with patch.object(
                TfidfRetriever, "retrieve", side_effect=RuntimeError("forced tfidf failure")
            ):
                results = run_experiments(
                    config_path=config_path, output_csv_path=output_csv_path
                )

            with output_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(results), 2)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["method"] == "bm25" for row in rows))

    def test_csv_persistence_failure_does_not_abort_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            with patch.object(
                experiment_runner_module,
                "_persist_results_to_csv",
                side_effect=OSError("forced csv write failure"),
            ):
                results = run_experiments(config_path=config_path)

        self.assertEqual(len(results), 4)

    def test_runner_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            output_csv_path = Path(tmpdir) / "results" / "experiment_results.csv"
            first = run_experiments(config_path=config_path, output_csv_path=output_csv_path)
            second = run_experiments(config_path=config_path, output_csv_path=output_csv_path)

        first_without_runtime = [
            {key: value for key, value in row.items() if key != "runtime_seconds"}
            for row in first
        ]
        second_without_runtime = [
            {key: value for key, value in row.items() if key != "runtime_seconds"}
            for row in second
        ]
        self.assertEqual(first_without_runtime, second_without_runtime)

    def test_runner_emits_run_summary_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            output_csv_path = Path(tmpdir) / "results" / "experiment_results.csv"
            with self.assertLogs("src.experiments.experiment_runner", level="INFO") as captured:
                run_experiments(config_path=config_path, output_csv_path=output_csv_path)

        combined = "\n".join(captured.output)
        self.assertIn("Processing dataset 'fixture_dataset'", combined)
        self.assertIn("Finished dataset 'fixture_dataset' model runs", combined)
        self.assertIn("Run summary:", combined)
        self.assertIn("successful_model_runs=4", combined)
        self.assertIn("failed_model_runs=0", combined)

    def test_runner_failure_logs_include_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            output_csv_path = Path(tmpdir) / "results" / "experiment_results.csv"
            with self.assertLogs("src.experiments.experiment_runner", level="INFO") as captured:
                with patch.object(
                    TfidfRetriever, "retrieve", side_effect=RuntimeError("forced tfidf failure")
                ):
                    run_experiments(config_path=config_path, output_csv_path=output_csv_path)

        combined = "\n".join(captured.output)
        self.assertIn("Model run failed for dataset='fixture_dataset'", combined)
        self.assertIn("Run summary:", combined)
        self.assertIn("failed_model_runs=2", combined)

    def test_runner_with_show_progress_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            output_csv_path = Path(tmpdir) / "results" / "experiment_results.csv"
            results = run_experiments(
                config_path=config_path,
                output_csv_path=output_csv_path,
                show_progress=False,
            )

        self.assertEqual(len(results), 4)

    def test_runner_with_show_progress_true_uses_tqdm(self) -> None:
        call_count = 0

        def _fake_tqdm(iterable, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return iterable

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            output_csv_path = Path(tmpdir) / "results" / "experiment_results.csv"
            with patch("src.experiments.experiment_runner.tqdm", side_effect=_fake_tqdm):
                results = run_experiments(
                    config_path=config_path,
                    output_csv_path=output_csv_path,
                    show_progress=True,
                )

        self.assertEqual(len(results), 4)
        self.assertGreater(call_count, 0)


if __name__ == "__main__":
    unittest.main()
