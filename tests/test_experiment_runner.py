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


def _default_experiments_yaml() -> str:
    return """
experiments:
  evaluation_ks: [1, 5, 10, 20, 50]
  tfidf_grid:
    - ngram_range: [1, 1]
      min_df: 1
      max_df: 1.0
      sublinear_tf: false
    - ngram_range: [1, 2]
      min_df: 1
      max_df: 1.0
      sublinear_tf: true
  bm25_grid:
    - k1: 1.5
      b: 0.75
    - k1: 1.2
      b: 0.75
"""


class ExperimentRunnerTests(unittest.TestCase):
    def _write_fixture_dataset(
        self,
        tmp: Path,
        experiments_yaml: str | None = None,
        include_experiments: bool = True,
    ) -> Path:
        source_path = tmp / "source.rdf"
        target_path = tmp / "target.rdf"
        alignment_path = tmp / "alignment.rdf"
        source_path.write_text(_source_rdf(), encoding="utf-8")
        target_path.write_text(_target_rdf(), encoding="utf-8")
        alignment_path.write_text(_alignment_rdf(), encoding="utf-8")

        config_body = (
            f"""
datasets:
  fixture_dataset:
    track: fixture_track
    version: "v1"
    source_rdf: {source_path}
    target_rdf: {target_path}
    alignment_rdf: {alignment_path}
""".strip()
            + "\n"
        )

        if include_experiments:
            config_body += "\n" + (experiments_yaml or _default_experiments_yaml()).strip() + "\n"

        config_path = tmp / "datasets.yaml"
        config_path.write_text(config_body, encoding="utf-8")
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
                "gold_count",
                "candidate_size",
                "dataset_prep_seconds",
                "recalls",
                "mrr",
                "runtime_seconds",
            ):
                self.assertIn(key, row)

            self.assertEqual(row["num_source_entities"], 2)  # owl:Class only
            self.assertEqual(row["num_target_entities"], 3)  # owl:Class only
            self.assertEqual(row["num_gold_pairs"], 2)  # non-class mapping filtered out
            self.assertEqual(row["gold_count"], 2)
            self.assertEqual(row["candidate_size"], 3)
            self.assertGreaterEqual(row["dataset_prep_seconds"], 0.0)

            recalls = row["recalls"]
            self.assertEqual(list(recalls.keys()), [1, 5, 10, 20, 50])
            for recall_value in recalls.values():
                self.assertIsInstance(recall_value, float)
                self.assertGreaterEqual(recall_value, 0.0)
                self.assertLessEqual(recall_value, 1.0)

            self.assertIsInstance(row["mrr"], float)
            self.assertGreaterEqual(row["mrr"], 0.0)
            self.assertLessEqual(row["mrr"], 1.0)
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
                        "gold_count",
                        "candidate_size",
                        "dataset_prep_seconds",
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
            self.assertEqual(row["gold_count"], "2")
            self.assertEqual(row["candidate_size"], "3")
            json.loads(row["hyperparameters"])
            dataset_prep_seconds = float(row["dataset_prep_seconds"])
            self.assertGreaterEqual(dataset_prep_seconds, 0.0)
            runtime_seconds = float(row["runtime_seconds"])
            self.assertGreaterEqual(runtime_seconds, 0.0)

    def test_runner_uses_yaml_defined_ks_and_grid_order(self) -> None:
        custom_experiments = """
experiments:
  evaluation_ks: [2, 4]
  tfidf_grid:
    - ngram_range: [1, 1]
      min_df: 1
      max_df: 1.0
      sublinear_tf: false
  bm25_grid:
    - k1: 1.5
      b: 0.75
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(
                Path(tmpdir), experiments_yaml=custom_experiments
            )
            output_csv_path = Path(tmpdir) / "results" / "experiment_results.csv"
            results = run_experiments(
                config_path=config_path, output_csv_path=output_csv_path
            )

            with output_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                fieldnames = reader.fieldnames
                rows = list(reader)

        self.assertEqual(len(results), 2)
        self.assertEqual([row["model"] for row in results], ["tfidf", "bm25"])
        for row in results:
            self.assertEqual(list(row["recalls"].keys()), [2, 4])

        self.assertEqual(
            fieldnames,
            [
                "track",
                "version",
                "dataset",
                "method",
                "hyperparameters",
                "gold_count",
                "candidate_size",
                "dataset_prep_seconds",
                "recall_at_2",
                "recall_at_4",
                "mrr",
                "runtime_seconds",
            ],
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["gold_count"] == "2" for row in rows))
        self.assertTrue(all(row["candidate_size"] == "3" for row in rows))

    def test_runner_fails_fast_for_invalid_experiment_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(
                Path(tmpdir), include_experiments=False
            )
            with patch.object(experiment_runner_module, "_load_store_with_fallback") as loader:
                with self.assertRaisesRegex(ValueError, "'experiments' mapping"):
                    run_experiments(config_path=config_path)
                loader.assert_not_called()

    def test_runner_preserves_yaml_dataset_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
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
  z_dataset:
    track: fixture_track
    version: "v1"
    source_rdf: {source_path}
    target_rdf: {target_path}
    alignment_rdf: {alignment_path}
  a_dataset:
    track: fixture_track
    version: "v1"
    source_rdf: {source_path}
    target_rdf: {target_path}
    alignment_rdf: {alignment_path}
experiments:
  evaluation_ks: [1]
  tfidf_grid:
    - ngram_range: [1, 1]
      min_df: 1
      max_df: 1.0
      sublinear_tf: false
  bm25_grid:
    - k1: 1.5
      b: 0.75
""".strip()
                + "\n",
                encoding="utf-8",
            )
            output_csv_path = tmp / "results" / "experiment_results.csv"
            results = run_experiments(
                config_path=config_path, output_csv_path=output_csv_path
            )

        self.assertEqual(
            [row["dataset_name"] for row in results],
            ["z_dataset", "z_dataset", "a_dataset", "a_dataset"],
        )

    def test_shared_preprocessing_runs_once_per_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            output_csv_path = Path(tmpdir) / "results" / "experiment_results.csv"
            with patch(
                "src.experiments.experiment_runner.preprocess_text",
                wraps=experiment_runner_module.preprocess_text,
            ) as preprocess_mock:
                run_experiments(config_path=config_path, output_csv_path=output_csv_path)

        # 3 target labels + 2 evaluated source labels; should not scale with model count.
        self.assertEqual(preprocess_mock.call_count, 5)

    def test_best_effort_writes_successful_rows_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            output_csv_path = Path(tmpdir) / "results" / "experiment_results.csv"
            with patch.object(
                TfidfRetriever,
                "retrieve_preprocessed",
                side_effect=RuntimeError("forced tfidf failure"),
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
            {
                key: value
                for key, value in row.items()
                if key not in {"runtime_seconds", "dataset_prep_seconds"}
            }
            for row in first
        ]
        second_without_runtime = [
            {
                key: value
                for key, value in row.items()
                if key not in {"runtime_seconds", "dataset_prep_seconds"}
            }
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
                    TfidfRetriever,
                    "retrieve_preprocessed",
                    side_effect=RuntimeError("forced tfidf failure"),
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
