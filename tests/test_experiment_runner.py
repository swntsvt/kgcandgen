import tempfile
import unittest
from pathlib import Path

from src.experiments.experiment_runner import run_experiments


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

    def test_runner_executes_and_returns_expected_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            results = run_experiments(config_path=config_path)

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

    def test_runner_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_fixture_dataset(Path(tmpdir))
            first = run_experiments(config_path=config_path)
            second = run_experiments(config_path=config_path)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
