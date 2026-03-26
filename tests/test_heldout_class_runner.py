import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.heldout_class_runner import (
    HeldoutClassRunnerValidationError,
    run_heldout_class_experiments,
)


def _source_rdf() -> str:
    return """<?xml version="1.0"?>
<rdf:RDF
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
  xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://example.org/source#Disease">
    <rdfs:label>DiseaseEntity</rdfs:label>
  </owl:Class>
</rdf:RDF>
"""


def _target_rdf() -> str:
    return """<?xml version="1.0"?>
<rdf:RDF
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
  xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://example.org/target#Disease">
    <rdfs:label>disease entity</rdfs:label>
  </owl:Class>
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
        <entity1 rdf:resource="http://example.org/source#Disease"/>
        <entity2 rdf:resource="http://example.org/target#Disease"/>
        <relation>=</relation>
        <measure>1.0</measure>
      </Cell>
    </map>
  </Alignment>
</rdf:RDF>
"""


class HeldoutClassRunnerTests(unittest.TestCase):
    def _write_config(
        self,
        tmp: Path,
        *,
        target_rdf_text: str | None = None,
    ) -> Path:
        source = tmp / "source.rdf"
        target = tmp / "target.rdf"
        alignment = tmp / "alignment.rdf"
        source.write_text(_source_rdf(), encoding="utf-8")
        target.write_text(target_rdf_text or _target_rdf(), encoding="utf-8")
        alignment.write_text(_alignment_rdf(), encoding="utf-8")

        config_path = tmp / "runtime.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "experiments:",
                    "  evaluation_ks: [1, 5]",
                    "  tfidf_grid:",
                    "    - ngram_range: [1, 1]",
                    "      min_df: 1",
                    "      max_df: 1.0",
                    "      sublinear_tf: false",
                    "  bm25_grid:",
                    "    - k1: 1.5",
                    "      b: 0.75",
                    "development_datasets:",
                    "  dev_fixture:",
                    "    track: conf",
                    '    version: "v1"',
                    f"    source_rdf: {source}",
                    f"    target_rdf: {target}",
                    f"    alignment_rdf: {alignment}",
                    "heldout_datasets:",
                    "  kg_fixture:",
                    "    track: kg",
                    '    version: "heldout-v1"',
                    f"    source_rdf: {source}",
                    f"    target_rdf: {target}",
                    f"    alignment_rdf: {alignment}",
                    "heldout_secondary_datasets:",
                    "  class_fixture:",
                    "    track: bioml-supervised",
                    '    version: "2022"',
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
        return config_path

    def _write_selected_settings(self, path: Path) -> None:
        payload = {
            "selected_settings": {
                "tfidf": {
                    "method": "tfidf",
                    "hyperparameters": {
                        "ngram_range": [1, 1],
                        "min_df": 1,
                        "max_df": 1.0,
                        "sublinear_tf": False,
                    },
                },
                "bm25": {
                    "method": "bm25",
                    "hyperparameters": {"k1": 1.7, "b": 0.75},
                },
            }
        }
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    def test_runner_generates_class_only_rows_for_all_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp)
            selected = tmp / "heldout_selected_settings.json"
            output_csv = tmp / "results" / "heldout_class.csv"
            self._write_selected_settings(selected)

            results = run_heldout_class_experiments(
                config_path=config,
                selected_settings_path=selected,
                output_csv_path=output_csv,
                show_progress=False,
            )

            self.assertEqual(len(results), 4)
            self.assertEqual({row["entity_type"] for row in results}, {"class"})
            self.assertEqual(
                {row["model"] for row in results},
                {"tfidf", "bm25", "exact_match", "char_ngram"},
            )
            with output_csv.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["entity_type"] for row in rows}, {"class"})

    def test_runner_uses_selected_settings_only_for_tunable_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp)
            selected = tmp / "heldout_selected_settings.json"
            self._write_selected_settings(selected)

            results = run_heldout_class_experiments(
                config_path=config,
                selected_settings_path=selected,
                show_progress=False,
            )

        tfidf_row = next(row for row in results if row["model"] == "tfidf")
        bm25_row = next(row for row in results if row["model"] == "bm25")
        exact_row = next(row for row in results if row["model"] == "exact_match")
        self.assertEqual(tfidf_row["hyperparameters"]["ngram_range"], [1, 1])
        self.assertEqual(bm25_row["hyperparameters"], {"k1": 1.7, "b": 0.75})
        self.assertEqual(exact_row["hyperparameters"], {"normalization": "light_v1"})

    def test_runner_fails_when_class_targets_are_missing(self) -> None:
        empty_target = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" />
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp, target_rdf_text=empty_target)
            selected = tmp / "heldout_selected_settings.json"
            self._write_selected_settings(selected)
            with self.assertRaises(HeldoutClassRunnerValidationError):
                run_heldout_class_experiments(
                    config_path=config,
                    selected_settings_path=selected,
                    show_progress=False,
                )


if __name__ == "__main__":
    unittest.main()
