import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from src.experiments.heldout_kg_runner import (
    HeldoutKgRunnerValidationError,
    run_heldout_kg_experiments,
)


def _heldout_source_rdf() -> str:
    return """<?xml version="1.0"?>
<rdf:RDF
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
  xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://dbkwik.webdatacommons.org/source/class/Hero">
    <rdfs:label>Hero</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://dbkwik.webdatacommons.org/source/class/Villain">
    <rdfs:label>Villain</rdfs:label>
  </owl:Class>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/source/resource/Thor">
    <rdfs:label>Thor</rdfs:label>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/source/resource/Loki">
    <rdfs:label>Loki</rdfs:label>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/source/resource/Asgard">
    <rdfs:label>Asgard</rdfs:label>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/source/resource/Mjolnir">
    <rdfs:label>Mjolnir</rdfs:label>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/source/resource/Film">
    <rdfs:label>Film</rdfs:label>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/source/resource/Asgard">
    <ns1:memberOf xmlns:ns1="http://dbkwik.webdatacommons.org/source/property/" rdf:resource="http://dbkwik.webdatacommons.org/source/resource/Heroes"/>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/source/resource/Thor">
    <ns1:appearsIn xmlns:ns1="http://dbkwik.webdatacommons.org/source/property/" rdf:resource="http://dbkwik.webdatacommons.org/source/resource/Film"/>
  </rdf:Description>
</rdf:RDF>
"""


def _heldout_target_rdf() -> str:
    return """<?xml version="1.0"?>
<rdf:RDF
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
  xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://dbkwik.webdatacommons.org/target/class/Hero">
    <rdfs:label>Hero</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://dbkwik.webdatacommons.org/target/class/Villain">
    <rdfs:label>Villain</rdfs:label>
  </owl:Class>
  <owl:Class rdf:about="http://dbkwik.webdatacommons.org/target/class/Location">
    <rdfs:label>Location</rdfs:label>
  </owl:Class>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/target/resource/Thor">
    <rdfs:label>Thor</rdfs:label>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/target/resource/Loki">
    <rdfs:label>Loki</rdfs:label>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/target/resource/Film">
    <rdfs:label>Film</rdfs:label>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/target/resource/Mjolnir">
    <rdfs:label>Mjolnir</rdfs:label>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/target/resource/Asgard">
    <rdfs:label>Asgard</rdfs:label>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/target/resource/Thor">
    <ns1:appearsIn xmlns:ns1="http://dbkwik.webdatacommons.org/target/property/" rdf:resource="http://dbkwik.webdatacommons.org/target/resource/Film"/>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/target/resource/Asgard">
    <ns1:memberOf xmlns:ns1="http://dbkwik.webdatacommons.org/target/property/" rdf:resource="http://dbkwik.webdatacommons.org/target/resource/Heroes"/>
  </rdf:Description>
</rdf:RDF>
"""


def _heldout_alignment_rdf(*, include_predicate: bool = True) -> str:
    predicate_map = """
    <map>
      <Cell>
        <entity1 rdf:resource="http://dbkwik.webdatacommons.org/source/property/appearsIn"/>
        <entity2 rdf:resource="http://dbkwik.webdatacommons.org/target/property/appearsIn"/>
        <relation>=</relation>
        <measure>1.0</measure>
      </Cell>
    </map>""" if include_predicate else ""
    return f"""<?xml version="1.0"?>
<rdf:RDF
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#">
  <Alignment>
    <map>
      <Cell>
        <entity1 rdf:resource="http://dbkwik.webdatacommons.org/source/class/Hero"/>
        <entity2 rdf:resource="http://dbkwik.webdatacommons.org/target/class/Hero"/>
        <relation>=</relation>
        <measure>1.0</measure>
      </Cell>
    </map>
    {predicate_map}
    <map>
      <Cell>
        <entity1 rdf:resource="http://dbkwik.webdatacommons.org/source/resource/Thor"/>
        <entity2 rdf:resource="http://dbkwik.webdatacommons.org/target/resource/Thor"/>
        <relation>=</relation>
        <measure>1.0</measure>
      </Cell>
    </map>
  </Alignment>
</rdf:RDF>
"""


class HeldoutKgRunnerTests(unittest.TestCase):
    def _write_config(
        self,
        tmp: Path,
        *,
        target_rdf_text: str | None = None,
        alignment_rdf_text: str | None = None,
    ) -> Path:
        source = tmp / "source.rdf"
        target = tmp / "target.rdf"
        alignment = tmp / "alignment.rdf"
        source.write_text(_heldout_source_rdf(), encoding="utf-8")
        target.write_text(target_rdf_text or _heldout_target_rdf(), encoding="utf-8")
        alignment.write_text(
            alignment_rdf_text or _heldout_alignment_rdf(),
            encoding="utf-8",
        )

        config_path = tmp / "runtime.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "experiments:",
                    "  evaluation_ks: [1, 2]",
                    "  tfidf_grid:",
                    "    - ngram_range: [2, 2]",
                    "      min_df: 3",
                    "      max_df: 0.9",
                    "      sublinear_tf: true",
                    "  bm25_grid:",
                    "    - k1: 0.1",
                    "      b: 0.1",
                    "development_datasets:",
                    "  dev_fixture:",
                    "    track: conference",
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

    def _write_selected_settings(
        self,
        path: Path,
        *,
        include_bm25: bool = True,
    ) -> None:
        selected_settings = {
            "tfidf": {
                "method": "tfidf",
                "hyperparameters": {
                    "ngram_range": [1, 1],
                    "min_df": 1,
                    "max_df": 1.0,
                    "sublinear_tf": False,
                },
            }
        }
        if include_bm25:
            selected_settings["bm25"] = {
                "method": "bm25",
                "hyperparameters": {
                    "k1": 1.8,
                    "b": 0.75,
                },
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "selected_settings": selected_settings,
                    "heldout_datasets": ["kg_fixture"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def test_runner_generates_per_type_rows_and_expected_csv_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp)
            selected = tmp / "comparisons" / "result_fixture" / "heldout_selected_settings.json"
            output_csv = tmp / "results" / "heldout.csv"
            self._write_selected_settings(selected)

            results = run_heldout_kg_experiments(
                config_path=config,
                selected_settings_path=selected,
                output_csv_path=output_csv,
                show_progress=False,
            )

            self.assertEqual(len(results), 12)
            self.assertTrue(output_csv.exists())
            with output_csv.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                self.assertEqual(
                    reader.fieldnames,
                    [
                        "track",
                        "version",
                        "dataset",
                        "entity_type",
                        "method",
                        "hyperparameters",
                        "gold_count",
                        "target_pool_size",
                        "retained_candidate_size",
                        "candidate_reduction_ratio",
                        "recall_at_1",
                        "recall_at_2",
                        "mrr",
                        "runtime_seconds",
                    ],
                )
                rows = list(reader)

        self.assertEqual(len(rows), 12)
        entity_types = {row["entity_type"] for row in rows}
        methods = {row["method"] for row in rows}
        self.assertEqual(entity_types, {"class", "predicate", "instance"})
        self.assertEqual(methods, {"tfidf", "bm25", "exact_match", "char_ngram"})

    def test_runner_uses_frozen_settings_not_grid_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp)
            selected = tmp / "heldout_selected_settings.json"
            self._write_selected_settings(selected)

            results = run_heldout_kg_experiments(
                config_path=config,
                selected_settings_path=selected,
                show_progress=False,
            )

        self.assertEqual(len(results), 12)
        for row in results:
            if row["model"] == "tfidf":
                self.assertEqual(
                    row["hyperparameters"],
                    {
                        "ngram_range": [1, 1],
                        "min_df": 1,
                        "max_df": 1.0,
                        "sublinear_tf": False,
                    },
                )
            elif row["model"] == "bm25":
                self.assertEqual(row["hyperparameters"], {"k1": 1.8, "b": 0.75})
            elif row["model"] == "exact_match":
                self.assertEqual(row["hyperparameters"], {"normalization": "light_v1"})
            else:
                self.assertEqual(
                    row["hyperparameters"],
                    {
                        "normalization": "light_v1",
                        "analyzer": "char_wb",
                        "ngram_range": [3, 5],
                    },
                )

    def test_candidate_reduction_ratio_handles_target_pool_smaller_than_kmax(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp)
            selected = tmp / "heldout_selected_settings.json"
            self._write_selected_settings(selected)

            results = run_heldout_kg_experiments(
                config_path=config,
                selected_settings_path=selected,
                show_progress=False,
            )

        predicate_row = next(
            row
            for row in results
            if row["entity_type"] == "predicate" and row["model"] == "tfidf"
        )
        class_row = next(
            row
            for row in results
            if row["entity_type"] == "class" and row["model"] == "tfidf"
        )
        self.assertEqual(predicate_row["target_pool_size"], 2)
        self.assertEqual(predicate_row["retained_candidate_size"], 2)
        self.assertAlmostEqual(predicate_row["candidate_reduction_ratio"], 0.0)
        self.assertEqual(class_row["target_pool_size"], 3)
        self.assertEqual(class_row["retained_candidate_size"], 2)
        self.assertAlmostEqual(class_row["candidate_reduction_ratio"], 1.0 - (2.0 / 3.0))

    def test_exact_match_heldout_rows_use_fixed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp)
            selected = tmp / "heldout_selected_settings.json"
            self._write_selected_settings(selected)

            results = run_heldout_kg_experiments(
                config_path=config,
                selected_settings_path=selected,
                show_progress=False,
            )

        exact_rows = [row for row in results if row["model"] == "exact_match"]
        self.assertEqual(len(exact_rows), 3)
        for row in exact_rows:
            self.assertEqual(row["hyperparameters"], {"normalization": "light_v1"})
            self.assertEqual(row["mrr"], 1.0)

    def test_char_ngram_heldout_rows_use_fixed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp)
            selected = tmp / "heldout_selected_settings.json"
            self._write_selected_settings(selected)

            results = run_heldout_kg_experiments(
                config_path=config,
                selected_settings_path=selected,
                show_progress=False,
            )

        char_ngram_rows = [row for row in results if row["model"] == "char_ngram"]
        self.assertEqual(len(char_ngram_rows), 3)
        for row in char_ngram_rows:
            self.assertEqual(
                row["hyperparameters"],
                {
                    "normalization": "light_v1",
                    "analyzer": "char_wb",
                    "ngram_range": [3, 5],
                },
            )
            self.assertEqual(row["mrr"], 1.0)

    def test_missing_selected_method_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp)
            selected = tmp / "heldout_selected_settings.json"
            self._write_selected_settings(selected, include_bm25=False)

            with self.assertRaisesRegex(
                HeldoutKgRunnerValidationError,
                "missing required method 'bm25'",
            ):
                run_heldout_kg_experiments(
                    config_path=config,
                    selected_settings_path=selected,
                    show_progress=False,
                )

    def test_malformed_selected_hyperparameters_raise_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp)
            selected = tmp / "heldout_selected_settings.json"
            self._write_selected_settings(selected)

            payload = json.loads(selected.read_text(encoding="utf-8"))
            del payload["selected_settings"]["tfidf"]["hyperparameters"]["min_df"]
            selected.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                HeldoutKgRunnerValidationError,
                "missing required hyperparameter keys: min_df",
            ):
                run_heldout_kg_experiments(
                    config_path=config,
                    selected_settings_path=selected,
                    show_progress=False,
                )

    def test_zero_target_entities_for_required_type_raises(self) -> None:
        target_without_predicates = _heldout_target_rdf().replace(
            """  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/target/resource/Thor">
    <ns1:appearsIn xmlns:ns1="http://dbkwik.webdatacommons.org/target/property/" rdf:resource="http://dbkwik.webdatacommons.org/target/resource/Film"/>
  </rdf:Description>
  <rdf:Description rdf:about="http://dbkwik.webdatacommons.org/target/resource/Asgard">
    <ns1:memberOf xmlns:ns1="http://dbkwik.webdatacommons.org/target/property/" rdf:resource="http://dbkwik.webdatacommons.org/target/resource/Heroes"/>
  </rdf:Description>
""",
            "",
            1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp, target_rdf_text=target_without_predicates)
            selected = tmp / "heldout_selected_settings.json"
            self._write_selected_settings(selected)

            with self.assertRaisesRegex(
                HeldoutKgRunnerValidationError,
                "zero target entities for entity type 'predicate'",
            ):
                run_heldout_kg_experiments(
                    config_path=config,
                    selected_settings_path=selected,
                    show_progress=False,
                )

    def test_zero_gold_for_required_type_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(
                tmp,
                alignment_rdf_text=_heldout_alignment_rdf(include_predicate=False),
            )
            selected = tmp / "heldout_selected_settings.json"
            self._write_selected_settings(selected)

            with self.assertRaisesRegex(
                HeldoutKgRunnerValidationError,
                "zero gold mappings for entity type 'predicate'",
            ):
                run_heldout_kg_experiments(
                    config_path=config,
                    selected_settings_path=selected,
                    show_progress=False,
                )

    def test_runner_uses_latest_selected_settings_when_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = self._write_config(tmp)
            results_root = tmp / "results" / "comparisons" / "result_fixture"
            selected = results_root / "heldout_selected_settings.json"
            self._write_selected_settings(selected)
            output_csv = tmp / "results" / "heldout.csv"
            old_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                results = run_heldout_kg_experiments(
                    config_path=config.name,
                    selected_settings_path=None,
                    output_csv_path=output_csv,
                    show_progress=False,
                )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(len(results), 12)
