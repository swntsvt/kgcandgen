import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.main import main


def _source_rdf() -> str:
    return """<?xml version="1.0"?>
<rdf:RDF
  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
  xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
  xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://example.org/source#SClass1">
    <rdfs:label>PlantHeight</rdfs:label>
  </owl:Class>
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
  </Alignment>
</rdf:RDF>
"""


class MainCliTests(unittest.TestCase):
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

    def test_cli_smoke_with_explicit_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = self._write_fixture_dataset(tmp_path)
            output_csv_path = tmp_path / "custom_results.csv"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--config-path",
                        str(config_path),
                        "--output-csv-path",
                        str(output_csv_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_csv_path.exists())
            self.assertIn(f"Results CSV: {output_csv_path}", stdout.getvalue())

    def test_cli_default_output_path_prints_created_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config_path = self._write_fixture_dataset(tmp_path)
            old_cwd = Path.cwd()
            stdout = io.StringIO()
            try:
                # Run from tmp dir so default results path is isolated for this test.
                os.chdir(tmp_path)
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(["--config-path", str(config_path)])
            finally:
                os.chdir(old_cwd)

            self.assertEqual(exit_code, 0)
            output_line = stdout.getvalue().strip()
            self.assertTrue(output_line.startswith("Results CSV: "))
            result_path = Path(output_line.replace("Results CSV: ", "", 1))
            self.assertTrue(result_path.name.startswith("result_"))
            self.assertTrue(result_path.suffix == ".csv")
            resolved_path = result_path if result_path.is_absolute() else (tmp_path / result_path)
            self.assertTrue(resolved_path.exists())

    def test_cli_failure_for_missing_config(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = main(["--config-path", "config/does_not_exist.yaml"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error:", stderr.getvalue())

    def test_cli_default_output_fails_when_no_new_file_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            stale_results_dir = tmp_path / "results"
            stale_results_dir.mkdir(parents=True, exist_ok=True)
            stale_file = stale_results_dir / "result_20000101_000000_deadbeef.csv"
            stale_file.write_text("track,version\n", encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            old_cwd = Path.cwd()
            try:
                os.chdir(tmp_path)
                with patch("src.main.run_experiments", return_value=[]):
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        exit_code = main(["--config-path", "config/datasets.yaml"])
            finally:
                os.chdir(old_cwd)

        self.assertEqual(exit_code, 1)
        self.assertIn("No new results file was created", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
