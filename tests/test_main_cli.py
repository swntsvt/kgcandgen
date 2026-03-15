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
        with self.assertLogs("src.main", level="ERROR") as captured:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exit_code = main(["--config-path", "config/does_not_exist.yaml"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error:", stderr.getvalue())
        combined = "\n".join(captured.output)
        self.assertIn(
            "CLI execution failed (command=run, config_path=config/does_not_exist.yaml",
            combined,
        )

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

    def test_cli_progress_flag_passes_true(self) -> None:
        with patch("src.main.run_experiments") as run_mock:
            with patch("src.main._resolve_new_result_file", return_value=Path("results/result_x.csv")):
                exit_code = main(["--config-path", "config/datasets.yaml", "--progress"])

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once()
        self.assertTrue(run_mock.call_args.kwargs["show_progress"])

    def test_cli_no_progress_flag_passes_false(self) -> None:
        with patch("src.main.run_experiments") as run_mock:
            with patch("src.main._resolve_new_result_file", return_value=Path("results/result_x.csv")):
                exit_code = main(["--config-path", "config/datasets.yaml", "--no-progress"])

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once()
        self.assertFalse(run_mock.call_args.kwargs["show_progress"])

    def test_cli_default_progress_passes_none(self) -> None:
        with patch("src.main.run_experiments") as run_mock:
            with patch("src.main._resolve_new_result_file", return_value=Path("results/result_x.csv")):
                exit_code = main(["--config-path", "config/datasets.yaml"])

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once()
        self.assertIsNone(run_mock.call_args.kwargs["show_progress"])

    def test_compare_models_explicit_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            results_csv = tmp / "result_fixture.csv"
            results_csv.write_text(
                "dataset,track,method,hyperparameters,mrr,recall_at_10\n"
                "d1,biodiv,tfidf,{},0.9,0.8\n"
                "d1,biodiv,bm25,{},0.8,0.7\n",
                encoding="utf-8",
            )
            output_dir = tmp / "comparisons"
            stdout = io.StringIO()

            with patch(
                "src.main.generate_model_comparison",
                return_value={"output_dir": output_dir},
            ) as compare_mock:
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "compare-models",
                            "--results-csv",
                            str(results_csv),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            compare_mock.assert_called_once_with(
                results_csv_path=str(results_csv),
                output_dir=str(output_dir),
            )
            self.assertIn(f"Comparison Report Dir: {output_dir}", stdout.getvalue())

    def test_compare_models_default_latest(self) -> None:
        stdout = io.StringIO()
        with patch(
            "src.main.generate_model_comparison",
            return_value={"output_dir": Path("results/comparisons/result_x")},
        ) as compare_mock:
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["compare-models"])

        self.assertEqual(exit_code, 0)
        compare_mock.assert_called_once_with(
            results_csv_path=None,
            output_dir="results/comparisons",
        )
        self.assertIn("Comparison Report Dir:", stdout.getvalue())

    def test_compare_models_failure_path(self) -> None:
        stderr = io.StringIO()
        with self.assertLogs("src.main", level="ERROR") as captured:
            with patch(
                "src.main.generate_model_comparison",
                side_effect=ValueError("bad compare input"),
            ):
                with contextlib.redirect_stderr(stderr):
                    exit_code = main(["compare-models"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: ValueError: bad compare input", stderr.getvalue())
        self.assertIn("CLI execution failed", "\n".join(captured.output))

    def test_tfidf_sensitivity_explicit_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_dir = tmp / "comparisons"
            stdout = io.StringIO()

            with patch(
                "src.main.generate_tfidf_sensitivity",
                return_value={"output_dir": output_dir},
            ) as sensitivity_mock:
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "tfidf-sensitivity",
                            "--results-csv",
                            str(tmp / "result_fixture.csv"),
                            "--config-path",
                            str(tmp / "datasets.yaml"),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            sensitivity_mock.assert_called_once_with(
                results_csv_path=str(tmp / "result_fixture.csv"),
                config_path=str(tmp / "datasets.yaml"),
                output_dir=str(output_dir),
            )
            self.assertIn(f"TF-IDF Sensitivity Dir: {output_dir}", stdout.getvalue())

    def test_tfidf_sensitivity_default_args(self) -> None:
        stdout = io.StringIO()
        with patch(
            "src.main.generate_tfidf_sensitivity",
            return_value={"output_dir": Path("results/comparisons/result_x")},
        ) as sensitivity_mock:
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["tfidf-sensitivity"])

        self.assertEqual(exit_code, 0)
        sensitivity_mock.assert_called_once_with(
            results_csv_path=None,
            config_path="config/datasets.yaml",
            output_dir="results/comparisons",
        )
        self.assertIn("TF-IDF Sensitivity Dir:", stdout.getvalue())

    def test_bm25_sensitivity_explicit_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_dir = tmp / "comparisons"
            stdout = io.StringIO()

            with patch(
                "src.main.generate_bm25_sensitivity",
                return_value={"output_dir": output_dir},
            ) as sensitivity_mock:
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "bm25-sensitivity",
                            "--results-csv",
                            str(tmp / "result_fixture.csv"),
                            "--config-path",
                            str(tmp / "datasets.yaml"),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            sensitivity_mock.assert_called_once_with(
                results_csv_path=str(tmp / "result_fixture.csv"),
                config_path=str(tmp / "datasets.yaml"),
                output_dir=str(output_dir),
            )
            self.assertIn(f"BM25 Sensitivity Dir: {output_dir}", stdout.getvalue())

    def test_bm25_sensitivity_default_args(self) -> None:
        stdout = io.StringIO()
        with patch(
            "src.main.generate_bm25_sensitivity",
            return_value={"output_dir": Path("results/comparisons/result_x")},
        ) as sensitivity_mock:
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["bm25-sensitivity"])

        self.assertEqual(exit_code, 0)
        sensitivity_mock.assert_called_once_with(
            results_csv_path=None,
            config_path="config/datasets.yaml",
            output_dir="results/comparisons",
        )
        self.assertIn("BM25 Sensitivity Dir:", stdout.getvalue())

    def test_depth_analysis_explicit_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_dir = tmp / "comparisons"
            stdout = io.StringIO()

            with patch(
                "src.main.generate_depth_analysis",
                return_value={"output_dir": output_dir},
            ) as depth_mock:
                with contextlib.redirect_stdout(stdout):
                    exit_code = main(
                        [
                            "depth-analysis",
                            "--results-csv",
                            str(tmp / "result_fixture.csv"),
                            "--output-dir",
                            str(output_dir),
                        ]
                    )

            self.assertEqual(exit_code, 0)
            depth_mock.assert_called_once_with(
                results_csv_path=str(tmp / "result_fixture.csv"),
                output_dir=str(output_dir),
            )
            self.assertIn(f"Depth Analysis Dir: {output_dir}", stdout.getvalue())

    def test_depth_analysis_default_args(self) -> None:
        stdout = io.StringIO()
        with patch(
            "src.main.generate_depth_analysis",
            return_value={"output_dir": Path("results/comparisons/result_x")},
        ) as depth_mock:
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["depth-analysis"])

        self.assertEqual(exit_code, 0)
        depth_mock.assert_called_once_with(
            results_csv_path=None,
            output_dir="results/comparisons",
        )
        self.assertIn("Depth Analysis Dir:", stdout.getvalue())

    def test_depth_analysis_failure_path(self) -> None:
        stderr = io.StringIO()
        with self.assertLogs("src.main", level="ERROR") as captured:
            with patch(
                "src.main.generate_depth_analysis",
                side_effect=ValueError("bad depth input"),
            ):
                with contextlib.redirect_stderr(stderr):
                    exit_code = main(["depth-analysis"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: ValueError: bad depth input", stderr.getvalue())
        self.assertIn("CLI execution failed", "\n".join(captured.output))

    def test_full_run_success_orders_stages_and_shares_results_csv(self) -> None:
        stdout = io.StringIO()
        call_order: list[str] = []
        resolved_csv = Path("results/result_shared.csv")
        shared_output_dir = Path("results/comparisons/result_shared")

        def _mark(name: str):
            def _fn(*_args, **_kwargs):
                call_order.append(name)
                return {"output_dir": shared_output_dir}

            return _fn

        with patch("src.main.run_experiments", side_effect=lambda **_: call_order.append("run_experiments")) as run_mock:
            with patch("src.main._resolve_new_result_file", return_value=resolved_csv) as resolve_mock:
                with patch("src.main.generate_model_comparison", side_effect=_mark("model_comparison")) as model_mock:
                    with patch("src.main.generate_tfidf_sensitivity", side_effect=_mark("tfidf_sensitivity")) as tfidf_mock:
                        with patch("src.main.generate_bm25_sensitivity", side_effect=_mark("bm25_sensitivity")) as bm25_mock:
                            with patch(
                                "src.main.generate_depth_analysis",
                                side_effect=_mark("depth_analysis"),
                            ) as depth_mock:
                                with contextlib.redirect_stdout(stdout):
                                    exit_code = main(
                                        [
                                            "full-run",
                                            "--config-path",
                                            "config/datasets.yaml",
                                            "--output-dir",
                                            "results/comparisons",
                                            "--no-progress",
                                        ]
                                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            call_order,
            [
                "run_experiments",
                "model_comparison",
                "tfidf_sensitivity",
                "bm25_sensitivity",
                "depth_analysis",
            ],
        )
        run_mock.assert_called_once_with(
            config_path="config/datasets.yaml",
            output_csv_path=None,
            show_progress=False,
        )
        resolve_mock.assert_called_once()
        model_mock.assert_called_once_with(
            results_csv_path=resolved_csv,
            output_dir="results/comparisons",
        )
        tfidf_mock.assert_called_once_with(
            results_csv_path=resolved_csv,
            config_path="config/datasets.yaml",
            output_dir="results/comparisons",
        )
        bm25_mock.assert_called_once_with(
            results_csv_path=resolved_csv,
            config_path="config/datasets.yaml",
            output_dir="results/comparisons",
        )
        depth_mock.assert_called_once_with(
            results_csv_path=resolved_csv,
            output_dir="results/comparisons",
        )
        output_text = stdout.getvalue()
        self.assertIn("Results CSV: results/result_shared.csv", output_text)
        self.assertIn("Full Run Manifest:", output_text)

    def test_full_run_explicit_output_csv_path_is_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_csv = tmp / "explicit.csv"
            output_root = tmp / "comparisons" / "result_explicit"
            output_root.mkdir(parents=True, exist_ok=True)

            def _run_side_effect(**_kwargs):
                output_csv.write_text("dataset,track,method,mrr,recall_at_10,recall_at_50\n", encoding="utf-8")

            with patch("src.main.run_experiments", side_effect=_run_side_effect) as run_mock:
                with patch("src.main._resolve_new_result_file") as resolve_mock:
                    with patch(
                        "src.main.generate_model_comparison",
                        return_value={"output_dir": output_root},
                    ) as model_mock:
                        with patch(
                            "src.main.generate_tfidf_sensitivity",
                            return_value={"output_dir": output_root},
                        ):
                            with patch(
                                "src.main.generate_bm25_sensitivity",
                                return_value={"output_dir": output_root},
                            ):
                                with patch(
                                    "src.main.generate_depth_analysis",
                                    return_value={"output_dir": output_root},
                                ):
                                    exit_code = main(
                                        [
                                            "full-run",
                                            "--config-path",
                                            "config/datasets.yaml",
                                            "--output-csv-path",
                                            str(output_csv),
                                            "--output-dir",
                                            str(tmp / "comparisons"),
                                        ]
                                    )

            self.assertEqual(exit_code, 0)
            run_mock.assert_called_once()
            resolve_mock.assert_not_called()
            model_mock.assert_called_once_with(
                results_csv_path=output_csv,
                output_dir=str(tmp / "comparisons"),
            )

    def test_full_run_fails_fast_when_stage_errors(self) -> None:
        stderr = io.StringIO()
        with self.assertLogs("src.main", level="ERROR") as captured:
            with patch("src.main.run_experiments", return_value=[]):
                with patch("src.main._resolve_new_result_file", return_value=Path("results/result_shared.csv")):
                    with patch(
                        "src.main.generate_model_comparison",
                        side_effect=ValueError("comparison failed"),
                    ):
                        with patch("src.main.generate_tfidf_sensitivity") as tfidf_mock:
                            with patch("src.main.generate_bm25_sensitivity") as bm25_mock:
                                with patch("src.main.generate_depth_analysis") as depth_mock:
                                    with contextlib.redirect_stderr(stderr):
                                        exit_code = main(["full-run", "--no-progress"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: ValueError: comparison failed", stderr.getvalue())
        self.assertIn("Full-run stage failed stage=model_comparison", "\n".join(captured.output))
        tfidf_mock.assert_not_called()
        bm25_mock.assert_not_called()
        depth_mock.assert_not_called()

    def test_full_run_writes_manifest_on_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            output_csv = tmp / "run.csv"
            output_root = tmp / "comparisons" / "result_run"
            output_root.mkdir(parents=True, exist_ok=True)

            def _run_side_effect(**_kwargs):
                output_csv.write_text("track,version,dataset,method,mrr,recall_at_10,recall_at_50\n", encoding="utf-8")

            with patch("src.main.run_experiments", side_effect=_run_side_effect):
                with patch(
                    "src.main.generate_model_comparison",
                    return_value={"output_dir": output_root},
                ):
                    with patch(
                        "src.main.generate_tfidf_sensitivity",
                        return_value={"output_dir": output_root},
                    ):
                        with patch(
                            "src.main.generate_bm25_sensitivity",
                            return_value={"output_dir": output_root},
                        ):
                            with patch(
                                "src.main.generate_depth_analysis",
                                return_value={"output_dir": output_root},
                            ):
                                exit_code = main(
                                    [
                                        "full-run",
                                        "--output-csv-path",
                                        str(output_csv),
                                        "--output-dir",
                                        str(tmp / "comparisons"),
                                    ]
                                )

            self.assertEqual(exit_code, 0)
            manifest_path = output_root / "full_run_manifest.txt"
            self.assertTrue(manifest_path.exists())
            manifest_text = manifest_path.read_text(encoding="utf-8")
            self.assertIn("generated_at_start=", manifest_text)
            self.assertIn("generated_at_end=", manifest_text)
            self.assertIn("elapsed_seconds=", manifest_text)
            self.assertIn("stage_run_experiments=success", manifest_text)
            self.assertIn("stage_model_comparison=success", manifest_text)
            self.assertIn("stage_tfidf_sensitivity=success", manifest_text)
            self.assertIn("stage_bm25_sensitivity=success", manifest_text)
            self.assertIn("stage_depth_analysis=success", manifest_text)
            self.assertIn(f"results_csv={output_csv.resolve()}", manifest_text)

    def test_top_level_help_lists_compare_models_command(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with self.assertRaises(SystemExit) as exc:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                main(["--help"])

        self.assertEqual(exc.exception.code, 0)
        help_text = stdout.getvalue()
        self.assertIn("compare-models", help_text)
        self.assertIn("tfidf-sensitivity", help_text)
        self.assertIn("depth-analysis", help_text)
        self.assertIn("bm25-sensitivity", help_text)
        self.assertIn("full-run", help_text)
        self.assertIn("run", help_text)


if __name__ == "__main__":
    unittest.main()
