import tempfile
import unittest
from pathlib import Path

from src.config_loader import (
    DatasetConfig,
    ExperimentConfig,
    get_dataset_config,
    load_runtime_config,
)


class ConfigLoaderTests(unittest.TestCase):
    def _build_valid_config_text(self, source: Path, target: Path, alignment: Path) -> str:
        return (
            f"""
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
development_datasets:
  conference_v1:
    track: conference
    version: "1"
    source_rdf: {source}
    target_rdf: {target}
    alignment_rdf: {alignment}
heldout_datasets:
  kg_v1:
    track: kg
    version: "heldout-v1"
    source_rdf: {source}
    target_rdf: {target}
    alignment_rdf: {alignment}
heldout:
  selection:
    metric: mrr
    lambda: 0.5
    weighting: equal_track_weight
    ranking: per_track_normalized_rank
""".strip()
            + "\n"
        )

    def _write_valid_config(self, tmp: Path) -> Path:
        source = tmp / "source.rdf"
        target = tmp / "target.rdf"
        alignment = tmp / "alignment.rdf"
        source.write_text("", encoding="utf-8")
        target.write_text("", encoding="utf-8")
        alignment.write_text("", encoding="utf-8")

        config_path = tmp / "runtime.yaml"
        config_path.write_text(
            self._build_valid_config_text(source, target, alignment),
            encoding="utf-8",
        )
        return config_path

    def test_load_and_get_dataset_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))

            runtime = load_runtime_config(config_path)
            datasets = runtime.development_datasets
            self.assertIn("conference_v1", datasets)
            self.assertIsInstance(datasets["conference_v1"], DatasetConfig)
            self.assertEqual(datasets["conference_v1"].version, "1")
            self.assertIn("kg_v1", runtime.heldout_datasets)

            conf = get_dataset_config("conference_v1", config_path)
            self.assertEqual(conf.track, "conference")
            self.assertEqual(conf.source_rdf, datasets["conference_v1"].source_rdf)

    def test_load_runtime_config_includes_experiments_and_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))
            runtime = load_runtime_config(config_path)

        self.assertIn("conference_v1", runtime.datasets)
        self.assertIsInstance(runtime.experiments, ExperimentConfig)
        self.assertEqual(runtime.experiments.evaluation_ks, [1, 5, 10, 20, 50])
        self.assertEqual(len(runtime.experiments.tfidf_grid), 2)
        self.assertEqual(len(runtime.experiments.bm25_grid), 2)
        self.assertEqual(runtime.heldout.selection.metric, "mrr")
        self.assertEqual(runtime.heldout.selection.lambda_penalty, 0.5)

    def test_logging_on_successful_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))
            with self.assertLogs("src.config_loader", level="INFO") as captured:
                runtime = load_runtime_config(config_path)

        self.assertIn("conference_v1", runtime.development_datasets)
        combined = "\n".join(captured.output)
        self.assertIn("Loading runtime config", combined)
        self.assertIn("development_datasets", combined)
        self.assertIn("heldout_datasets", combined)
        self.assertIn("Validated heldout selection policy", combined)

    def test_invalid_yaml_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "runtime.yaml"
            config_path.write_text("development_datasets: [broken", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_runtime_config(config_path)

    def test_missing_dataset_file_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config_path = tmp / "runtime.yaml"
            config_path.write_text(
                self._build_valid_config_text(tmp / "missing_source.rdf", target, alignment),
                encoding="utf-8",
            )

            with self.assertLogs("src.config_loader", level="ERROR") as captured:
                with self.assertRaises(FileNotFoundError):
                    load_runtime_config(config_path)

        self.assertIn("missing file for 'source_rdf'", "\n".join(captured.output))

    def test_missing_development_datasets_mapping_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace(
                    "development_datasets:", "datasets:", 1
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "'development_datasets' mapping"):
                load_runtime_config(config_path)

    def test_missing_heldout_mapping_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))
            text = config_path.read_text(encoding="utf-8")
            config_path.write_text(text.split("heldout:\n", 1)[0], encoding="utf-8")

            runtime = load_runtime_config(config_path)

        self.assertEqual(runtime.heldout_datasets["kg_v1"].track, "kg")
        self.assertIsNone(runtime.heldout)

    def test_missing_heldout_mapping_raises_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))
            text = config_path.read_text(encoding="utf-8")
            config_path.write_text(text.split("heldout:\n", 1)[0], encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "'heldout' mapping"):
                load_runtime_config(config_path, require_heldout=True)

    def test_dev_only_config_loads_for_non_heldout_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.rdf"
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            source.write_text("", encoding="utf-8")
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config_path = tmp / "runtime.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "experiments:",
                        "  evaluation_ks: [1]",
                        "  tfidf_grid:",
                        "    - ngram_range: [1, 1]",
                        "      min_df: 1",
                        "      max_df: 1.0",
                        "      sublinear_tf: false",
                        "  bm25_grid:",
                        "    - k1: 1.5",
                        "      b: 0.75",
                        "development_datasets:",
                        "  conference_v1:",
                        "    track: conference",
                        '    version: "1"',
                        f"    source_rdf: {source}",
                        f"    target_rdf: {target}",
                        f"    alignment_rdf: {alignment}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            runtime = load_runtime_config(config_path)

        self.assertIn("conference_v1", runtime.development_datasets)
        self.assertEqual(runtime.heldout_datasets, {})
        self.assertIsNone(runtime.heldout)

    def test_invalid_heldout_policy_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace(
                    "lambda: 0.5", "lambda: 0.7", 1
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "fixed supported policy"):
                load_runtime_config(config_path, require_heldout=True)

    def test_invalid_evaluation_ks_raises(self) -> None:
        invalid_cases = [
            ("[]", "experiments.evaluation_ks must be a non-empty list"),
            ("[1, 0]", r"experiments.evaluation_ks\[1\] must be a positive integer"),
            ("[1, 1]", "duplicates k=1"),
        ]

        for ks_value, error_pattern in invalid_cases:
            with self.subTest(ks_value=ks_value):
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_path = self._write_valid_config(Path(tmpdir))
                    config_path.write_text(
                        config_path.read_text(encoding="utf-8").replace(
                            "evaluation_ks: [1, 5, 10, 20, 50]",
                            f"evaluation_ks: {ks_value}",
                            1,
                        ),
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(ValueError, error_pattern):
                        load_runtime_config(config_path)

    def test_invalid_tfidf_grid_entry_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace(
                    "ngram_range: [1, 2]",
                    "ngram_range: [2, 1]",
                    1,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "experiments.tfidf_grid\\[1\\].ngram_range must satisfy min_n <= max_n",
            ):
                load_runtime_config(config_path)

    def test_invalid_bm25_grid_entry_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace("b: 0.75", "b: 1.5", 1),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "experiments.bm25_grid\\[0\\].b must be between 0 and 1",
            ):
                load_runtime_config(config_path)

    def test_bm25_k1_allows_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace("k1: 1.5", "k1: 0.0", 1),
                encoding="utf-8",
            )

            runtime = load_runtime_config(config_path)

        self.assertEqual(runtime.experiments.bm25_grid[0].k1, 0.0)


if __name__ == "__main__":
    unittest.main()
