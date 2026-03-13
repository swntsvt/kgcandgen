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
datasets:
  conference_v1:
    track: conference
    version: "1"
    source_rdf: {source}
    target_rdf: {target}
    alignment_rdf: {alignment}
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
            + "\n"
        )

    def _write_valid_config(self, tmp: Path) -> Path:
        source = tmp / "source.rdf"
        target = tmp / "target.rdf"
        alignment = tmp / "alignment.rdf"
        source.write_text("", encoding="utf-8")
        target.write_text("", encoding="utf-8")
        alignment.write_text("", encoding="utf-8")

        config_path = tmp / "datasets.yaml"
        config_path.write_text(
            self._build_valid_config_text(source, target, alignment), encoding="utf-8"
        )
        return config_path

    def test_load_and_get_dataset_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))

            runtime = load_runtime_config(config_path)
            datasets = runtime.datasets
            self.assertIn("conference_v1", datasets)
            self.assertIsInstance(datasets["conference_v1"], DatasetConfig)
            self.assertEqual(datasets["conference_v1"].version, "1")

            conf = get_dataset_config("conference_v1", config_path)
            self.assertEqual(conf.track, "conference")
            self.assertEqual(conf.source_rdf, datasets["conference_v1"].source_rdf)

    def test_load_runtime_config_includes_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))

            runtime = load_runtime_config(config_path)

        self.assertIn("conference_v1", runtime.datasets)
        self.assertIsInstance(runtime.experiments, ExperimentConfig)
        self.assertEqual(runtime.experiments.evaluation_ks, [1, 5, 10, 20, 50])
        self.assertEqual(len(runtime.experiments.tfidf_grid), 2)
        self.assertEqual(len(runtime.experiments.bm25_grid), 2)

    def test_logging_on_successful_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))

            with self.assertLogs("src.config_loader", level="INFO") as captured:
                runtime = load_runtime_config(config_path)

        self.assertIn("conference_v1", runtime.datasets)
        combined = "\n".join(captured.output)
        self.assertIn("Loading datasets config", combined)
        self.assertIn("Discovered 1 dataset entry(ies)", combined)
        self.assertIn("Validated dataset 'conference_v1'", combined)
        self.assertIn("Validated experiment config", combined)

    def test_invalid_yaml_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "datasets.yaml"
            config_path.write_text("datasets: [broken", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_runtime_config(config_path)

    def test_missing_dataset_file_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config_path = tmp / "datasets.yaml"
            config_path.write_text(
                (
                    f"""
datasets:
  conference_v1:
    track: conference
    version: "1"
    source_rdf: {tmp / "missing_source.rdf"}
    target_rdf: {target}
    alignment_rdf: {alignment}
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
                    + "\n"
                ),
                encoding="utf-8",
            )

            with self.assertLogs("src.config_loader", level="ERROR") as captured:
                with self.assertRaises(FileNotFoundError):
                    load_runtime_config(config_path)

        combined = "\n".join(captured.output)
        self.assertIn("missing file for 'source_rdf'", combined)

    def test_missing_experiments_mapping_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.rdf"
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            source.write_text("", encoding="utf-8")
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config_path = tmp / "datasets.yaml"
            config_path.write_text(
                (
                    f"""
datasets:
  conference_v1:
    track: conference
    version: "1"
    source_rdf: {source}
    target_rdf: {target}
    alignment_rdf: {alignment}
""".strip()
                    + "\n"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "'experiments' mapping"):
                load_runtime_config(config_path)

    def test_experiments_must_be_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.rdf"
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            source.write_text("", encoding="utf-8")
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config_path = tmp / "datasets.yaml"
            config_path.write_text(
                (
                    f"""
datasets:
  conference_v1:
    track: conference
    version: "1"
    source_rdf: {source}
    target_rdf: {target}
    alignment_rdf: {alignment}
experiments: []
""".strip()
                    + "\n"
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "experiments must be a mapping"):
                load_runtime_config(config_path)

    def test_invalid_evaluation_ks_raises(self) -> None:
        invalid_cases = [
            ("[]", "experiments.evaluation_ks must be a non-empty list"),
            ("[1, 0]", r"experiments.evaluation_ks\[1\] must be a positive integer"),
            ("[1, 1]", "duplicates k=1"),
            ("[1, true]", r"experiments.evaluation_ks\[1\] must be a positive integer"),
        ]

        for ks_value, error_pattern in invalid_cases:
            with self.subTest(ks_value=ks_value):
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_path = self._write_valid_config(Path(tmpdir))
                    text = config_path.read_text(encoding="utf-8").replace(
                        "evaluation_ks: [1, 5, 10, 20, 50]",
                        f"evaluation_ks: {ks_value}",
                    )
                    config_path.write_text(text, encoding="utf-8")

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

    def test_invalid_tfidf_df_thresholds_raise(self) -> None:
        invalid_cases = [
            (
                "min_df: 1\n      max_df: 1.0",
                "min_df: 0\n      max_df: 1.0",
                r"experiments.tfidf_grid\[0\]\.min_df must be >= 1 when provided as an integer",
            ),
            (
                "min_df: 1\n      max_df: 1.0",
                "min_df: 0.0\n      max_df: 1.0",
                r"experiments.tfidf_grid\[0\]\.min_df must be in \(0.0, 1.0\] when provided as a float",
            ),
            (
                "min_df: 1\n      max_df: 1.0",
                "min_df: 1\n      max_df: 1.2",
                r"experiments.tfidf_grid\[0\]\.max_df must be in \(0.0, 1.0\] when provided as a float",
            ),
            (
                "min_df: 1\n      max_df: 1.0",
                "min_df: 3\n      max_df: 2",
                r"experiments.tfidf_grid\[0\] must satisfy min_df <= max_df when both are integers",
            ),
            (
                "min_df: 1\n      max_df: 1.0",
                "min_df: 0.8\n      max_df: 0.7",
                r"experiments.tfidf_grid\[0\] must satisfy min_df <= max_df when both are floats",
            ),
        ]

        for find_text, replace_text, error_pattern in invalid_cases:
            with self.subTest(replace_text=replace_text):
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_path = self._write_valid_config(Path(tmpdir))
                    config_path.write_text(
                        config_path.read_text(encoding="utf-8").replace(
                            find_text,
                            replace_text,
                            1,
                        ),
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(ValueError, error_pattern):
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

    def test_invalid_bm25_negative_k1_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._write_valid_config(Path(tmpdir))
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace("k1: 1.5", "k1: -0.1", 1),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "experiments.bm25_grid\\[0\\].k1 must be greater than or equal to 0",
            ):
                load_runtime_config(config_path)


if __name__ == "__main__":
    unittest.main()
