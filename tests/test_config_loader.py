import tempfile
import unittest
from pathlib import Path

from src.config_loader import DatasetConfig, get_dataset_config, load_datasets_config


class ConfigLoaderTests(unittest.TestCase):
    def test_load_and_get_dataset_config(self) -> None:
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
                f"""
datasets:
  conference_v1:
    track: conference
    version: "1"
    source_rdf: {source}
    target_rdf: {target}
    alignment_rdf: {alignment}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            datasets = load_datasets_config(config_path)
            self.assertIn("conference_v1", datasets)
            self.assertIsInstance(datasets["conference_v1"], DatasetConfig)
            self.assertEqual(datasets["conference_v1"].version, "1")

            conf = get_dataset_config("conference_v1", config_path)
            self.assertEqual(conf.track, "conference")
            self.assertEqual(conf.source_rdf, source)

    def test_logging_on_successful_load(self) -> None:
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
                f"""
datasets:
  conference_v1:
    track: conference
    version: "1"
    source_rdf: {source}
    target_rdf: {target}
    alignment_rdf: {alignment}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertLogs("src.config_loader", level="INFO") as captured:
                datasets = load_datasets_config(config_path)

        self.assertIn("conference_v1", datasets)
        combined = "\n".join(captured.output)
        self.assertIn("Loading datasets config", combined)
        self.assertIn("Discovered 1 dataset entry(ies)", combined)
        self.assertIn("Validated dataset 'conference_v1'", combined)
        self.assertIn("Loaded 1 validated dataset config(s)", combined)

    def test_invalid_yaml_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "datasets.yaml"
            config_path.write_text("datasets: [broken", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_datasets_config(config_path)

    def test_missing_dataset_file_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            target = tmp / "target.rdf"
            alignment = tmp / "alignment.rdf"
            target.write_text("", encoding="utf-8")
            alignment.write_text("", encoding="utf-8")

            config_path = tmp / "datasets.yaml"
            config_path.write_text(
                f"""
datasets:
  conference_v1:
    track: conference
    version: "1"
    source_rdf: {tmp / "missing_source.rdf"}
    target_rdf: {target}
    alignment_rdf: {alignment}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertLogs("src.config_loader", level="ERROR") as captured:
                with self.assertRaises(FileNotFoundError):
                    load_datasets_config(config_path)

        combined = "\n".join(captured.output)
        self.assertIn("missing file for 'source_rdf'", combined)


if __name__ == "__main__":
    unittest.main()
