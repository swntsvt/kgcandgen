import unittest
from unittest.mock import patch

import src.preprocessing.text_preprocessor as text_preprocessor_module
from src.preprocessing.text_preprocessor import preprocess_text, validate_nltk_assets


class TextPreprocessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_ready = text_preprocessor_module._NLTK_RESOURCES_READY
        self._original_paths = list(text_preprocessor_module.nltk.data.path)
        text_preprocessor_module._NLTK_RESOURCES_READY = False

    def tearDown(self) -> None:
        text_preprocessor_module._NLTK_RESOURCES_READY = self._original_ready
        text_preprocessor_module.nltk.data.path[:] = self._original_paths

    def test_lowercasing(self) -> None:
        self.assertEqual(preprocess_text("HeLLo WoRLD"), ["hello", "world"])

    def test_camel_case_splitting(self) -> None:
        self.assertEqual(
            preprocess_text("PlantHeightValue"), ["plant", "height", "value"]
        )

    def test_tokenization_with_mixed_separators(self) -> None:
        self.assertEqual(
            preprocess_text("Leaf-size,Shape/Color"),
            ["leaf", "size", "shape", "color"],
        )

    def test_stopword_removal(self) -> None:
        self.assertEqual(preprocess_text("The color of the leaf"), ["color", "leaf"])

    def test_punctuation_removal(self) -> None:
        self.assertEqual(preprocess_text("leaf, (green)."), ["leaf", "green"])

    def test_full_pipeline_combination(self) -> None:
        self.assertEqual(
            preprocess_text("The PlantHeightValue, of LeafSize!"),
            ["plant", "height", "value", "leaf", "size"],
        )

    def test_edge_cases_empty_and_punctuation_only(self) -> None:
        self.assertEqual(preprocess_text(""), [])
        self.assertEqual(preprocess_text("...!!!"), [])

    def test_deterministic_output(self) -> None:
        text = "PlantHeightValue of LeafSize"
        first = preprocess_text(text)
        second = preprocess_text(text)
        self.assertEqual(first, second)

    def test_no_runtime_download_attempted(self) -> None:
        with patch(
            "src.preprocessing.text_preprocessor.nltk.download",
            side_effect=AssertionError("nltk.download must not be called"),
        ) as download_mock:
            tokens = preprocess_text("The PlantHeightValue, of LeafSize!")

        self.assertEqual(tokens, ["plant", "height", "value", "leaf", "size"])
        download_mock.assert_not_called()

    def test_project_nltk_path_is_prioritized(self) -> None:
        project_path = str(text_preprocessor_module._PROJECT_NLTK_DATA_DIR)
        text_preprocessor_module.nltk.data.path[:] = [
            path for path in text_preprocessor_module.nltk.data.path if path != project_path
        ]
        text_preprocessor_module.nltk.data.path.append(project_path)

        validate_nltk_assets()

        self.assertEqual(text_preprocessor_module.nltk.data.path[0], project_path)

    def test_missing_assets_raise_actionable_error(self) -> None:
        with patch(
            "src.preprocessing.text_preprocessor.nltk.data.find",
            side_effect=LookupError("missing resource"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Required bundled NLTK resources are missing or unreadable",
            ) as exc:
                validate_nltk_assets()

        message = str(exc.exception)
        self.assertIn("resources/nltk_data", message)
        self.assertIn("package='punkt'", message)

    def test_corrupt_assets_raise_actionable_error(self) -> None:
        with patch(
            "src.preprocessing.text_preprocessor.stopwords.words",
            side_effect=OSError("corrupt stopwords"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "present but unreadable/corrupt",
            ) as exc:
                validate_nltk_assets()

        message = str(exc.exception)
        self.assertIn("resources/nltk_data", message)
        self.assertIn("corrupt stopwords", message)


if __name__ == "__main__":
    unittest.main()
