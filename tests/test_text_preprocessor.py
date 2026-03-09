import unittest

from src.preprocessing.text_preprocessor import preprocess_text


class TextPreprocessorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
