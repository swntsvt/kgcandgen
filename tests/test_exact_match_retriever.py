import unittest

from src.preprocessing.exact_match_normalizer import normalize_exact_match_text
from src.retrieval.exact_match_retriever import ExactMatchRetriever


class ExactMatchRetrieverTests(unittest.TestCase):
    def test_normalization_is_conservative_and_preserves_stopwords(self) -> None:
        self.assertEqual(normalize_exact_match_text("PlantHeight"), "plant height")
        self.assertEqual(
            normalize_exact_match_text("Plant-Height/Value_Test"),
            "plant height value test",
        )
        self.assertEqual(normalize_exact_match_text("The Hero"), "the hero")

    def test_retrieves_normalized_exact_match(self) -> None:
        retriever = ExactMatchRetriever()
        retriever.fit(
            ["urn:target:1", "urn:target:2"],
            ["plant height", "leaf color"],
        )

        self.assertEqual(
            retriever.retrieve("PlantHeight", k=5),
            [("urn:target:1", 1.0)],
        )

    def test_multiple_matches_are_ranked_by_target_uri(self) -> None:
        retriever = ExactMatchRetriever()
        retriever.fit(
            ["urn:target:z", "urn:target:a", "urn:target:m"],
            ["Thor", "thor", "Loki"],
        )

        self.assertEqual(
            retriever.retrieve("THOR", k=5),
            [("urn:target:a", 1.0), ("urn:target:z", 1.0)],
        )

    def test_empty_match_returns_no_candidates(self) -> None:
        retriever = ExactMatchRetriever()
        retriever.fit(["urn:target:1"], ["plant height"])

        self.assertEqual(retriever.retrieve("root length", k=5), [])


if __name__ == "__main__":
    unittest.main()
