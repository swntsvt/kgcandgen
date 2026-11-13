import unittest

from src.retrieval.char_ngram_retriever import CharNgramRetriever


class CharNgramRetrieverTests(unittest.TestCase):
    def test_retrieves_similar_labels_after_light_normalization(self) -> None:
        retriever = CharNgramRetriever()
        retriever.fit(
            ["urn:target:1", "urn:target:2"],
            ["plant height", "leaf color"],
        )

        results = retriever.retrieve("PlantHeight", k=2)

        self.assertEqual([entity_id for entity_id, _score in results], ["urn:target:1"])
        self.assertGreater(results[0][1], 0.0)

        padded = CharNgramRetriever(candidate_policy="padded")
        padded.fit(["urn:target:1", "urn:target:2"], ["plant height", "leaf color"])
        padded_results = padded.retrieve("PlantHeight", k=2)
        self.assertEqual(padded_results[0][0], "urn:target:1")
        self.assertGreater(padded_results[0][1], padded_results[1][1])

    def test_is_robust_to_separator_and_camel_case_variation(self) -> None:
        retriever = CharNgramRetriever()
        retriever.fit(
            ["urn:target:1", "urn:target:2"],
            ["member of", "appears in"],
        )

        results = retriever.retrieve("memberOf", k=2)

        self.assertEqual(results[0][0], "urn:target:1")

    def test_deterministic_tie_ordering_uses_target_uri_order(self) -> None:
        retriever = CharNgramRetriever()
        retriever.fit(
            ["urn:target:b", "urn:target:a"],
            ["hero", "hero"],
        )

        results = retriever.retrieve("hero", k=2)

        self.assertEqual([entity_id for entity_id, _score in results], ["urn:target:a", "urn:target:b"])
        self.assertAlmostEqual(results[0][1], 1.0)
        self.assertAlmostEqual(results[1][1], 1.0)

    def test_no_shared_ngrams_returns_no_candidates(self) -> None:
        retriever = CharNgramRetriever()
        retriever.fit(
            ["urn:target:1", "urn:target:2"],
            ["thor", "loki"],
        )

        self.assertEqual(retriever.retrieve("asgard", k=2), [])

    def test_padded_policy_still_returns_zero_score_candidates(self) -> None:
        retriever = CharNgramRetriever(candidate_policy="padded")
        retriever.fit(
            ["urn:target:1", "urn:target:2"],
            ["thor", "loki"],
        )

        results = retriever.retrieve("asgard", k=2)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(score == 0.0 for _, score in results))


if __name__ == "__main__":
    unittest.main()
