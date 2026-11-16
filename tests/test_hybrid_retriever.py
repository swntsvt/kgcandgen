import pickle
import unittest
from unittest.mock import patch

from src.retrieval.char_ngram_retriever import CharNgramRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from tests.test_dense_retriever import _StubModel

IDS = [f"urn:{i}" for i in range(8)]
LABELS = ["red apple", "green apple", "car", "tree", "old tree", "cart", "heart valve", "lung"]


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_scores_are_summed_reciprocal_ranks(self) -> None:
        fused = reciprocal_rank_fusion(
            [[("a", 9.0), ("b", 5.0)], [("b", 0.9), ("c", 0.1)]], rrf_k=60, k=10
        )
        self.assertEqual([target for target, _ in fused], ["b", "a", "c"])
        self.assertAlmostEqual(dict(fused)["b"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(dict(fused)["a"], 1 / 61)
        self.assertAlmostEqual(dict(fused)["c"], 1 / 62)

    def test_ties_are_broken_by_target_id(self) -> None:
        fused = reciprocal_rank_fusion([[("z", 1.0)], [("m", 1.0)]], rrf_k=60, k=2)
        self.assertEqual([target for target, _ in fused], ["m", "z"])

    def test_union_is_truncated_to_k_and_empty_lists_are_fine(self) -> None:
        self.assertEqual(len(reciprocal_rank_fusion([[("a", 1), ("b", 1)], [("c", 1)]], rrf_k=60, k=2)), 2)
        self.assertEqual(reciprocal_rank_fusion([[], []], rrf_k=60, k=5), [])

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], rrf_k=0, k=1)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], rrf_k=60, k=0)


class HybridRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("src.retrieval.dense_retriever._load_model", return_value=_StubModel())
        patcher.start()
        self.addCleanup(patcher.stop)

    def _fitted(self, **kwargs) -> HybridRetriever:
        retriever = HybridRetriever(model_name="stub", revision="r1", **kwargs)
        retriever.fit(IDS, LABELS)
        return retriever

    def test_matches_fusion_of_its_components(self) -> None:
        lexical = CharNgramRetriever()
        lexical.fit(IDS, LABELS)
        dense = DenseRetriever(model_name="stub", revision="r1")
        dense.fit(IDS, LABELS)
        hybrid = self._fitted(depth=4)
        for query in ("apple", "trees", "heart valves"):
            expected = reciprocal_rank_fusion(
                [lexical.retrieve(query, 4), dense.retrieve(query, 4)], rrf_k=60, k=3
            )
            self.assertEqual(hybrid.retrieve(query, k=3), expected)

    def test_retrieve_many_matches_retrieve(self) -> None:
        hybrid = self._fitted()
        queries = ["apple", "trees", "carts", "zzz"]
        self.assertEqual(hybrid.retrieve_many(queries, k=3), [hybrid.retrieve(q, k=3) for q in queries])

    def test_nonzero_policy_returns_no_candidate_when_neither_component_does(self) -> None:
        hybrid = HybridRetriever(model_name="stub", revision="r1")
        hybrid.fit(["urn:a", "urn:b"], ["abc", "abd"])
        self.assertEqual(hybrid.retrieve("xyz", k=5), [])

    def test_is_picklable_for_index_size(self) -> None:
        self.assertGreater(len(pickle.dumps(self._fitted())), 0)

    def test_invalid_settings_raise(self) -> None:
        for kwargs in ({"rrf_k": 0}, {"depth": 0}, {"candidate_policy": "all"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HybridRetriever(model_name="stub", **kwargs)
        with self.assertRaises(ValueError):
            self._fitted().retrieve("apple", k=0)


if __name__ == "__main__":
    unittest.main()
