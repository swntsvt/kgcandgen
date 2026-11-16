import unittest
from unittest.mock import patch

import numpy as np

from src.retrieval.dense_retriever import DenseRetriever


class _StubModel:
    """Letter-count embeddings: deterministic, and 'zzz' is orthogonal to most labels."""

    def encode(self, texts, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        vectors = np.zeros((len(texts), 27), dtype=np.float32)
        for row, text in enumerate(texts):
            for char in text:
                index = ord(char) - ord("a") if "a" <= char <= "z" else 26
                vectors[row, index] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)


def _retriever(**kwargs) -> DenseRetriever:
    return DenseRetriever(model_name="stub", revision="r1", **kwargs)


class DenseRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("src.retrieval.dense_retriever._load_model", return_value=_StubModel())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_ranks_by_cosine_and_normalises_labels(self) -> None:
        retriever = _retriever()
        retriever.fit(["urn:b", "urn:a"], ["HeartValve", "lung"])
        results = retriever.retrieve("heart_valve", k=2)
        self.assertEqual(results[0][0], "urn:b")
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

    def test_nonzero_policy_drops_non_positive_scores(self) -> None:
        retriever = _retriever()
        retriever.fit(["urn:a", "urn:b"], ["abc", "xyz"])
        self.assertEqual([e for e, _ in retriever.retrieve("abc", k=5)], ["urn:a"])
        padded = _retriever(candidate_policy="padded")
        padded.fit(["urn:a", "urn:b"], ["abc", "xyz"])
        self.assertEqual(len(padded.retrieve("abc", k=5)), 2)

    def test_ties_follow_target_uri_order(self) -> None:
        retriever = _retriever()
        retriever.fit(["urn:z", "urn:m", "urn:a"], ["hero", "hero", "hero"])
        self.assertEqual([e for e, _ in retriever.retrieve("hero", k=3)], ["urn:a", "urn:m", "urn:z"])

    def test_retrieve_many_matches_retrieve(self) -> None:
        retriever = _retriever()
        retriever.fit([f"urn:{i}" for i in range(6)], ["red apple", "green apple", "car", "tree", "old tree", "cart"])
        queries = ["apple", "trees", "carts", "zzz"]
        self.assertEqual(retriever.retrieve_many(queries, k=3), [retriever.retrieve(q, k=3) for q in queries])

    def test_invalid_use_raises(self) -> None:
        with self.assertRaises(ValueError):
            _retriever().retrieve("x", k=1)
        with self.assertRaises(ValueError):
            _retriever(batch_size=0)


if __name__ == "__main__":
    unittest.main()
