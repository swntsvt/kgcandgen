import unittest

from src.retrieval.bm25_retriever import Bm25Retriever


class Bm25RetrieverTests(unittest.TestCase):
    def test_basic_top_k_retrieval(self) -> None:
        retriever = Bm25Retriever()
        entity_ids = ["e1", "e2", "e3"]
        labels = ["red apple", "blue car", "green tree"]
        retriever.fit(entity_ids, labels)

        results = retriever.retrieve("apple", k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "e1")
        self.assertGreaterEqual(results[0][1], results[1][1])

    def test_hyperparameters_are_propagated(self) -> None:
        retriever = Bm25Retriever(k1=1.2, b=0.65)

        self.assertEqual(retriever._model.k1, 1.2)
        self.assertEqual(retriever._model.b, 0.65)

    def test_k_larger_than_index_size_returns_all(self) -> None:
        retriever = Bm25Retriever()
        retriever.fit(["e1", "e2"], ["alpha", "beta"])

        results = retriever.retrieve("alpha", k=10)

        self.assertEqual(len(results), 2)

    def test_preprocessing_integration(self) -> None:
        retriever = Bm25Retriever()
        retriever.fit(["e1", "e2"], ["PlantHeightValue", "LeafColor"])

        results = retriever.retrieve("plant height value", k=1)

        self.assertEqual(results[0][0], "e1")

    def test_tokenized_entrypoints(self) -> None:
        retriever = Bm25Retriever()
        retriever.fit_tokenized(
            ["e1", "e2"],
            [["plant", "height", "value"], ["leaf", "color"]],
        )

        results = retriever.retrieve_tokenized(["plant", "height", "value"], k=1)

        self.assertEqual(results[0][0], "e1")

    def test_retrieve_before_fit_raises(self) -> None:
        retriever = Bm25Retriever()
        with self.assertRaises(ValueError):
            retriever.retrieve("query", k=1)

    def test_invalid_inputs_raise(self) -> None:
        retriever = Bm25Retriever()

        with self.assertRaises(ValueError):
            retriever.fit(["e1"], ["label1", "label2"])

        with self.assertRaises(ValueError):
            retriever.fit([], [])

        retriever.fit(["e1"], ["label1"])
        with self.assertRaises(ValueError):
            retriever.retrieve("label1", k=0)

    def test_deterministic_repeated_retrieval(self) -> None:
        retriever = Bm25Retriever()
        retriever.fit(["e1", "e2", "e3"], ["alpha beta", "alpha gamma", "delta"])

        first = retriever.retrieve("alpha", k=3)
        second = retriever.retrieve("alpha", k=3)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
