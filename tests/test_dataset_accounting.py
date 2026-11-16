import math
import unittest

from pyoxigraph import Literal, NamedNode, Quad, Store

from src.analysis.dataset_accounting import label_statistics
from src.rdf_utils.label_extractor import RDFS_LABEL, extract_entity_label_with_source


def _store(labels: dict[str, list[str]]) -> Store:
    store = Store()
    for uri, values in labels.items():
        for value in values:
            store.add(Quad(NamedNode(uri), NamedNode(RDFS_LABEL), Literal(value)))
    return store


class DatasetAccountingTests(unittest.TestCase):
    def test_label_source_reports_uri_fallback_and_multiple_labels(self) -> None:
        store = _store({"http://x/A": ["beta", "alpha"]})
        self.assertEqual(
            extract_entity_label_with_source(store, "http://x/A"), ("alpha", "rdfs_label", 2)
        )
        self.assertEqual(
            extract_entity_label_with_source(store, "http://x/onto#HeartValve"),
            ("HeartValve", "uri", 0),
        )

    def test_label_statistics(self) -> None:
        store = _store({"http://x/A": ["heart valve"], "http://x/B": ["the"], "http://x/C": ["lung"]})
        stats = label_statistics(store, ["http://x/A", "http://x/B", "http://x/C", "http://x/onto#Kidney"])

        self.assertEqual(stats["count"], 4)
        self.assertAlmostEqual(stats["single_token_share"], 0.5)  # lung, kidney
        self.assertAlmostEqual(stats["zero_token_share"], 0.25)  # "the" is a stopword
        self.assertAlmostEqual(stats["uri_fallback_share"], 0.25)
        self.assertAlmostEqual(stats["tokens_mean"], (2 + 0 + 1 + 1) / 4)

    def test_empty_input_returns_nan(self) -> None:
        self.assertTrue(math.isnan(label_statistics(Store(), [])["chars_mean"]))


if __name__ == "__main__":
    unittest.main()
