import unittest

from pyoxigraph import Literal, NamedNode, Quad, Store

from src.rdf_utils.label_extractor import extract_entity_label


class LabelExtractorTests(unittest.TestCase):
    def test_returns_rdfs_label_when_available(self) -> None:
        graph = Store()
        uri = "http://example.org/entity#A"
        graph.add(
            Quad(
                NamedNode(uri),
                NamedNode("http://www.w3.org/2000/01/rdf-schema#label"),
                Literal("Entity A"),
            )
        )

        label = extract_entity_label(graph, uri)

        self.assertEqual(label, "Entity A")

    def test_falls_back_to_skos_pref_label(self) -> None:
        graph = Store()
        uri = "http://example.org/entity#B"
        graph.add(
            Quad(
                NamedNode(uri),
                NamedNode("http://www.w3.org/2004/02/skos/core#prefLabel"),
                Literal("Preferred Label"),
            )
        )

        label = extract_entity_label(graph, uri)

        self.assertEqual(label, "Preferred Label")

    def test_uri_fragment_fallback(self) -> None:
        graph = Store()
        uri = "http://example.org/entity#EntityA"

        label = extract_entity_label(graph, uri)

        self.assertEqual(label, "EntityA")

    def test_uri_last_path_segment_fallback(self) -> None:
        graph = Store()
        uri = "http://example.org/entities/EntityA"

        label = extract_entity_label(graph, uri)

        self.assertEqual(label, "EntityA")

    def test_multiple_labels_use_lexicographic_first(self) -> None:
        graph = Store()
        uri = "http://example.org/entity#C"
        graph.add(
            Quad(
                NamedNode(uri),
                NamedNode("http://www.w3.org/2000/01/rdf-schema#label"),
                Literal("Zulu"),
            )
        )
        graph.add(
            Quad(
                NamedNode(uri),
                NamedNode("http://www.w3.org/2000/01/rdf-schema#label"),
                Literal("Alpha"),
            )
        )

        label = extract_entity_label(graph, uri)

        self.assertEqual(label, "Alpha")

    def test_priority_prefers_rdfs_label_over_skos(self) -> None:
        graph = Store()
        uri = "http://example.org/entity#D"
        graph.add(
            Quad(
                NamedNode(uri),
                NamedNode("http://www.w3.org/2000/01/rdf-schema#label"),
                Literal("RDFS Label"),
            )
        )
        graph.add(
            Quad(
                NamedNode(uri),
                NamedNode("http://www.w3.org/2004/02/skos/core#prefLabel"),
                Literal("SKOS Label"),
            )
        )

        label = extract_entity_label(graph, uri)

        self.assertEqual(label, "RDFS Label")


if __name__ == "__main__":
    unittest.main()
