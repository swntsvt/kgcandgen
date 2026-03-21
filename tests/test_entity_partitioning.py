import unittest

from pyoxigraph import NamedNode, Quad, Store

from src.rdf_utils.entity_partitioning import (
    EntityType,
    extract_typed_entity_partitions,
    partition_alignment_mappings_by_entity_type,
)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
RDF_PROPERTY = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"
OWL_DATATYPE_PROPERTY = "http://www.w3.org/2002/07/owl#DatatypeProperty"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


def _add_named_node_quad(store: Store, subject: str, predicate: str, obj: str) -> None:
    store.add(Quad(NamedNode(subject), NamedNode(predicate), NamedNode(obj)))


class EntityPartitioningTests(unittest.TestCase):
    def test_extract_typed_entity_partitions_uses_explicit_and_uri_rules(self) -> None:
        store = Store()
        explicit_class = "http://example.org/entities/ExplicitClass"
        explicit_property = "http://example.org/entities/ExplicitProperty"
        object_property = "http://example.org/entities/ObjectProperty"
        datatype_property = "http://example.org/entities/DatatypeProperty"
        fallback_class = "http://dbkwik.webdatacommons.org/wiki/class/Character"
        fallback_property = "http://dbkwik.webdatacommons.org/wiki/property/appearsIn"
        fallback_instance = "http://dbkwik.webdatacommons.org/wiki/resource/Thor"
        untyped = "http://example.org/entities/Untyped"
        explicit_class_on_property_uri = "http://dbkwik.webdatacommons.org/wiki/property/StillAClass"
        explicit_property_on_class_uri = "http://dbkwik.webdatacommons.org/wiki/class/StillAProperty"

        _add_named_node_quad(store, explicit_class, RDF_TYPE, OWL_CLASS)
        _add_named_node_quad(store, explicit_property, RDF_TYPE, RDF_PROPERTY)
        _add_named_node_quad(store, object_property, RDF_TYPE, OWL_OBJECT_PROPERTY)
        _add_named_node_quad(store, datatype_property, RDF_TYPE, OWL_DATATYPE_PROPERTY)
        _add_named_node_quad(store, fallback_class, RDFS_LABEL, "http://example.org/value/Class")
        _add_named_node_quad(store, fallback_property, RDFS_LABEL, "http://example.org/value/Property")
        _add_named_node_quad(store, fallback_instance, RDFS_LABEL, "http://example.org/value/Instance")
        _add_named_node_quad(store, untyped, RDFS_LABEL, "http://example.org/value/Untyped")
        _add_named_node_quad(store, explicit_class_on_property_uri, RDF_TYPE, OWL_CLASS)
        _add_named_node_quad(store, explicit_property_on_class_uri, RDF_TYPE, RDF_PROPERTY)

        partitions = extract_typed_entity_partitions(store, graph_name="fixture graph")

        self.assertEqual(
            partitions.classes,
            (
                fallback_class,
                explicit_class_on_property_uri,
                explicit_class,
            ),
        )
        self.assertEqual(
            partitions.predicates,
            (
                explicit_property_on_class_uri,
                fallback_property,
                datatype_property,
                explicit_property,
                object_property,
            ),
        )
        self.assertEqual(partitions.instances, (fallback_instance,))
        self.assertEqual(partitions.untyped, (untyped,))
        self.assertEqual(
            partitions.entity_type_by_uri()[fallback_instance],
            EntityType.INSTANCE,
        )

    def test_extract_typed_entity_partitions_logs_counts_and_empty_partitions(self) -> None:
        store = Store()
        resource_uri = "http://dbkwik.webdatacommons.org/wiki/resource/Hela"
        untyped_uri = "http://example.org/entities/Untyped"
        _add_named_node_quad(store, resource_uri, RDFS_LABEL, "http://example.org/value/Resource")
        _add_named_node_quad(store, untyped_uri, RDFS_LABEL, "http://example.org/value/Untyped")

        with self.assertLogs("src.rdf_utils.entity_partitioning", level="INFO") as captured:
            partitions = extract_typed_entity_partitions(store, graph_name="source graph")

        combined = "\n".join(captured.output)
        self.assertEqual(partitions.classes, ())
        self.assertEqual(partitions.predicates, ())
        self.assertEqual(partitions.instances, (resource_uri,))
        self.assertEqual(partitions.untyped, (untyped_uri,))
        self.assertIn("Typed entity partitions for source graph", combined)
        self.assertIn("Untyped entities for source graph: count=1", combined)
        self.assertIn("Empty class partition for source graph", combined)
        self.assertIn("Empty predicate partition for source graph", combined)

    def test_partition_alignment_mappings_by_entity_type_splits_consistent_pairs(self) -> None:
        source_store = Store()
        target_store = Store()
        source_class = "http://dbkwik.webdatacommons.org/wiki/class/Character"
        source_predicate = "http://dbkwik.webdatacommons.org/wiki/property/appearsIn"
        source_instance = "http://dbkwik.webdatacommons.org/wiki/resource/Thor"
        target_class = "http://dbkwik.webdatacommons.org/other/class/Character"
        target_predicate = "http://dbkwik.webdatacommons.org/other/property/appearsIn"
        target_instance = "http://dbkwik.webdatacommons.org/other/resource/Thor"

        _add_named_node_quad(source_store, source_class, RDFS_LABEL, "http://example.org/value/Class")
        _add_named_node_quad(source_store, source_predicate, RDFS_LABEL, "http://example.org/value/Predicate")
        _add_named_node_quad(source_store, source_instance, RDFS_LABEL, "http://example.org/value/Instance")
        _add_named_node_quad(target_store, target_class, RDFS_LABEL, "http://example.org/value/Class")
        _add_named_node_quad(target_store, target_predicate, RDFS_LABEL, "http://example.org/value/Predicate")
        _add_named_node_quad(target_store, target_instance, RDFS_LABEL, "http://example.org/value/Instance")

        partitions = partition_alignment_mappings_by_entity_type(
            {
                source_instance: target_instance,
                source_class: target_class,
                source_predicate: target_predicate,
            },
            extract_typed_entity_partitions(source_store, graph_name="source"),
            extract_typed_entity_partitions(target_store, graph_name="target"),
            alignment_name="fixture alignment",
        )

        self.assertEqual(partitions.classes, {source_class: target_class})
        self.assertEqual(partitions.predicates, {source_predicate: target_predicate})
        self.assertEqual(partitions.instances, {source_instance: target_instance})
        self.assertEqual(partitions.skipped_mixed_type_pairs, 0)
        self.assertEqual(partitions.skipped_untyped_pairs, 0)

    def test_partition_alignment_mappings_skips_mixed_and_untyped_pairs(self) -> None:
        source_store = Store()
        target_store = Store()
        source_class = "http://dbkwik.webdatacommons.org/wiki/class/Character"
        source_predicate = "http://dbkwik.webdatacommons.org/wiki/property/appearsIn"
        source_unknown = "http://example.org/source/Unknown"
        target_class = "http://dbkwik.webdatacommons.org/other/class/Character"
        target_predicate = "http://dbkwik.webdatacommons.org/other/property/appearsIn"
        target_unknown = "http://example.org/target/Unknown"

        _add_named_node_quad(source_store, source_class, RDFS_LABEL, "http://example.org/value/Class")
        _add_named_node_quad(source_store, source_predicate, RDFS_LABEL, "http://example.org/value/Predicate")
        _add_named_node_quad(source_store, source_unknown, RDFS_LABEL, "http://example.org/value/Unknown")
        _add_named_node_quad(target_store, target_class, RDFS_LABEL, "http://example.org/value/Class")
        _add_named_node_quad(target_store, target_predicate, RDFS_LABEL, "http://example.org/value/Predicate")
        _add_named_node_quad(target_store, target_unknown, RDFS_LABEL, "http://example.org/value/Unknown")

        source_partitions = extract_typed_entity_partitions(source_store, graph_name="source")
        target_partitions = extract_typed_entity_partitions(target_store, graph_name="target")

        with self.assertLogs("src.rdf_utils.entity_partitioning", level="INFO") as captured:
            partitions = partition_alignment_mappings_by_entity_type(
                {
                    source_class: target_predicate,
                    source_predicate: target_class,
                    source_unknown: target_class,
                    source_class: target_unknown,
                },
                source_partitions,
                target_partitions,
                alignment_name="mixed alignment",
            )

        combined = "\n".join(captured.output)
        self.assertEqual(partitions.classes, {})
        self.assertEqual(partitions.predicates, {})
        self.assertEqual(partitions.instances, {})
        self.assertEqual(partitions.skipped_mixed_type_pairs, 1)
        self.assertEqual(partitions.skipped_untyped_pairs, 2)
        self.assertIn("Skipped mixed-type gold pairs for mixed alignment: count=1", combined)
        self.assertIn("Skipped untyped gold pairs for mixed alignment: count=2", combined)
        self.assertIn("Empty class gold partition for mixed alignment", combined)
        self.assertIn("Empty predicate gold partition for mixed alignment", combined)
        self.assertIn("Empty instance gold partition for mixed alignment", combined)

    def test_empty_partitions_are_returned_cleanly(self) -> None:
        store = Store()

        partitions = extract_typed_entity_partitions(store, graph_name="empty graph")
        gold_partitions = partition_alignment_mappings_by_entity_type(
            {},
            partitions,
            partitions,
            alignment_name="empty alignment",
        )

        self.assertEqual(partitions.classes, ())
        self.assertEqual(partitions.predicates, ())
        self.assertEqual(partitions.instances, ())
        self.assertEqual(partitions.untyped, ())
        self.assertEqual(gold_partitions.classes, {})
        self.assertEqual(gold_partitions.predicates, {})
        self.assertEqual(gold_partitions.instances, {})
        self.assertEqual(gold_partitions.skipped_mixed_type_pairs, 0)
        self.assertEqual(gold_partitions.skipped_untyped_pairs, 0)
