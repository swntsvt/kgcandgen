"""Utilities for conservative KG entity typing and partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Mapping

from pyoxigraph import NamedNode, Store

logger = logging.getLogger(__name__)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
OWL_CLASS = "http://www.w3.org/2002/07/owl#Class"
RDF_PROPERTY = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
OWL_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#ObjectProperty"
OWL_DATATYPE_PROPERTY = "http://www.w3.org/2002/07/owl#DatatypeProperty"

CLASS_TYPE_IRIS = frozenset({OWL_CLASS})
PREDICATE_TYPE_IRIS = frozenset(
    {RDF_PROPERTY, OWL_OBJECT_PROPERTY, OWL_DATATYPE_PROPERTY}
)


class EntityType(str, Enum):
    """Supported entity partitions for KG evaluation."""

    CLASS = "class"
    PREDICATE = "predicate"
    INSTANCE = "instance"


@dataclass(frozen=True)
class TypedEntityPartitions:
    """Typed entity partitions extracted from one RDF graph."""

    classes: tuple[str, ...]
    predicates: tuple[str, ...]
    instances: tuple[str, ...]
    untyped: tuple[str, ...]

    def entity_type_by_uri(self) -> dict[str, EntityType]:
        """Return a deterministic URI -> EntityType mapping for typed entities."""
        typed: dict[str, EntityType] = {}
        for uri in self.classes:
            typed[uri] = EntityType.CLASS
        for uri in self.predicates:
            typed[uri] = EntityType.PREDICATE
        for uri in self.instances:
            typed[uri] = EntityType.INSTANCE
        return typed


@dataclass(frozen=True)
class TypedAlignmentPartitions:
    """Gold alignment partitions grouped by entity type."""

    classes: dict[str, tuple[str, ...]]
    predicates: dict[str, tuple[str, ...]]
    instances: dict[str, tuple[str, ...]]
    skipped_mixed_type_pairs: int
    skipped_untyped_pairs: int


def _classify_entity(
    uri: str,
    *,
    explicit_classes: set[str],
    explicit_predicates: set[str],
) -> EntityType | None:
    # Prefer direct graph evidence over URI fallback when signals disagree.
    if uri in explicit_classes:
        return EntityType.CLASS
    if uri in explicit_predicates:
        return EntityType.PREDICATE
    if "/class/" in uri:
        return EntityType.CLASS
    if "/property/" in uri:
        return EntityType.PREDICATE
    if "/resource/" in uri:
        return EntityType.INSTANCE
    return None


def _log_entity_partition_counts(
    partitions: TypedEntityPartitions,
    *,
    graph_name: str,
) -> None:
    logger.info(
        "Typed entity partitions for %s: classes=%d predicates=%d instances=%d untyped=%d",
        graph_name,
        len(partitions.classes),
        len(partitions.predicates),
        len(partitions.instances),
        len(partitions.untyped),
    )
    if partitions.untyped:
        logger.info(
            "Untyped entities for %s: count=%d",
            graph_name,
            len(partitions.untyped),
        )
    for entity_type, values in (
        ("class", partitions.classes),
        ("predicate", partitions.predicates),
        ("instance", partitions.instances),
    ):
        if not values:
            logger.warning(
                "Empty %s partition for %s",
                entity_type,
                graph_name,
            )


def extract_typed_entity_partitions(
    store: Store,
    *,
    graph_name: str = "graph",
) -> TypedEntityPartitions:
    """Extract sorted class, predicate, instance, and untyped URI partitions."""
    explicit_classes: set[str] = set()
    explicit_predicates: set[str] = set()
    candidate_uris: set[str] = set()

    for quad in store.quads_for_pattern(None, None, None, None):
        if isinstance(quad.subject, NamedNode):
            candidate_uris.add(quad.subject.value)
        if (
            isinstance(quad.predicate, NamedNode)
            and "/property/" in quad.predicate.value
        ):
            # KG-track property entities can appear only in predicate position.
            candidate_uris.add(quad.predicate.value)
        if (
            isinstance(quad.subject, NamedNode)
            and isinstance(quad.predicate, NamedNode)
            and isinstance(quad.object, NamedNode)
            and quad.predicate.value == RDF_TYPE
        ):
            if quad.object.value in CLASS_TYPE_IRIS:
                explicit_classes.add(quad.subject.value)
            elif quad.object.value in PREDICATE_TYPE_IRIS:
                explicit_predicates.add(quad.subject.value)

    classes: list[str] = []
    predicates: list[str] = []
    instances: list[str] = []
    untyped: list[str] = []

    for uri in sorted(candidate_uris):
        entity_type = _classify_entity(
            uri,
            explicit_classes=explicit_classes,
            explicit_predicates=explicit_predicates,
        )
        if entity_type is EntityType.CLASS:
            classes.append(uri)
        elif entity_type is EntityType.PREDICATE:
            predicates.append(uri)
        elif entity_type is EntityType.INSTANCE:
            instances.append(uri)
        else:
            untyped.append(uri)

    partitions = TypedEntityPartitions(
        classes=tuple(classes),
        predicates=tuple(predicates),
        instances=tuple(instances),
        untyped=tuple(untyped),
    )
    _log_entity_partition_counts(partitions, graph_name=graph_name)
    return partitions


def partition_alignment_mappings_by_entity_type(
    mappings: Mapping[str, str | tuple[str, ...]],
    source_partitions: TypedEntityPartitions,
    target_partitions: TypedEntityPartitions,
    *,
    alignment_name: str = "alignments",
) -> TypedAlignmentPartitions:
    """Partition gold pairs into type-consistent class/predicate/instance subsets.

    ``mappings`` values may be one target URI or a tuple of target URIs; each
    (source, target) pair is classified on its own, and the skip counters count pairs.
    """
    source_types = source_partitions.entity_type_by_uri()
    target_types = target_partitions.entity_type_by_uri()
    grouped: dict[EntityType, dict[str, list[str]]] = {
        EntityType.CLASS: {},
        EntityType.PREDICATE: {},
        EntityType.INSTANCE: {},
    }
    skipped_untyped_pairs = 0
    skipped_mixed_type_pairs = 0
    for source_uri in sorted(mappings):
        raw_targets = mappings[source_uri]
        targets = (raw_targets,) if isinstance(raw_targets, str) else tuple(raw_targets)
        source_type = source_types.get(source_uri)
        for target_uri in targets:
            target_type = target_types.get(str(target_uri))
            if source_type is None or target_type is None:
                skipped_untyped_pairs += 1
                continue
            if source_type is not target_type:
                skipped_mixed_type_pairs += 1
                continue
            grouped[source_type].setdefault(source_uri, []).append(str(target_uri))

    def _freeze(values: dict[str, list[str]]) -> dict[str, tuple[str, ...]]:
        return {source: tuple(targets) for source, targets in values.items()}

    class_pairs = _freeze(grouped[EntityType.CLASS])
    predicate_pairs = _freeze(grouped[EntityType.PREDICATE])
    instance_pairs = _freeze(grouped[EntityType.INSTANCE])
    partitions = TypedAlignmentPartitions(
        classes=class_pairs,
        predicates=predicate_pairs,
        instances=instance_pairs,
        skipped_mixed_type_pairs=skipped_mixed_type_pairs,
        skipped_untyped_pairs=skipped_untyped_pairs,
    )

    logger.info(
        "Typed gold partitions for %s: classes=%d predicates=%d instances=%d skipped_mixed=%d skipped_untyped=%d",
        alignment_name,
        len(partitions.classes),
        len(partitions.predicates),
        len(partitions.instances),
        partitions.skipped_mixed_type_pairs,
        partitions.skipped_untyped_pairs,
    )
    if partitions.skipped_mixed_type_pairs:
        logger.info(
            "Skipped mixed-type gold pairs for %s: count=%d",
            alignment_name,
            partitions.skipped_mixed_type_pairs,
        )
    if partitions.skipped_untyped_pairs:
        logger.info(
            "Skipped untyped gold pairs for %s: count=%d",
            alignment_name,
            partitions.skipped_untyped_pairs,
        )
    for entity_type, values in (
        ("class", partitions.classes),
        ("predicate", partitions.predicates),
        ("instance", partitions.instances),
    ):
        if not values:
            logger.warning(
                "Empty %s gold partition for %s",
                entity_type,
                alignment_name,
            )

    return partitions
