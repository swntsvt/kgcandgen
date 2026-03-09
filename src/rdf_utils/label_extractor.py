"""Utilities for extracting entity labels from RDF graphs."""

from __future__ import annotations

from urllib.parse import urlparse

from pyoxigraph import Literal, NamedNode

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
SKOS_PREF_LABEL = "http://www.w3.org/2004/02/skos/core#prefLabel"


def _collect_literal_values(graph, subject: NamedNode, predicate_iri: str) -> list[str]:
    predicate = NamedNode(predicate_iri)
    values: list[str] = []
    for quad in graph.quads_for_pattern(subject, predicate, None, None):
        if isinstance(quad.object, Literal):
            values.append(quad.object.value)
    return values


def _uri_fallback_label(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.fragment:
        return parsed.fragment

    path = parsed.path.rstrip("/")
    if path:
        return path.split("/")[-1]

    return uri


def extract_entity_label(graph, uri: str) -> str:
    """Extract one label for a URI using priority rdfs:label > skos:prefLabel > URI fallback."""
    subject = NamedNode(uri)

    rdfs_labels = _collect_literal_values(graph, subject, RDFS_LABEL)
    if rdfs_labels:
        return sorted(rdfs_labels)[0]

    skos_labels = _collect_literal_values(graph, subject, SKOS_PREF_LABEL)
    if skos_labels:
        return sorted(skos_labels)[0]

    return _uri_fallback_label(uri)
