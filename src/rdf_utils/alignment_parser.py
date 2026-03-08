"""Utilities for parsing OAEI-style alignment RDF files."""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path

from pyoxigraph import Literal, NamedNode, RdfFormat, parse

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

logger = logging.getLogger(__name__)


def _iter_quads_with_fallback(path: Path):
    """Parse with strict RDF/XML first, then fallback to lenient mode."""
    try:
        yield from parse(path=str(path), format=RdfFormat.RDF_XML)
    except SyntaxError:
        logger.warning("Strict RDF/XML parse failed for %s; retrying in lenient mode", path)
        yield from parse(path=str(path), format=RdfFormat.RDF_XML, lenient=True)


def _is_absolute_http_iri(term: object) -> bool:
    return (
        isinstance(term, NamedNode)
        and (term.value.startswith("http://") or term.value.startswith("https://"))
    )


def load_alignment_mappings(alignment_path: str | Path) -> dict[str, str]:
    """Load alignment mappings from an RDF/XML file as {source_entity: target_entity}."""
    path = Path(alignment_path)
    if not path.exists():
        raise FileNotFoundError(f"Alignment file not found: {path}")

    cell_data: dict[str, dict[str, object]] = defaultdict(dict)

    for quad in _iter_quads_with_fallback(path):
        predicate = str(quad.predicate.value)
        subject_id = str(quad.subject)

        if predicate == RDF_TYPE and isinstance(quad.object, NamedNode):
            if quad.object.value.endswith("Cell"):
                cell_data[subject_id]["is_cell"] = True
            continue

        if predicate.endswith("entity1"):
            cell_data[subject_id]["entity1"] = quad.object
            continue

        if predicate.endswith("entity2"):
            cell_data[subject_id]["entity2"] = quad.object
            continue

        if predicate.endswith("relation"):
            cell_data[subject_id]["relation"] = quad.object

    mappings: dict[str, str] = {}
    skipped_invalid_entities = 0
    for fields in cell_data.values():
        if not fields.get("is_cell"):
            continue

        relation = fields.get("relation")
        entity1 = fields.get("entity1")
        entity2 = fields.get("entity2")

        if not isinstance(relation, Literal) or relation.value != "=":
            continue
        if not _is_absolute_http_iri(entity1) or not _is_absolute_http_iri(entity2):
            skipped_invalid_entities += 1
            continue
        if entity1.value in mappings:
            continue

        mappings[entity1.value] = entity2.value

    if skipped_invalid_entities:
        logger.info(
            "Skipped %d malformed alignment cells due to invalid entity IRIs",
            skipped_invalid_entities,
        )

    return mappings
