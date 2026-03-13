"""Utilities for parsing OAEI-style alignment RDF files."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path

from pyoxigraph import Literal as OxLiteral
from pyoxigraph import NamedNode, RdfFormat, parse
from rdflib import Graph
from rdflib.term import Literal as RdflibLiteral
from rdflib.term import URIRef

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedTerm:
    kind: str
    value: str


def _is_absolute_http_iri(term: ParsedTerm | None) -> bool:
    return (
        term is not None
        and term.kind == "iri"
        and (term.value.startswith("http://") or term.value.startswith("https://"))
    )


def _pyoxigraph_term(term: object) -> ParsedTerm:
    if isinstance(term, NamedNode):
        return ParsedTerm(kind="iri", value=term.value)
    if isinstance(term, OxLiteral):
        return ParsedTerm(kind="literal", value=term.value)
    return ParsedTerm(kind="other", value=str(term))


def _rdflib_term(term: object) -> ParsedTerm:
    if isinstance(term, URIRef):
        return ParsedTerm(kind="iri", value=str(term))
    if isinstance(term, RdflibLiteral):
        return ParsedTerm(kind="literal", value=str(term))
    return ParsedTerm(kind="other", value=str(term))


def _consume_statement(
    cell_data: dict[str, dict[str, object]],
    subject_id: str,
    predicate: str,
    obj: ParsedTerm,
) -> None:
    if predicate == RDF_TYPE and obj.kind == "iri":
        if obj.value.endswith("Cell"):
            cell_data[subject_id]["is_cell"] = True
        return

    if predicate.endswith("entity1"):
        cell_data[subject_id]["entity1"] = obj
        return

    if predicate.endswith("entity2"):
        cell_data[subject_id]["entity2"] = obj
        return

    if predicate.endswith("relation"):
        cell_data[subject_id]["relation"] = obj


def _collect_cell_data_pyoxigraph(path: Path, *, lenient: bool) -> dict[str, dict[str, object]]:
    cell_data: dict[str, dict[str, object]] = defaultdict(dict)
    for quad in parse(path=str(path), format=RdfFormat.RDF_XML, lenient=lenient):
        _consume_statement(
            cell_data=cell_data,
            subject_id=str(quad.subject),
            predicate=str(quad.predicate.value),
            obj=_pyoxigraph_term(quad.object),
        )
    return cell_data


def _collect_cell_data_rdflib(path: Path) -> dict[str, dict[str, object]]:
    graph = Graph()
    graph.parse(path, format="xml")
    cell_data: dict[str, dict[str, object]] = defaultdict(dict)
    for subject, predicate, obj in graph:
        _consume_statement(
            cell_data=cell_data,
            subject_id=str(subject),
            predicate=str(predicate),
            obj=_rdflib_term(obj),
        )
    return cell_data


def load_alignment_mappings(alignment_path: str | Path) -> dict[str, str]:
    """Load alignment mappings from an RDF/XML file as {source_entity: target_entity}."""
    path = Path(alignment_path)
    if not path.exists():
        raise FileNotFoundError(f"Alignment file not found: {path}")

    parser_backend = "pyoxigraph-strict"
    try:
        cell_data = _collect_cell_data_pyoxigraph(path, lenient=False)
    except SyntaxError as strict_exc:
        logger.warning(
            "Strict RDF/XML parse failed for %s; retrying in lenient mode", path
        )
        try:
            cell_data = _collect_cell_data_pyoxigraph(path, lenient=True)
            parser_backend = "pyoxigraph-lenient"
        except SyntaxError as lenient_exc:
            logger.warning(
                "Lenient pyoxigraph parse failed for %s; falling back to rdflib RDF/XML parser",
                path,
            )
            try:
                cell_data = _collect_cell_data_rdflib(path)
                parser_backend = "rdflib-fallback"
            except Exception as rdflib_exc:
                raise RuntimeError(
                    "Failed to parse alignment RDF/XML "
                    f"(path={path}, pyoxigraph_strict={strict_exc}, "
                    f"pyoxigraph_lenient={lenient_exc}, rdflib={rdflib_exc})"
                ) from rdflib_exc

    logger.info("Alignment parser backend for %s: %s", path, parser_backend)

    mappings: dict[str, str] = {}
    skipped_invalid_entities = 0
    for fields in cell_data.values():
        if not fields.get("is_cell"):
            continue

        relation = fields.get("relation")
        entity1 = fields.get("entity1")
        entity2 = fields.get("entity2")

        if not isinstance(relation, ParsedTerm) or relation.kind != "literal" or relation.value != "=":
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
