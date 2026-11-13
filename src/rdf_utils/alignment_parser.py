"""Utilities for parsing OAEI-style alignment RDF files."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal, Mapping

from pyoxigraph import Literal as OxLiteral
from pyoxigraph import NamedNode, RdfFormat, parse
from rdflib import Graph
from rdflib.term import Literal as RdflibLiteral
from rdflib.term import URIRef

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

AlignmentPolicy = Literal["all_targets", "legacy_first"]
ALIGNMENT_POLICIES: tuple[str, ...] = ("all_targets", "legacy_first")
DEFAULT_ALIGNMENT_POLICY: AlignmentPolicy = "all_targets"

GoldTargets = dict[str, tuple[str, ...]]

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


def _parse_alignment_cells(alignment_path: str | Path) -> dict[str, dict[str, object]]:
    """Parse alignment RDF/XML into raw per-cell fields, trying stricter parsers first."""
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
    return cell_data


def _iter_equivalence_pairs(
    cell_data: dict[str, dict[str, object]],
) -> list[tuple[str, str]]:
    """Return (entity1, entity2) for valid '=' cells, in parser iteration order."""
    pairs: list[tuple[str, str]] = []
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
        pairs.append((entity1.value, entity2.value))

    if skipped_invalid_entities:
        logger.info(
            "Skipped %d malformed alignment cells due to invalid entity IRIs",
            skipped_invalid_entities,
        )
    return pairs


def load_alignment_mappings(alignment_path: str | Path) -> dict[str, str]:
    """Load alignment mappings as {source_entity: target_entity} (submitted-version behaviour).

    Keeps only the first target seen per source, in parser iteration order. That order is
    file order for the pyoxigraph backends but hash-seed dependent for the rdflib fallback,
    and deduplication happens before any entity-type filtering. Prefer
    ``load_alignment_targets(..., policy="all_targets")`` for new results.
    """
    mappings: dict[str, str] = {}
    for source, target in _iter_equivalence_pairs(_parse_alignment_cells(alignment_path)):
        if source in mappings:
            continue
        mappings[source] = target
    return mappings


def load_alignment_targets(
    alignment_path: str | Path,
    policy: AlignmentPolicy = DEFAULT_ALIGNMENT_POLICY,
) -> GoldTargets:
    """Load alignment correspondences as {source_entity: (target_entity, ...)}.

    - ``all_targets``: every distinct '=' target is kept per source; sources and targets
      are sorted, so the result does not depend on parser iteration order.
    - ``legacy_first``: the submitted-version single-target behaviour of
      ``load_alignment_mappings``, wrapped in 1-tuples.
    """
    if policy == "legacy_first":
        return {
            source: (target,)
            for source, target in load_alignment_mappings(alignment_path).items()
        }
    if policy != "all_targets":
        raise ValueError(
            f"alignment policy must be one of {list(ALIGNMENT_POLICIES)}; got {policy!r}."
        )

    grouped: dict[str, set[str]] = defaultdict(set)
    for source, target in _iter_equivalence_pairs(_parse_alignment_cells(alignment_path)):
        grouped[source].add(target)
    return {source: tuple(sorted(grouped[source])) for source in sorted(grouped)}


def filter_alignment_targets(
    gold: Mapping[str, tuple[str, ...]],
    source_entities: set[str] | frozenset[str],
    target_entities: set[str] | frozenset[str],
) -> GoldTargets:
    """Keep pairs whose source and target are both evaluated entities; drop empty sources."""
    filtered: GoldTargets = {}
    for source in sorted(gold):
        if source not in source_entities:
            continue
        kept = tuple(target for target in gold[source] if target in target_entities)
        if kept:
            filtered[source] = kept
    return filtered


def gold_pair_count(gold: Mapping[str, tuple[str, ...]]) -> int:
    """Return the number of (source, target) gold pairs."""
    return sum(len(targets) for targets in gold.values())


def multi_target_source_count(gold: Mapping[str, tuple[str, ...]]) -> int:
    """Return the number of sources with more than one gold target."""
    return sum(1 for targets in gold.values() if len(targets) > 1)
