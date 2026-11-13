import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.rdf_utils.alignment_parser import (
    ParsedTerm,
    filter_alignment_targets,
    gold_pair_count,
    load_alignment_mappings,
    load_alignment_targets,
    multi_target_source_count,
)


def _alignment_rdf(maps_xml: str) -> str:
    return f"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#">
  <Alignment>
{maps_xml}
  </Alignment>
</rdf:RDF>
"""


class AlignmentParserTests(unittest.TestCase):
    def test_namespace_variant_without_hash_is_parsed(self) -> None:
        rdf = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment">
  <Alignment>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#A"/>
        <entity2 rdf:resource="http://example.org/target#B"/>
        <relation>=</relation>
      </Cell>
    </map>
  </Alignment>
</rdf:RDF>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.rdf"
            path.write_text(rdf, encoding="utf-8")

            mappings = load_alignment_mappings(path)

        self.assertEqual(
            mappings, {"http://example.org/source#A": "http://example.org/target#B"}
        )

    def test_happy_path_parses_equivalence_mappings(self) -> None:
        maps = """    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#A"/>
        <entity2 rdf:resource="http://example.org/target#B"/>
        <relation>=</relation>
        <measure>1.0</measure>
      </Cell>
    </map>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#C"/>
        <entity2 rdf:resource="http://example.org/target#D"/>
        <relation>=</relation>
        <measure>1.0</measure>
      </Cell>
    </map>"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.rdf"
            path.write_text(_alignment_rdf(maps), encoding="utf-8")

            mappings = load_alignment_mappings(path)

        self.assertEqual(
            mappings,
            {
                "http://example.org/source#A": "http://example.org/target#B",
                "http://example.org/source#C": "http://example.org/target#D",
            },
        )

    def test_relation_filter_keeps_only_equivalence(self) -> None:
        maps = """    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#A"/>
        <entity2 rdf:resource="http://example.org/target#B"/>
        <relation>=</relation>
      </Cell>
    </map>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#X"/>
        <entity2 rdf:resource="http://example.org/target#Y"/>
        <relation>&lt;</relation>
      </Cell>
    </map>"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.rdf"
            path.write_text(_alignment_rdf(maps), encoding="utf-8")

            mappings = load_alignment_mappings(path)

        self.assertEqual(
            mappings, {"http://example.org/source#A": "http://example.org/target#B"}
        )

    def test_duplicate_source_keeps_first_target(self) -> None:
        maps = """    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#A"/>
        <entity2 rdf:resource="http://example.org/target#First"/>
        <relation>=</relation>
      </Cell>
    </map>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#A"/>
        <entity2 rdf:resource="http://example.org/target#Second"/>
        <relation>=</relation>
      </Cell>
    </map>"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.rdf"
            path.write_text(_alignment_rdf(maps), encoding="utf-8")

            mappings = load_alignment_mappings(path)

        self.assertEqual(
            mappings,
            {"http://example.org/source#A": "http://example.org/target#First"},
        )

    def test_malformed_cells_are_skipped(self) -> None:
        maps = """    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#A"/>
        <relation>=</relation>
      </Cell>
    </map>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#Valid"/>
        <entity2 rdf:resource="http://example.org/target#Valid"/>
        <relation>=</relation>
      </Cell>
    </map>"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.rdf"
            path.write_text(_alignment_rdf(maps), encoding="utf-8")

            mappings = load_alignment_mappings(path)

        self.assertEqual(
            mappings,
            {"http://example.org/source#Valid": "http://example.org/target#Valid"},
        )

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_alignment_mappings("/tmp/does-not-exist-alignment.rdf")

    def test_malformed_absolute_iri_cell_is_skipped(self) -> None:
        maps = """    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#Good"/>
        <entity2 rdf:resource="http://example.org/target#Good"/>
        <relation>=</relation>
      </Cell>
    </map>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#Bad"/>
        <entity2 rdf:resource="PATO_0000103"/>
        <relation>=</relation>
      </Cell>
    </map>"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.rdf"
            path.write_text(_alignment_rdf(maps), encoding="utf-8")

            mappings = load_alignment_mappings(path)

        self.assertEqual(
            mappings,
            {"http://example.org/source#Good": "http://example.org/target#Good"},
        )

    def test_strict_fallback_still_returns_valid_cells(self) -> None:
        maps = """    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#First"/>
        <entity2 rdf:resource="http://example.org/target#First"/>
        <relation>=</relation>
      </Cell>
    </map>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#Second"/>
        <entity2 rdf:resource="BROKEN_ENTITY_IRI"/>
        <relation>=</relation>
      </Cell>
    </map>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#Third"/>
        <entity2 rdf:resource="http://example.org/target#Third"/>
        <relation>=</relation>
      </Cell>
    </map>"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.rdf"
            path.write_text(_alignment_rdf(maps), encoding="utf-8")

            mappings = load_alignment_mappings(path)

        self.assertEqual(
            mappings,
            {
                "http://example.org/source#First": "http://example.org/target#First",
                "http://example.org/source#Third": "http://example.org/target#Third",
            },
        )

    def test_cid_attribute_uses_rdflib_fallback(self) -> None:
        rdf = """<?xml version='1.0' encoding='utf-8'?>
<rdf:RDF xmlns='http://knowledgeweb.semanticweb.org/heterogeneity/alignment'
         xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'
         xmlns:xsd='http://www.w3.org/2001/XMLSchema#'>
  <Alignment>
    <map>
      <Cell cid='1'>
        <entity1 rdf:resource='http://example.org/source#A'/>
        <entity2 rdf:resource='http://example.org/target#B'/>
        <measure rdf:datatype='xsd:float'>1.0</measure>
        <relation>=</relation>
      </Cell>
    </map>
  </Alignment>
</rdf:RDF>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.rdf"
            path.write_text(rdf, encoding="utf-8")

            mappings = load_alignment_mappings(path)

        self.assertEqual(
            mappings, {"http://example.org/source#A": "http://example.org/target#B"}
        )

    def test_rdflib_fallback_when_pyoxigraph_fails(self) -> None:
        rdf = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment">
  <Alignment>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#A"/>
        <entity2 rdf:resource="http://example.org/target#B"/>
        <relation>=</relation>
      </Cell>
    </map>
  </Alignment>
</rdf:RDF>
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.rdf"
            path.write_text(rdf, encoding="utf-8")

            with patch(
                "src.rdf_utils.alignment_parser.parse",
                side_effect=SyntaxError("forced pyoxigraph failure"),
            ):
                mappings = load_alignment_mappings(path)

        self.assertEqual(
            mappings, {"http://example.org/source#A": "http://example.org/target#B"}
        )

    def test_both_parsers_fail_raise_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alignment.rdf"
            path.write_text("not-rdf", encoding="utf-8")

            with patch(
                "src.rdf_utils.alignment_parser.parse",
                side_effect=SyntaxError("forced pyoxigraph failure"),
            ):
                with patch(
                    "src.rdf_utils.alignment_parser.Graph.parse",
                    side_effect=SyntaxError("forced rdflib failure"),
                ):
                    with self.assertRaises(RuntimeError) as exc:
                        load_alignment_mappings(path)

        message = str(exc.exception)
        self.assertIn("Failed to parse alignment RDF/XML", message)
        self.assertIn(str(path), message)
        self.assertIn("pyoxigraph_strict=", message)
        self.assertIn("pyoxigraph_lenient=", message)
        self.assertIn("rdflib=", message)



def _cell(source: str, target: str) -> dict[str, object]:
    return {
        "is_cell": True,
        "entity1": ParsedTerm(kind="iri", value=source),
        "entity2": ParsedTerm(kind="iri", value=target),
        "relation": ParsedTerm(kind="literal", value="="),
    }


class AlignmentTargetsTests(unittest.TestCase):
    _ONE_TO_MANY = """    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#A"/>
        <entity2 rdf:resource="http://example.org/target#Z"/>
        <relation>=</relation>
      </Cell>
    </map>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#A"/>
        <entity2 rdf:resource="http://example.org/target#B"/>
        <relation>=</relation>
      </Cell>
    </map>
    <map>
      <Cell>
        <entity1 rdf:resource="http://example.org/source#C"/>
        <entity2 rdf:resource="http://example.org/target#D"/>
        <relation>=</relation>
      </Cell>
    </map>"""

    def _write(self, tmpdir: str) -> Path:
        path = Path(tmpdir) / "alignment.rdf"
        path.write_text(_alignment_rdf(self._ONE_TO_MANY), encoding="utf-8")
        return path

    def test_all_targets_keeps_every_target_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gold = load_alignment_targets(self._write(tmpdir), "all_targets")

        self.assertEqual(
            gold,
            {
                "http://example.org/source#A": (
                    "http://example.org/target#B",
                    "http://example.org/target#Z",
                ),
                "http://example.org/source#C": ("http://example.org/target#D",),
            },
        )
        self.assertEqual(gold_pair_count(gold), 3)
        self.assertEqual(multi_target_source_count(gold), 1)

    def test_legacy_first_keeps_first_parsed_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gold = load_alignment_targets(self._write(tmpdir), "legacy_first")

        self.assertEqual(gold["http://example.org/source#A"], ("http://example.org/target#Z",))

    def test_all_targets_does_not_depend_on_parser_iteration_order(self) -> None:
        cells = [
            ("c1", _cell("http://s/A", "http://t/Z")),
            ("c2", _cell("http://s/A", "http://t/B")),
            ("c3", _cell("http://s/C", "http://t/D")),
        ]
        results = []
        for ordering in (cells, list(reversed(cells))):
            with patch(
                "src.rdf_utils.alignment_parser._parse_alignment_cells",
                return_value=dict(ordering),
            ):
                results.append(load_alignment_targets("unused.rdf", "all_targets"))

        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]["http://s/A"], ("http://t/B", "http://t/Z"))

    def test_unknown_policy_raises(self) -> None:
        with self.assertRaises(ValueError):
            load_alignment_targets("unused.rdf", "first")  # type: ignore[arg-type]

    def test_filter_keeps_valid_targets_and_drops_empty_sources(self) -> None:
        gold = {"s1": ("t1", "t9"), "s2": ("t9",), "s3": ("t2",)}

        filtered = filter_alignment_targets(gold, {"s1", "s2"}, {"t1", "t2"})

        # s1 keeps its one valid target instead of being lost to a dedupe-then-filter order.
        self.assertEqual(filtered, {"s1": ("t1",)})


if __name__ == "__main__":
    unittest.main()
