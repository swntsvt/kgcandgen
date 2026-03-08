import tempfile
import unittest
from pathlib import Path

from src.rdf_utils.alignment_parser import load_alignment_mappings


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


if __name__ == "__main__":
    unittest.main()
