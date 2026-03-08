# kgcandgen

Knowledge Graph Candidate Generation experiments for evaluating lexical retrieval methods in knowledge graph alignment.

## Getting Started

Use Python 3.12.

Create and activate a virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python3.12 -m pip install -r requirements.txt
```

## Project Structure

```text
kgcandgen/
├── config/     # Dataset configuration templates and local config
├── src/        # Source code
├── tests/      # Unit tests
├── logs/       # Experiment log files
├── results/    # Experiment outputs
└── data/       # Local datasets (ignored in git)
```

## Dataset Configuration

Create your local dataset config from the example template:

```bash
cp config/datasets.example.yaml config/datasets.yaml
```

Then edit `config/datasets.yaml` with your local RDF and alignment file paths.

Expected schema:

```yaml
datasets:
  <dataset_name>:
    track: <track_name>
    version: "<track_or_version_string>"
    source_rdf: /absolute/path/to/source.rdf
    target_rdf: /absolute/path/to/target.rdf
    alignment_rdf: /absolute/path/to/alignment.rdf
```

## Testing

Run tests with explicit discovery:

```bash
./venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

## Alignment RDF Parsing Notes

During validation on `biodiv/2018/flopo-pto/reference.rdf`, we identified a malformed IRI:

- `rdf:resource="PATO_0000103"` (not an absolute IRI)

This can cause strict RDF/XML parsing to fail with an error like:

- `No scheme found in an absolute IRI`

Fixes applied in `alignment_parser`:

1. Strict-then-lenient parse fallback for RDF/XML.
2. Tolerant predicate matching for OAEI namespace variants (with and without `#`).
3. Validation that mapped entities must be absolute HTTP(S) IRIs.
4. Malformed cells are skipped instead of aborting the full parse.

Result: real dataset parsing now succeeds and returns non-zero mappings for `flopo-pto`.
