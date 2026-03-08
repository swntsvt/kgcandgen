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
