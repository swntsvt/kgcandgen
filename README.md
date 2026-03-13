# kgcandgen

Knowledge Graph Candidate Generation experiments for evaluating lexical retrieval methods in knowledge graph alignment.

## Getting Started

Use Python 3.12.

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
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
experiments:
  evaluation_ks: [1, 5, 10, 20, 50]
  tfidf_grid:
    - ngram_range: [1, 1]
      min_df: 1
      max_df: 1.0
      sublinear_tf: false
    - ngram_range: [1, 2]
      min_df: 1
      max_df: 1.0
      sublinear_tf: true
  bm25_grid:
    - k1: 1.5
      b: 0.75
    - k1: 1.2
      b: 0.75

datasets:
  <dataset_name>:
    track: <track_name>
    version: "<track_or_version_string>"
    source_rdf: /absolute/path/to/source.rdf
    target_rdf: /absolute/path/to/target.rdf
    alignment_rdf: /absolute/path/to/alignment.rdf
```

## Testing

Run tests:

```bash
./venv/bin/python -m unittest -v
```

Equivalent explicit discovery command:

```bash
./venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

Note: run tests in an environment where `requirements.txt` has been installed.

## CLI Usage

Use the CLI entry point to run experiments across all datasets configured in YAML:

```bash
python src/main.py
```

With explicit config path:

```bash
python src/main.py --config-path config/datasets.yaml
```

With explicit output CSV path:

```bash
python src/main.py --output-csv-path results/my_run.csv
```

With progress bars forced on:

```bash
python src/main.py --progress
```

With progress bars forced off:

```bash
python src/main.py --no-progress
```

CLI behavior:

- Runs all datasets listed in the provided config.
- Uses logging for detailed progress/errors.
- Progress bars use auto mode by default:
  - enabled in interactive terminals
  - disabled in non-interactive contexts (CI/redirected output)
- `--progress` and `--no-progress` override default progress behavior.
- Prints only the final results CSV path on success.

## Logging

Experiment logs are written to timestamped files under `logs/`:

- `logs/experiment_<timestamp>.log`

What gets logged:

- Config loading start/validation results.
- Dataset and model run phases in experiment execution.
- Per-run completion metrics and runtime.
- CSV persistence status and path.
- Final run summary (datasets processed, successes/failures, result rows, error records).

Error handling in logs:

- Failures are logged with stack traces and accumulated as error records.
- The final summary block includes total error records so run health can be checked quickly.

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

## Text Preprocessing

Use `preprocess_text(text)` from `src/preprocessing/text_preprocessor.py` to normalize labels
for lexical retrieval.

Processing order:

1. lowercase
2. camel-case splitting
3. tokenization
4. stopword removal
5. punctuation removal

Example:

```python
from src.preprocessing.text_preprocessor import preprocess_text

tokens = preprocess_text("The PlantHeightValue, of LeafSize!")
# ["plant", "height", "value", "leaf", "size"]
```

NLTK resources:

- The preprocessor checks required NLTK resources (`punkt`, `punkt_tab`, `stopwords`).
- If missing, it auto-downloads them at runtime.

## TF-IDF Retrieval

Use `TfidfRetriever` from `src/retrieval/tfidf_retriever.py` to generate top-k lexical
candidate alignments.

Public API:

- `fit(entity_ids: list[str], labels: list[str])`
- `retrieve(query_text: str, k: int) -> list[tuple[str, float]]`

Hyperparameters:

- `ngram_range`
- `min_df`
- `max_df`
- `sublinear_tf`

Defaults:

- `ngram_range=(1, 1)`
- `min_df=1`
- `max_df=1.0`
- `sublinear_tf=False`

Behavior:

- Labels and query text are preprocessed with `preprocess_text(...)`.
- Retrieval returns ranked `(entity_id, score)` pairs in descending similarity order.
- If `k` is larger than index size, all available candidates are returned.
- Scoring uses a linear kernel over L2-normalized TF-IDF vectors (cosine-equivalent).

Example:

```python
from src.retrieval.tfidf_retriever import TfidfRetriever

retriever = TfidfRetriever(ngram_range=(1, 2), min_df=1, max_df=1.0, sublinear_tf=False)
retriever.fit(
    entity_ids=["e1", "e2", "e3"],
    labels=["plant height", "leaf color", "root length"],
)
results = retriever.retrieve("plant height value", k=2)
```

## BM25 Retrieval

Use `Bm25Retriever` from `src/retrieval/bm25_retriever.py` to generate top-k lexical
candidate alignments with BM25.

Public API:

- `fit(entity_ids: list[str], labels: list[str])`
- `retrieve(query_text: str, k: int) -> list[tuple[str, float]]`

Hyperparameters:

- `k1`
- `b`

Defaults:

- `k1=1.5`
- `b=0.75`

Behavior:

- Labels and query text are preprocessed with `preprocess_text(...)`.
- Retrieval returns ranked `(entity_id, score)` pairs in descending similarity order.
- If `k` is larger than index size, all available candidates are returned.

Example:

```python
from src.retrieval.bm25_retriever import Bm25Retriever

retriever = Bm25Retriever(k1=1.5, b=0.75)
retriever.fit(
    entity_ids=["e1", "e2", "e3"],
    labels=["plant height", "leaf color", "root length"],
)
results = retriever.retrieve("plant height value", k=2)
```

## Retrieval Library Choices

The project uses different libraries for TF-IDF and BM25 intentionally:

- `scikit-learn` for TF-IDF:
  - Mature and well-tested `TfidfVectorizer`.
  - Direct control over TF-IDF hyperparameters (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`).
  - Simple sparse-matrix workflow for deterministic top-k ranking.

- `bm25s` for BM25:
  - Purpose-built BM25 implementation with direct BM25 hyperparameters (`k1`, `b`).
  - Efficient indexing/retrieval API for tokenized corpora.
  - Keeps BM25 logic explicit and separate from TF-IDF vectorization.

This separation keeps each retriever implementation clear, comparable, and easy to tune independently.

## Evaluation Metrics

Use `src/evaluation/metrics.py` to measure candidate generation quality with:

- `Recall@k`
- `MRR` (Mean Reciprocal Rank)

Input format:

- `predictions: dict[source_id, list[(target_id, score)]]`
- `gold: dict[source_id, target_id]`

Implemented functions:

- `compute_recall_at_k(predictions, gold, k) -> float`
- `compute_mrr(predictions, gold) -> float`
- `compute_recall_at_k_and_mrr(predictions, gold, k) -> tuple[float, float]`

Behavior:

- Candidate list order is treated as rank order.
- Missing prediction entries for a gold source contribute `0`.
- `Recall@k` checks whether the gold target appears in top `k`.
- `MRR` uses reciprocal rank of the first correct target (`1/rank`), else `0`.
- `compute_recall_at_k_and_mrr(...)` computes both metrics in one pass for efficiency.

Validation rules:

- `k <= 0` raises `ValueError` for `Recall@k`.
- Empty `gold` raises `ValueError` for both metrics.

Example:

```python
from src.evaluation.metrics import compute_mrr, compute_recall_at_k
from src.evaluation.metrics import compute_recall_at_k_and_mrr

predictions = {
    "s1": [("t1", 0.9), ("t2", 0.8)],
    "s2": [("t3", 0.7)],
}
gold = {
    "s1": "t2",
    "s2": "t9",
}

recall_at_1 = compute_recall_at_k(predictions, gold, k=1)
mrr = compute_mrr(predictions, gold)
recall_at_1_combined, mrr_combined = compute_recall_at_k_and_mrr(predictions, gold, k=1)
```

## Experiment Runner

Use `run_experiments(...)` from `src/experiments/experiment_runner.py` to execute retrieval
experiments across configured datasets and hyperparameter grids for both TF-IDF and BM25.

Public API:

- `run_experiments(config_path: str | Path = "config/datasets.yaml", output_csv_path: str | Path | None = None, show_progress: bool | None = None) -> list[dict]`

What it does:

1. Loads datasets and experiment settings from `config/datasets.yaml`.
2. Loads source/target RDF with strict-then-lenient RDF/XML fallback.
3. Selects only `owl:Class` entities from both graphs.
4. Extracts labels and parses/filter gold alignments.
5. Runs TF-IDF and BM25 hyperparameter configurations from YAML.
6. Computes `Recall@k` for configured `evaluation_ks` values and `MRR`.
7. Writes CSV output to a run-stamped file by default:
   `results/result_YYYYMMDD_HHMMSS_<gitsha>.csv`.
   If `output_csv_path` is provided, that exact path is used (overwrite mode).
8. Returns results in memory.
9. Shows optional tqdm progress bars when enabled.

Best-effort behavior:

- Dataset/model failures are logged and execution continues for remaining runs.
- Error records are accumulated and logged at the end of execution.
- If all runs fail, the runner logs a zero-success warning and returns an empty result list.

Result fields (per run, in-memory):

- `dataset_name`, `track`, `version`
- `model`, `hyperparameters`
- `num_source_entities`, `num_target_entities`, `num_gold_pairs`
- `recalls` (`dict[int, float]` for configured `evaluation_ks`), `mrr`, `runtime_seconds`

CSV output columns:

- `track`
- `version`
- `dataset`
- `method`
- `hyperparameters` (JSON string, sorted keys)
- `candidate_size` (`max(evaluation_ks)`)
- dynamic `recall_at_<k>` columns in the same order as `evaluation_ks`
- `mrr`
- `runtime_seconds` (per dataset+method+hyperparameter run)

Example:

```python
from src.experiments.experiment_runner import run_experiments

results = run_experiments("config/datasets.yaml")
```

Test command:

```bash
./venv/bin/python -m unittest tests/test_experiment_runner.py -v
```
