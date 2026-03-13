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

## Experimental Hardware

The reported experiment runs in this repository were generated on:

- Machine: Mac mini
- Chip: Apple M4 Pro
- Memory: 64 GB RAM
- OS: macOS Tahoe 26.3.1
- Python: 3.12.12
- CPU cores: 12 total (8 Performance, 4 Efficiency)
- No other heavy workloads were running concurrently

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

## Methodology

This project focuses on the **candidate generation stage** of KG alignment.

Pipeline:

```text
dataset config
  -> RDF load (source, target, alignment)
  -> class filtering (owl:Class)
  -> label extraction
  -> text preprocessing
  -> lexical retrieval (TF-IDF / BM25)
  -> metric computation (Recall@k, MRR)
  -> CSV result storage
```

Where methods fit:

- **TF-IDF** and **BM25** are the candidate retrievers.
- **Recall@k** and **MRR** are computed on ranked candidate lists produced per source entity.

Mathematical definitions (LaTeX-ready):

Full search space:

$$
|\mathcal{S}| \times |\mathcal{T}|
$$

Candidate search space after top-k retrieval:

$$
|\mathcal{S}_{eval}| \times \min(k_{\max}, |\mathcal{T}_{class}|)
$$

Gold filtering to class-level evaluation set:

$$
\mathcal{G}_f = \{(s,t) \in \mathcal{G}_{raw} \mid s \in \mathcal{S}_{class},\ t \in \mathcal{T}_{class}\}
$$

Recall@k:

$$
\text{Recall@}k = \frac{1}{|\mathcal{G}_f|} \sum_{s \in \text{dom}(\mathcal{G}_f)}
\mathbf{1}\left[g(s) \in \text{TopK}(P_s, k)\right]
$$

MRR:

$$
\text{MRR} = \frac{1}{|\mathcal{G}_f|} \sum_{s \in \text{dom}(\mathcal{G}_f)}
\begin{cases}
\frac{1}{\text{rank}_s}, & \text{if } g(s) \text{ is retrieved} \\
0, & \text{otherwise}
\end{cases}
$$

Plain-text note: candidate list order is treated as rank order; missing prediction entries for a gold source contribute `0`.

## Assumptions and Limitations

- Gold alignments are evaluated as **1:1 source->target mappings** (`dict[source] -> target`).
- If duplicate source mappings are present in alignment files, the first valid target is kept.
- Dataset paths in config are expected to be **absolute paths** and must exist.
- Source/target graph identity is inferred from dataset config naming (`dataset`, `track`, `version`) and is intentionally not duplicated as separate CSV identifier columns.
- Concurrent near-simultaneous experiment runs are out of scope (default run-stamped filenames are not designed as a multi-writer coordination mechanism).

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

Config semantics and validation:

- `experiments.evaluation_ks`:
  - non-empty list of positive integers
  - values must be unique
  - order is preserved and used for output column order
- `experiments.tfidf_grid`:
  - non-empty list
  - each entry requires `ngram_range`, `min_df`, `max_df`, `sublinear_tf`
  - entries are executed in YAML order
- `experiments.bm25_grid`:
  - non-empty list
  - each entry requires `k1`, `b`
  - entries are executed in YAML order
- `datasets.<name>`:
  - requires `track`, `version`, `source_rdf`, `target_rdf`, `alignment_rdf`
  - RDF/alignment file paths are validated for existence before run starts

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
python -m src.main
```

With explicit config path:

```bash
python -m src.main --config-path config/datasets.yaml
```

With explicit output CSV path:

```bash
python -m src.main --output-csv-path results/my_run.csv
```

With progress bars forced on:

```bash
python -m src.main --progress
```

With progress bars forced off:

```bash
python -m src.main --no-progress
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

1. Parser backend fallback chain for alignment files:
   - `pyoxigraph` strict RDF/XML
   - `pyoxigraph` lenient RDF/XML
   - `rdflib` RDF/XML fallback (only if both `pyoxigraph` attempts fail)
2. Tolerant predicate matching for OAEI namespace variants (with and without `#`).
3. Validation that mapped entities must be absolute HTTP(S) IRIs.
4. Malformed cells are skipped instead of aborting the full parse.

Additional real-world case:

- Some OAEI alignment files (for example `cmt-conference`) use non-standard XML attributes
  like non-namespaced `cid` on `Cell`, which can break `pyoxigraph` parsing.
- In those cases, `rdflib` fallback is used to recover mappings instead of failing the dataset.

Scope note:

- This fallback is currently for **alignment RDF parsing only**.
- Source/target ontology loading remains on the existing `pyoxigraph` strict-then-lenient path.

Result: dataset runs no longer fail due to these alignment parsing variants.

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

- Required English-only assets are bundled in-repo under `resources/nltk_data/`:
  - `tokenizers/punkt/english.pickle`
  - `tokenizers/punkt_tab/english/`
  - `corpora/stopwords/english`
- The preprocessor validates these local assets before use and does not download at runtime.
- The bundled path is prepended to `nltk.data.path`, so project-managed assets are used first.
- If assets are missing/corrupt, preprocessing fails fast with an actionable error that includes
  missing resource names and expected local paths.

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
- `gold_count`, `candidate_size`
- `dataset_prep_seconds` (shared per-dataset preprocessing time)
- `recalls` (`dict[int, float]` for configured `evaluation_ks`), `mrr`, `runtime_seconds`

CSV output columns:

- `track`
- `version`
- `dataset`
- `method`
- `hyperparameters` (JSON string, sorted keys)
- `gold_count` (filtered gold mappings used for evaluation)
- `candidate_size` (`min(max(evaluation_ks), num_target_entities)`)
- `dataset_prep_seconds` (dataset-level shared preprocessing)
- dynamic `recall_at_<k>` columns in the same order as `evaluation_ks`
- `mrr`
- `runtime_seconds` (model-specific index+retrieve+evaluate time)

Timing semantics:

- `dataset_prep_seconds` is shared preprocessing time computed once per dataset and copied to each model row for that dataset:
  - RDF load
  - `owl:Class` extraction
  - alignment parsing + filtering
  - label extraction + shared text preprocessing
- `runtime_seconds` is model-specific time only:
  - retriever fit/index construction
  - source retrieval loop
  - metric computation

Metadata semantics:

- `gold_count = |\mathcal{G}_f|` (filtered gold used for evaluation).
- `candidate_size = min(max(evaluation_ks), num_target_entities)`.

Exact CSV order:

`track,version,dataset,method,hyperparameters,gold_count,candidate_size,dataset_prep_seconds,recall_at_<k...>,mrr,runtime_seconds`

The `recall_at_<k>` columns appear in the same order as `evaluation_ks` in YAML.

Example:

```python
from src.experiments.experiment_runner import run_experiments

results = run_experiments("config/datasets.yaml")
```

## End-to-End Example (Config -> Run -> Result)

Minimal example config:

```yaml
experiments:
  evaluation_ks: [1, 5, 10]
  tfidf_grid:
    - ngram_range: [1, 1]
      min_df: 1
      max_df: 1.0
      sublinear_tf: false
  bm25_grid:
    - k1: 1.5
      b: 0.75

datasets:
  conference_v1:
    track: conference
    version: "1"
    source_rdf: /abs/path/source.rdf
    target_rdf: /abs/path/target.rdf
    alignment_rdf: /abs/path/alignment.rdf
```

Run:

```bash
python src/main.py --config-path config/datasets.yaml
```

CLI prints the generated file path:

```text
Results CSV: results/result_YYYYMMDD_HHMMSS_<gitsha>.csv
```

Example header + row shape:

```csv
track,version,dataset,method,hyperparameters,gold_count,candidate_size,dataset_prep_seconds,recall_at_1,recall_at_5,recall_at_10,mrr,runtime_seconds
conference,1,conference_v1,tfidf,"{\"max_df\":1.0,\"min_df\":1,\"ngram_range\":[1,1],\"sublinear_tf\":false}",120,50,1.2378,0.4417,0.7333,0.8083,0.5871,0.4216
```

Interpretation:

- `dataset_prep_seconds` should be compared across datasets (shared setup complexity).
- `runtime_seconds` should be compared across model/hyperparameter runs (retrieval/evaluation cost).
- `dataset`, `track`, `version` identify the graph pair context; source/target IDs are intentionally not duplicated as separate CSV columns.

Test command:

```bash
./venv/bin/python -m unittest tests/test_experiment_runner.py -v
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
