# Reproducing the SN Computer Science revision results

Every number in the revised manuscript comes from one run of the commands below at the
tagged commit `v1.1-sncs-r1`, using `config/runtime.paper.yaml`. The submitted manuscript's
numbers come from commit `6a34d4e` (tag `v1.0-submitted`); see "Reproducing the submitted
version" at the end.

## 1. Environment

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.lock.txt
```

`requirements.lock.txt` pins every package. The reference run used CPython 3.12.16 on
macOS (Apple M4 Pro, 64 GB RAM). NLTK tokenizer and stopword data are bundled in
`resources/nltk_data`, so nothing is downloaded at run time.

### Dense models

The dense baselines download two pinned Hugging Face models on first use (about 90 MB and
400 MB): `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243…` and
`FremyCompany/BioLORD-2023` at revision `167aab52…` (full hashes in
`config/runtime.paper.yaml`). After the first run, set `HF_HUB_OFFLINE=1` to guarantee that
the cached revisions are used. Encoding runs on CPU (`device: cpu`) for determinism.

## 2. Benchmark data

The 34 alignment pairs (24 development, 5 OAEI Knowledge Graph, 5 OAEI Bio-ML) use 100
distinct RDF files, listed with sizes and SHA-256 checksums in `config/oaei_manifest.csv`.
Place them under `data/oaei_track_cache/` (or point `KGCANDGEN_DATA_ROOT` at an existing
copy), then verify:

```bash
./venv/bin/python scripts/fetch_oaei.py --data-root data/oaei_track_cache
```

The script exits non-zero if any file is missing or differs from its checksum. Rows that
carry a `url` can be fetched with `--download`.

## 3. Development tier and configuration selection

```bash
./venv/bin/python -m src.main full-run --config-path config/runtime.paper.yaml
```

This writes `results/result_<timestamp>_<sha>.csv`, the per-gold-pair
`results/query_result_<...>.csv`, the failed-run log `results/result_<...>_errors.csv`, and
the development reports (model comparison, TF-IDF and BM25 sensitivity, depth analysis)
under `results/comparisons/result_<...>/`.

```bash
./venv/bin/python -m src.main grid-diagnostics --results-csv results/result_<...>.csv
```

## 4. Configuration selection and held-out tiers (frozen settings)

`heldout-full-run` first selects the TF-IDF and BM25 settings from the development results
(lambda-penalised rule, `heldout.selection` in the config) and writes
`heldout_selected_settings.json`, printing its path as `Selected Settings JSON:`. It then runs
the KG tier with those settings and writes its reports. `heldout-class-full-run` does the same
for the Bio-ML tier.

```bash
./venv/bin/python -m src.main heldout-full-run --config-path config/runtime.paper.yaml \
    --dev-results-csv results/result_<...>.csv
./venv/bin/python -m src.main heldout-class-full-run --config-path config/runtime.paper.yaml \
    --dev-results-csv results/result_<...>.csv
```

## 5. Selection regret (post hoc, never used for selection)

```bash
./venv/bin/python -m src.main run-heldout-kg --config-path config/runtime.paper.yaml \
    --full-grid --output-csv-path results/heldout_grid_result_<label>.csv
./venv/bin/python -m src.main run-heldout-class --config-path config/runtime.paper.yaml \
    --full-grid --output-csv-path results/heldout_class_grid_result_<label>.csv
./venv/bin/python -m src.main selection-regret \
    --development-csv results/result_<...>.csv \
    --heldout-grid kg=results/heldout_grid_result_<label>.csv \
                   bioml=results/heldout_class_grid_result_<label>.csv \
    --output-csv results/comparisons/selection_regret.csv
```

## 6. Paired inference (micro and macro, clustered by dataset)

```bash
./venv/bin/python -m src.main paired-inference \
    --query-csv results/heldout_query_result_<...>.csv results/heldout_class_query_result_<...>.csv \
    --output-csv results/comparisons/paired_inference_heldout.csv
./venv/bin/python -m src.main paired-inference \
    --query-csv results/query_result_<...>.csv \
    --selected-settings-json <path printed as "Selected Settings JSON"> \
    --output-csv results/comparisons/paired_inference_development.csv
```

Deltas are `method_a - method_b`. Intervals come from a dataset-clustered bootstrap; p-values
come from an exact cluster-level sign-flip test (`p_floor` is the smallest attainable value)
and are Holm-adjusted per scope and metric.

## 7. Dataset accounting and efficiency

```bash
./venv/bin/python -m src.main dataset-accounting --config-path config/runtime.paper.yaml
./venv/bin/python -m src.main efficiency-benchmark --config-path config/runtime.paper.yaml \
    --selected-settings-json <path printed as "Selected Settings JSON">
```

The efficiency benchmark measures each method in a fresh single-threaded process; run it
with no other heavy workload on the machine.

## Determinism

- Candidate lists: ties are broken by target URI order for every method
  (`src/retrieval/topk.py`); only targets with score > 0 are candidates
  (`candidate_policy: nonzero`).
- Gold standard: every `=` target is kept and sorted (`alignment_policy: all_targets`),
  so results do not depend on parser iteration order or `PYTHONHASHSEED`.
- Bootstrap and sign-flip resampling use fixed seeds recorded in the output.

## Reproducing the submitted version

Check out `v1.0-submitted` (`6a34d4e`) with the absolute-path configuration used then, or
use the current code with:

```yaml
experiments:
  candidate_policy: padded
  alignment_policy: legacy_first
```

and `PYTHONHASHSEED=2`. With those settings the current code reproduces all 912 submitted
development rows exactly. Under `legacy_first` the target kept for a duplicated source in
`cmt-conference` depends on the hash seed, because that alignment file is parsed by rdflib.
