"""Write the Supplementary Information tables of the revised paper as LaTeX.

    python scripts/old_vs_new.py results/canonical/v1.2-sncs-r1/manifest.json ovn.csv
    python scripts/paper_supplement.py --label v1.2-sncs-r1 --old-vs-new ovn.csv \
        --output <paper>/supplementary_tables.tex

Every number comes from a file listed in ``results/canonical/<label>/manifest.json``,
except the "submitted" column of Table S1, which ``old_vs_new.py`` computes from the
v1.0-submitted result files. Tables are numbered S1-S16 in the order written,
plus S17 from ``scripts/scaling_benchmark.py`` (``--scaling-csv``), S18 from
``scripts/selection_bootstrap.py`` (``--selection-bootstrap-csv``), an extra column in S3-S5 from
``scripts/all_source_zero_share.py`` (``--all-source-zero-share-csv``), S19, the any-gold variant, and
S20-S21, macro and micro averages of every metric, and S22, each method's mean with a
dataset-clustered interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Method names follow the paper's glossary: SHORT for row labels, HEAD for column headers.
SHORT = {
    "exact_match": "Exact match", "tfidf": "TF-IDF", "bm25": "BM25", "char_ngram": "Char $n$-gram",
    "dense_minilm": "MiniLM", "dense_biolord": "BioLORD",
    "hybrid_char_minilm": "Hybrid (MiniLM)", "hybrid_char_biolord": "Hybrid (BioLORD)",
}
HEAD = {
    **SHORT,
    "exact_match": "\\shortstack{Exact\\\\match}", "char_ngram": "\\shortstack{Char\\\\$n$-gram}",
    "hybrid_char_minilm": "\\shortstack{Hybrid\\\\(MiniLM)}",
    "hybrid_char_biolord": "\\shortstack{Hybrid\\\\(BioLORD)}",
}
FIXED_ORDER = list(SHORT)
KG_ABBREV = {
    "marvelcinematicuniverse-marvel": "MCU--Marvel", "memoryalpha-memorybeta": "MA--MB",
    "memoryalpha-stexpanded": "MA--STX", "starwars-swg": "SW--SWG", "starwars-swtor": "SW--SWTOR",
}
BIO_ABBREV = {
    "ncit-doid": "NCIT--DOID", "omim-ordo": "OMIM--ORDO", "snomed.body-fma.body": "SNOMED--FMA (Body)",
    "snomed.neoplas-ncit.neoplas": "SNOMED--NCIT (Neoplas)", "snomed.pharm-ncit.pharm": "SNOMED--NCIT (Pharm)",
}
DEV_ABBREV = {"flopo-pto": "FLOPO--PTO", "envo-sweet": "ENVO--SWEET", "mouse-human-suite": "MA--NCI"}


def pair_name(dataset: str) -> str:
    """Glossary name of a pair; Conference pairs keep their ontology names."""
    return (KG_ABBREV.get(dataset) or BIO_ABBREV.get(dataset) or DEV_ABBREV.get(dataset)
            or tt(dataset.replace("-", "--")))
TRACK_NAME = {"biodiv": "Biodiversity", "anatomy_track": "Anatomy", "conference": "Conference"}
CATEGORY = {"valid": "valid", "reduced_scoring": "reduced-scoring", "feature_loss": "feature-loss", "failed": "failed"}


def tt(text: str) -> str:
    return text.replace("_", r"\_")


def f3(value: float) -> str:
    return "--" if pd.isna(value) else f"{value:.3f}"


def table(caption: str, label: str, columns: str, header: list[str], rows: list[list[str]],
          *, size: str = r"\small", midrules: set[int] | None = None) -> str:
    midrules = midrules or set()
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             size, rf"\begin{{tabular}}{{{columns}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for index, row in enumerate(rows):
        if index in midrules:
            lines.append(r"\midrule")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def describe(method: str, hyperparameters: str) -> str:
    hp = json.loads(hyperparameters)
    if method == "bm25":
        return f"$k_1={hp['k1']:g}$, $b={hp['b']:g}$"
    if method == "tfidf":
        lo, hi = hp["ngram_range"]
        return f"$({lo},{hi})$, min\\_df $={hp['min_df']}$, sublinear {str(hp['sublinear_tf']).lower()}"
    lo, hi = hp["ngram_range"]
    return f"\\texttt{{{tt(hp['analyzer'])}}}, $[{lo},{hi}]$, min\\_df $={hp['min_df']}$"


class Run:
    def __init__(self, label: str) -> None:
        self.manifest = json.loads((ROOT / "results" / "canonical" / label / "manifest.json").read_text())
        if self.manifest.get("status") != "complete":
            raise SystemExit(f"Run {label} is not complete.")
        self.paths = list(self.manifest["artifacts_sha256"])

    def path(self, predicate) -> Path:
        matches = [p for p in self.paths if predicate(p)]
        if len(matches) != 1:
            raise SystemExit(f"Expected one matching manifest artefact, found {matches}")
        return ROOT / matches[0]

    def csv(self, predicate) -> pd.DataFrame:
        return pd.read_csv(self.path(predicate))


def selected_filter(frame: pd.DataFrame, selected: dict[str, str]) -> pd.DataFrame:
    keep = frame.method.isin([m for m in FIXED_ORDER if m not in selected])
    for method, hp in selected.items():
        canonical = frame.hyperparameters.map(lambda h: json.dumps(json.loads(h), sort_keys=True, separators=(",", ":")))
        keep |= (frame.method == method) & (canonical == hp)
    return frame[keep]


def s1_old_vs_new(ovn: pd.DataFrame) -> str:
    wanted = [
        ("Development MRR (24 pairs)", "Table 2 (dev per track)", "all 24 pairs", "MRR"),
        ("KG classes MRR", "Table 3", "class", "mrr"),
        ("KG classes Recall@50", "Table 3", "class", "recall_at_50"),
        ("KG predicates MRR", "Table 3", "predicate", "mrr"),
        ("KG predicates Recall@50", "Table 3", "predicate", "recall_at_50"),
        ("KG instances MRR", "Table 4", "instance", "mrr"),
        ("KG instances Recall@50", "Table 4", "instance", "recall_at_50"),
        ("KG macro MRR", "§6 text (aggregation)", "KG macro MRR", "MRR"),
        ("KG micro MRR", "§6 text (aggregation)", "KG micro MRR", "MRR"),
        ("Bio-ML MRR (macro)", "Table 5", "macro", "mrr"),
        ("Bio-ML Recall@50 (macro)", "Table 5", "macro", "recall_at_50"),
        ("Marvel instances Recall@50", "§6.2", "Marvel instance pool", "recall_at_50"),
        ("Dev.\\ Recall gain 20$\\to$50", "§5.1 text (depth gains)", "Recall 20->50 mean gain", "recall"),
    ]
    methods = ["Exact match", "TF-IDF", "BM25", "Char n-gram"]
    rows = []
    for name, location, quantity, metric in wanted:
        cells = [name]
        for method in methods:
            hit = ovn[ovn.location.str.startswith(location) & (ovn.quantity == quantity)
                      & (ovn.metric == metric) & (ovn.method == method)]
            if hit.empty or pd.isna(hit.old.iloc[0]):
                cells.append("--")
            else:
                cells.append(f"{float(hit.old.iloc[0]):.3f} $\\to$ {float(hit.new.iloc[0]):.3f}")
        rows.append(cells)
    return table(
        "Values from the earlier code release (tag v1.0-submitted) against this study (v1.2-sncs-r1) for the "
        "lexical methods. "
        "The changes come from three corrections: candidate lists are no longer padded with zero-score targets, "
        "all gold targets of a source are kept, and ties are broken deterministically. "
        "BM25 and TF-IDF use the selected configuration, which is the same in both versions.",
        "tab:s-old-new", "@{}lcccc@{}", ["Quantity", "Exact match", "TF-IDF", "BM25", "Char $n$-gram"], rows,
        size=r"\footnotesize")


def s2_dev_pairs(dev: pd.DataFrame, selected: dict[str, str]) -> str:
    frame = selected_filter(dev, selected)
    methods = ["exact_match", "tfidf", "bm25", "char_ngram", "dense_minilm", "hybrid_char_minilm"]
    pivot = frame.pivot_table(index=["track", "dataset"], columns="method", values="mrr")
    gold = frame.groupby(["track", "dataset"]).gold_count.first()
    rows, midrules, last = [], set(), None
    for track in ["biodiv", "anatomy_track", "conference"]:
        for (t, dataset), values in pivot.loc[[track]].iterrows():
            if last is not None and t != last:
                midrules.add(len(rows))
            last = t
            rows.append([TRACK_NAME[t], pair_name(dataset), f"{int(gold[(t, dataset)]):,}"] + [f3(values[m]) for m in methods])
    return table(
        "Development MRR per pair. BM25 and TF-IDF use the selected configuration. "
        "Gold is the number of gold pairs.",
        "tab:s-dev-pairs", "@{}llrcccccc@{}",
        ["Track", "Pair", "Gold"] + [HEAD[m] for m in methods], rows, size=r"\footnotesize", midrules=midrules)


def s3_grids(diagnostics: pd.DataFrame, all_sources: pd.DataFrame | None = None) -> str:
    """Tables S3-S5. ``all_sources`` (summary CSV of scripts/all_source_zero_share.py) adds the
    zero-candidate share over every source class, which needs no gold mappings."""
    out = []
    specs = [("bm25", "BM25", "tab:s-grid-bm25"), ("tfidf", "TF-IDF", "tab:s-grid-tfidf"),
             ("char_ngram_sensitivity", "character $n$-gram", "tab:s-grid-char")]
    share_all = {}
    if all_sources is not None:
        keys = all_sources.hyperparameters.map(lambda h: json.dumps(json.loads(h), sort_keys=True,
                                                                     separators=(",", ":")))
        share_all = dict(zip(zip(all_sources.method, keys), all_sources.mean_share_all))
    for method, name, label in specs:
        frame = diagnostics[diagnostics.method == method].sort_values("mean_mrr", ascending=False)
        rows = []
        for r in frame.itertuples():
            row = [describe(method, r.hyperparameters), f3(r.mean_mrr), f3(r.mean_recall_at_50),
                   f"{r.mean_zero_candidate_query_share:.3f}"]
            if all_sources is not None:
                key = (method, json.dumps(json.loads(r.hyperparameters), sort_keys=True, separators=(",", ":")))
                row.append(f"{share_all[key]:.3f}" if key in share_all else "--")
            rows.append(row + [str(int(r.datasets_completed)), CATEGORY[r.category]])
        header = ["Configuration", "MRR", "R@50", "No cand."]
        columns = "@{}lccccl@{}"
        caption = (f"Every {name} grid cell on the development tier, sorted by mean MRR over pairs. "
                   "``No cand.'' is the mean share of gold queries with no candidate; ")
        if all_sources is not None:
            header.append("All src.")
            columns = "@{}lcccccl@{}"
            caption += ("``All src.'' is the same share over every source class of the pair, "
                        "which needs no gold mappings; ")
        caption += "``Pairs'' is the number of the 24 pairs on which the cell could be run."
        out.append(table(caption, label, columns, header + ["Pairs", "Category"], rows,
                         size=r"\footnotesize"))
    return "\n".join(out)


def s4_selection(regret: pd.DataFrame) -> str:
    order = ["lambda=0", "lambda=0.25", "lambda=0.5", "lambda=1", "lambda=2", "mean_mrr_dataset_weighted",
             "mean_mrr_track_weighted", "worst_track_mrr", "leave_out_anatomy_track(lambda=0.5)",
             "leave_out_biodiv(lambda=0.5)", "leave_out_conference(lambda=0.5)"]
    names = {
        "mean_mrr_dataset_weighted": "Mean MRR over pairs", "mean_mrr_track_weighted": "Mean MRR over tracks",
        "worst_track_mrr": "Worst-track MRR", "leave_out_anatomy_track(lambda=0.5)": "$\\lambda=0.5$, without Anatomy",
        "leave_out_biodiv(lambda=0.5)": "$\\lambda=0.5$, without Biodiversity",
        "leave_out_conference(lambda=0.5)": "$\\lambda=0.5$, without Conference",
    }
    rows, midrules = [], set()
    for method in ["bm25", "tfidf"]:
        if rows:
            midrules.add(len(rows))
        for rule in order:
            kg = regret[(regret.tier == "kg") & (regret.rule == rule) & (regret.method == method)].iloc[0]
            bio = regret[(regret.tier == "bioml") & (regret.rule == rule) & (regret.method == method)].iloc[0]
            label = names[rule] if rule in names else f"$\\lambda={rule.split('=', 1)[1]}$"
            rows.append([SHORT[method], label, describe(method, kg.hyperparameters),
                         f"{kg.regret_macro:.4f} ({int(kg.rank_macro)}/{int(kg.configurations)})",
                         f"{bio.regret_macro:.4f} ({int(bio.rank_macro)}/{int(bio.configurations)})"])
    return table(
        "Configuration picked by each development-based selection rule, and its held-out selection regret: "
        "the post-hoc best held-out macro MRR minus that of the picked configuration, with the picked "
        "configuration's held-out rank among the configurations that ran on every pair of the tier.",
        "tab:s-selection", "@{}lllcc@{}", ["Method", "Rule", "Picked", "KG regret (rank)", "Bio-ML regret (rank)"],
        rows, size=r"\footnotesize", midrules=midrules)


def s5_kg_units(kg: pd.DataFrame, selected: dict[str, str]) -> str:
    frame = selected_filter(kg, selected)
    methods = ["exact_match", "tfidf", "bm25", "char_ngram", "dense_minilm", "hybrid_char_minilm"]
    out = []
    for metric, name, label in [("mrr", "MRR", "tab:s-kg-mrr"), ("recall_at_50", "Recall@50", "tab:s-kg-r50")]:
        pivot = frame.pivot_table(index=["dataset", "entity_type"], columns="method", values=metric)
        gold = frame.groupby(["dataset", "entity_type"]).gold_count.first()
        rows, midrules = [], set()
        for dataset in KG_ABBREV:
            if rows:
                midrules.add(len(rows))
            for entity in ["class", "predicate", "instance"]:
                values = pivot.loc[(dataset, entity)]
                rows.append([KG_ABBREV[dataset] if entity == "class" else "", entity.capitalize(),
                             f"{int(gold[(dataset, entity)]):,}"] + [f3(values[m]) for m in methods])
        out.append(table(f"KG Track {name} per held-out pair and entity type.", label, "@{}llrcccccc@{}",
                         ["Pair", "Type", "Gold"] + [HEAD[m] for m in methods], rows,
                         size=r"\footnotesize", midrules=midrules))
    return "\n".join(out)


def s6_bioml(bio: pd.DataFrame, selected: dict[str, str]) -> str:
    frame = selected_filter(bio, selected)
    rows, midrules = [], set()
    for dataset in BIO_ABBREV:
        if rows:
            midrules.add(len(rows))
        sub = frame[frame.dataset == dataset].set_index("method")
        for metric, name in [("mrr", "MRR"), ("recall_at_50", "R@50")]:
            rows.append([BIO_ABBREV[dataset] if metric == "mrr" else "", name] + [f3(sub.loc[m, metric]) for m in FIXED_ORDER])
    return table("Bio-ML Track MRR and Recall@50 per pair.",
                 "tab:s-bioml", "@{}llcccccccc@{}", ["Pair", ""] + [HEAD[m] for m in FIXED_ORDER], rows,
                 size=r"\scriptsize", midrules=midrules)


def s7_reduction(kg: pd.DataFrame, bio: pd.DataFrame, selected: dict[str, str]) -> str:
    rows, midrules = [], set()
    columns = ["exact_match", "tfidf", "bm25", "char_ngram", "dense_minilm"]
    kgf, biof = selected_filter(kg, selected), selected_filter(bio, selected)
    for dataset in KG_ABBREV:
        for entity in ["class", "predicate", "instance"]:
            sub = kgf[(kgf.dataset == dataset) & (kgf.entity_type == entity)].set_index("method")
            rows.append([KG_ABBREV[dataset] if entity == "class" else "", entity.capitalize(),
                         f"{int(sub.target_pool_size.iloc[0]):,}"] + [f3(sub.loc[m, "candidate_reduction_ratio"]) for m in columns])
    midrules.add(len(rows))
    for dataset in BIO_ABBREV:
        sub = biof[biof.dataset == dataset].set_index("method")
        rows.append([BIO_ABBREV[dataset], "Class", f"{int(sub.target_pool_size.iloc[0]):,}"]
                    + [f3(sub.loc[m, "candidate_reduction_ratio"]) for m in columns])
    return table(
        "Candidate reduction ratio $1-\\bar{c}/|\\mathcal{T}|$, from the mean number $\\bar{c}$ of candidates "
        "returned per query~($k_{\\max}=50$). ``MiniLM \\& hybrids'' covers MiniLM and the "
        "hybrids, whose list lengths are identical on every unit (BioLORD and its hybrid also match MiniLM on Bio-ML).",
        "tab:s-reduction", "@{}llrccccc@{}", ["Pair", "Type", "$|\\mathcal{T}|$", HEAD["exact_match"], "TF-IDF", "BM25", HEAD["char_ngram"], "\\shortstack{MiniLM \\\\ \\& hybrids}"],
        rows, size=r"\footnotesize", midrules=midrules)


def s8_efficiency(efficiency: pd.DataFrame) -> str:
    frame = efficiency[efficiency.target_pool_size >= 1000].copy()
    frame["pool"] = frame.apply(
        lambda r: pair_name(r.dataset)
        + ("" if r.tier != "kg" else f" ({r.entity_type})"), axis=1)
    order = frame.drop_duplicates("pool").sort_values("target_pool_size")[["pool", "target_pool_size"]]
    out = []
    for column, name, label, fmt in [
        ("build_seconds", "Index build time in seconds", "tab:s-build", lambda v: f"{v:.2f}" if v < 10 else f"{v:.1f}"),
        ("latency_median_ms", "Median query latency in milliseconds", "tab:s-latency",
         lambda v: f"{v:.3f}" if v < 1 else f"{v:.2f}" if v < 10 else f"{v:.1f}"),
    ]:
        pivot = frame.pivot_table(index="pool", columns="method", values=column)
        rows = [[pool, f"{int(size):,}"] + [("--" if pd.isna(pivot.loc[pool].get(m)) else fmt(pivot.loc[pool, m]))
                                            for m in FIXED_ORDER] for pool, size in order.itertuples(index=False)]
        out.append(table(f"{name}, single-threaded on CPU, for every pool of at least 1,000 targets.", label,
                         "@{}lrcccccccc@{}", ["Pool", "$|\\mathcal{T}|$"] + [HEAD[m] for m in FIXED_ORDER], rows,
                         size=r"\scriptsize"))
    return "\n".join(out)


def _unit_name(row) -> str:
    return pair_name(row.dataset)


def _accounting_rows(accounting: pd.DataFrame, cells) -> tuple[list[list[str]], set[int]]:
    rows, midrules, last_tier, last_dataset = [], set(), None, None
    for r in accounting.itertuples():
        if last_tier is not None and r.tier != last_tier:
            midrules.add(len(rows))
        same = r.dataset == last_dataset
        last_tier, last_dataset = r.tier, r.dataset
        rows.append(["" if same else _unit_name(r), r.entity_type.capitalize()] + cells(r))
    return rows, midrules


def s9_accounting(accounting: pd.DataFrame) -> str:
    def n(v) -> str:
        return "0" if pd.isna(v) else f"{int(v):,}"

    first = ~accounting.dataset.eq(accounting.dataset.shift())
    many_to_one = {i: (n(v) if f else "") for i, v, f in
                   zip(accounting.index, accounting.many_to_one_targets, first)}
    accounting = accounting.assign(m21_shown=[many_to_one[i] for i in accounting.index])
    rows, midrules = _accounting_rows(accounting, lambda r: [
        n(r.source_entities), n(r.target_entities), n(r.eval_queries), n(r.gold_pairs),
        n(r.multi_target_queries), r.m21_shown, n(r.excluded_pairs)])
    return table(
        "Accounting per evaluation unit. $|\\mathcal{S}|$ and $|\\mathcal{T}|$ are the source and target "
        "entities of the evaluated type; ``Queries'' are the source entities with at least one gold target; "
        "``Gold'' the gold pairs evaluated; ``Multi'' the queries with more than one gold target; ``M:1'' the "
        "targets of the pair's reference alignment that are matched by more than one source (given once per "
        "pair; each such source is queried and scored independently); ``Excl.'' the "
        "equivalence pairs of the reference alignment that were excluded. All exclusions are development pairs "
        "whose source entity is not a named \\texttt{owl:Class}; no KG pair is excluded, and no reference "
        "alignment contains non-equivalence cells.",
        "tab:s-accounting", "@{}llrrrrrrr@{}",
        ["Pair", "Type", "$|\\mathcal{S}|$", "$|\\mathcal{T}|$", "Queries", "Gold", "Multi", "M:1", "Excl."],
        rows, size=r"\scriptsize", midrules=midrules)


def s10_labels(accounting: pd.DataFrame) -> str:
    def pct(v) -> str:
        return f"{100 * v:.0f}"

    rows, midrules = _accounting_rows(accounting, lambda r: [
        f"{r.query_label_chars_median:.0f}", f"{r.query_label_tokens_median:.0f}", f"{r.query_label_tokens_p90:.0f}",
        pct(r.query_label_single_token_share), pct(r.query_label_uri_fallback_share),
        f"{r.target_label_chars_median:.0f}", f"{r.target_label_tokens_median:.0f}",
        pct(r.target_label_single_token_share), pct(r.target_label_uri_fallback_share)])
    return table(
        "Label statistics per evaluation unit, for query (source) labels and target labels: median length in "
        "characters, median and 90th-percentile number of tokens after the token pipeline, percentage of "
        "single-token labels, and percentage of labels taken from the URI because no \\texttt{rdfs:label} or "
        "\\texttt{skos:prefLabel} exists.",
        "tab:s-labels", "@{}llrrrrrrrrr@{}",
        ["Pair", "Type", "Q chars", "Q tok.", "Q tok.\\ p90", "Q 1-tok.\\,\\%", "Q URI\\,\\%",
         "T chars", "T tok.", "T 1-tok.\\,\\%", "T URI\\,\\%"],
        rows, size=r"\scriptsize\setlength{\tabcolsep}{2.5pt}", midrules=midrules)


def s11_provenance(config_path: Path, manifest_path: Path) -> str:
    import yaml

    config = yaml.safe_load(config_path.read_text())
    files = pd.read_csv(manifest_path)
    files["prefix"] = files.relative_path.str.split("/").str[:3].str.join("/")
    datasets = {**config["development_datasets"], **config["heldout_datasets"],
                **config["heldout_secondary_datasets"]}
    tracks: dict[tuple[str, str], int] = {}
    for spec in datasets.values():
        tracks[(spec["track"], str(spec["version"]))] = tracks.get((spec["track"], str(spec["version"])), 0) + 1
    rows = []
    for (track, version), pairs in tracks.items():
        prefix = next(p for p in files.prefix.unique() if p.split("/")[1] == track and p.split("/")[2] == version)
        sub = files[files.prefix == prefix]
        dates = ", ".join(sorted(set(sub.local_file_mtime_utc)))
        rows.append([tt(track), tt(version), str(pairs), str(len(sub)), f"{sub.bytes.sum() / 1e6:,.1f}",
                     f"\\texttt{{{tt(prefix.split('/', 1)[1])}/}}", dates])
    return table(
        "Benchmark provenance. Files were obtained from the OAEI test-data repository hosted at "
        "\\texttt{oaei.webdatacommons.org}, in the layout shown, and verified by SHA-256 against "
        "\\texttt{config/oaei\\_manifest.csv} before every run. ``Date'' is the date of the local copy, i.e.\\ "
        "of access.",
        "tab:s-provenance", "@{}llrrrll@{}",
        ["Track", "Version", "Pairs", "Files", "MB", "Path prefix", "Date"], rows, size=r"\scriptsize")


def s12_adaptive_k(dev_q: pd.DataFrame, kg_q: pd.DataFrame, accounting: pd.DataFrame,
                   selected: dict[str, str]) -> str:
    """Recall at a pool-relative depth on the small schema pools (reviewer request R1-6).

    Gold ranks are stored to depth 50, so k = ceil(0.3|T|) is capped at 50; on units with
    more than 166 targets the 0.3|T| column is therefore R@50, a lower bound.
    """
    import math

    pools = {(r.dataset, r.entity_type): r.target_entities for r in accounting.itertuples()}
    groups = [("Development: Conference classes", dev_q[dev_q.track == "conference"]),
              ("KG Track: classes", kg_q[kg_q.entity_type == "class"]),
              ("KG Track: predicates", kg_q[kg_q.entity_type == "predicate"])]
    rows: list[list[str]] = []
    midrules: set[int] = set()
    for name, frame in groups:
        frame = selected_filter(frame, selected).copy()
        frame["T"] = [pools[(d, e)] for d, e in zip(frame.dataset, frame.entity_type)]
        frame["k10"] = frame["T"].map(lambda t: math.ceil(0.1 * t))
        frame["k30"] = frame["T"].map(lambda t: min(math.ceil(0.3 * t), 50))
        if (frame.k10 > 50).any():
            raise SystemExit("0.1|T| exceeds the stored rank depth")
        rank = frame.gold_rank
        hit = lambda k: ((rank >= 1) & (rank <= k)).groupby(frame.method).mean()  # noqa: E731
        r5, r10, r30, r50 = hit(5), hit(frame.k10), hit(frame.k30), hit(50)
        units = frame.drop_duplicates(["dataset", "entity_type"])
        t = units["T"]
        capped = int((units["T"].map(lambda v: math.ceil(0.3 * v)) > 50).sum())
        if rows:
            midrules.add(len(rows))
        rows.append([f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{name}}}: {len(units)} units, $|\\mathcal{{T}}|$ "
                     f"{t.min()}--{t.max()}, median $k$ {units.k10.median():.0f} / {units.k30.median():.0f}"
                     f"{f', {capped} capped' if capped else ''}}}"])
        rows.append(["\\quad Pool reduction", "--",
                     f3((1 - units.k10 / t).mean()), f3((1 - units.k30 / t).mean()),
                     f3((1 - t.clip(upper=50) / t).mean())])
        for method in [m for m in FIXED_ORDER if m in r50.index]:
            rows.append([f"\\quad {SHORT[method]}", f3(r5[method]), f3(r10[method]), f3(r30[method]), f3(r50[method])])
    return table(
        "Recall at a pool-relative candidate depth on the small schema pools. Recall is pair-level and micro-averaged "
        "over the units of each group; lexical methods use their selected configuration. $k_{0.1}=\\lceil "
        "0.1|\\mathcal{T}|\\rceil$ and $k_{0.3}=\\lceil 0.3|\\mathcal{T}|\\rceil$ are set per unit. Ranks are "
        "stored to depth 50, so $k_{0.3}$ is capped at 50 on units with more than 166 targets (``capped''); "
        "there it is a lower bound. ``Pool reduction'' is the mean share of the target pool excluded by a list of "
        "that length.",
        "tab:s-adaptive-k", "@{}lrrrr@{}",
        ["", "R@5", "R@$k_{0.1}$", "R@$k_{0.3}$", "R@50"], rows, size=r"\small", midrules=midrules)


def s13_scaling(scaling: pd.DataFrame) -> str:
    """Marvel instance queries against growing target pools (reviewer request R1-5)."""
    methods = [m for m in FIXED_ORDER if m in set(scaling.method)]
    sizes = sorted(scaling.target_pool_size.unique())
    pooled = int(scaling.loc[scaling.pool == "pooled_kg", "target_pool_size"].max())
    full = int(scaling.loc[scaling.pool == "marvel_full", "target_pool_size"].max())
    value = scaling.set_index(["target_pool_size", "method"])

    def name(size: int) -> str:
        return f"{size:,}" + (" (full)" if size == full else " (pooled)" if size == pooled else "")

    panels = [("Build time (s)", "build_seconds", lambda v: f"{v:.1f}"),
              ("Median query latency (ms)", "latency_median_ms", lambda v: f"{v:.3f}" if v < 1 else f"{v:.1f}"),
              ("Recall@50", "recall_at_50", f3),
              ("MRR@50", "mrr", f3)]
    rows: list[list[str]] = []
    midrules: set[int] = set()
    for title, column, fmt in panels:
        if rows:
            midrules.add(len(rows))
        rows.append([f"\\multicolumn{{{len(methods) + 1}}}{{@{{}}l}}{{\\emph{{{title}}}}}"])
        for size in sizes:
            rows.append([f"\\quad {name(size)}"] + [fmt(value.loc[(size, m), column]) for m in methods])
    queries = int(scaling.query_count.iloc[0])
    return table(
        f"Scaling on the Marvel instance task. The {queries:,} Marvel instance queries are run against nested "
        "subsamples of the Marvel target pool (every gold target plus a seeded sample of the others), the full "
        f"pool, and a pooled set of {pooled:,} targets that adds the instance pools of the other four KG Track "
        "pairs. Single-threaded on CPU with the protocol of Tables~\\ref{tab:s-build}--\\ref{tab:s-latency}; "
        "Recall and MRR are pair-level.",
        "tab:s-scaling", "@{}l" + "r" * len(methods) + "@{}",
        [""] + [HEAD[m] for m in methods], rows, size=r"\small", midrules=midrules)


def s14_selection_bootstrap(boot: pd.DataFrame, resamples: int) -> str:
    """How often each configuration is selected when development pairs are resampled (R2-2)."""
    rows: list[list[str]] = []
    midrules: set[int] = set()
    for method in ("bm25", "tfidf"):
        sub = boot[boot.method == method].sort_values("picks", ascending=False)
        if rows:
            midrules.add(len(rows))
        for r in sub.itertuples():
            label = SHORT[method] if not rows or len(rows) in midrules else ""
            mark = "$^\\ast$" if r.full_sample_pick else ""
            rows.append([label, describe(method, r.hyperparameters) + mark, f"{100 * r.share:.1f}",
                         CATEGORY[r.category], f3(r.development_mean_mrr),
                         f3(r.kg_regret_macro), f3(r.bioml_regret_macro)])
    return table(
        f"Uncertainty of the selected configuration. Development pairs are resampled with replacement within "
        f"each track ({resamples:,} resamples) and the stability-aware rule ($\\lambda=0.5$) is applied to each "
        "resample. The table lists every configuration picked at least once, how often it was picked, its "
        "diagnostic category, its full-sample development MRR, and its held-out macro-MRR regret against the "
        "post-hoc best on each held-out tier. $^\\ast$: the configuration selected on the full development "
        "tier. Anatomy has one development pair and Biodiversity two, so most of the resampling variation "
        "comes from the 21 Conference pairs.",
        "tab:s-selection-boot", "@{}llrlrrr@{}",
        ["", "Configuration", "Picked\\,\\%", "Category", "Dev.\\ MRR", "KG regret", "Bio-ML regret"],
        rows, size=r"\small", midrules=midrules)


def s15_any_gold(dev: pd.DataFrame, kg: pd.DataFrame, bio: pd.DataFrame, selected: dict[str, str]) -> str:
    """Pair-level metrics against the any-gold-target variant (R1-4)."""
    rows: list[list[str]] = []
    midrules: set[int] = set()
    for name, frame in [("Development", dev), ("KG Track", kg), ("Bio-ML Track", bio)]:
        frame = selected_filter(frame, selected)
        means = frame.groupby("method")[["recall_at_50", "any_gold_recall_at_50", "mrr", "any_gold_mrr"]].mean()
        if rows:
            midrules.add(len(rows))
        keys = ["dataset", "entity_type"] if "entity_type" in frame else ["dataset"]
        multi = int(frame.drop_duplicates(keys).multi_target_query_count.sum())
        rows.append([f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{name}}}: {multi:,} multi-target queries}}"])
        for m in [m for m in FIXED_ORDER if m in means.index]:
            r = means.loc[m]
            rows.append([f"\\quad {SHORT[m]}", f3(r.recall_at_50), f3(r.any_gold_recall_at_50), f3(r.mrr),
                         f3(r.any_gold_mrr)])
    return table(
        "Pair-level metrics, used throughout the paper, against the any-gold variant, in which a query with "
        "several gold targets counts as a hit if any of them is retrieved and its reciprocal rank is that of the "
        "best-ranked one. Macro averages over evaluation units; lexical methods use their selected configuration.",
        "tab:s-any-gold", "@{}lrrrr@{}",
        ["", "R@50", "R@50 any", "MRR", "MRR any"], rows, size=r"\small", midrules=midrules)


METRIC_COLS = ["recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20", "recall_at_50", "mrr"]


def _micro_macro_rows(frame: pd.DataFrame) -> list[list[str]]:
    rows = []
    for m in [m for m in FIXED_ORDER if m in set(frame.method)]:
        sub = frame[frame.method == m]
        macro = sub[METRIC_COLS].mean()
        micro = (sub[METRIC_COLS].multiply(sub.gold_count, axis=0).sum() / sub.gold_count.sum())
        rows.append([SHORT[m], "macro"] + [f3(macro[c]) for c in METRIC_COLS])
        rows.append(["", "micro"] + [f3(micro[c]) for c in METRIC_COLS])
    return rows


def s16_micro_macro(dev: pd.DataFrame, kg: pd.DataFrame, bio: pd.DataFrame, selected: dict[str, str]) -> str:
    """Macro and micro averages of every metric on every tier (R2-7)."""
    header = ["", "", "R@1", "R@5", "R@10", "R@20", "R@50", "MRR"]
    out = []
    for label, caption, groups in [
        ("tab:s-micro-macro-dev-bio",
         "Macro~(mean over evaluation units) and micro~(mean over gold pairs) averages of every metric on the "
         "development and Bio-ML tiers. Lexical methods use their selected configuration.",
         [("Development (24 pairs, 2,668 gold pairs)", dev), ("Bio-ML Track (5 pairs, 17,689 gold pairs)", bio)]),
        ("tab:s-micro-macro-kg",
         "Macro and micro averages of every metric on the KG Track, over all 15 evaluation units and per entity "
         "type. Instances supply 15,127 of the 15,357 gold pairs, so the overall micro average follows them.",
         [("All units", kg), ("Classes", kg[kg.entity_type == "class"]),
          ("Predicates", kg[kg.entity_type == "predicate"]), ("Instances", kg[kg.entity_type == "instance"])]),
    ]:
        rows: list[list[str]] = []
        midrules: set[int] = set()
        for name, frame in groups:
            if rows:
                midrules.add(len(rows))
            rows.append([f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{name}}}}}"])
            rows += _micro_macro_rows(selected_filter(frame, selected))
        out.append(table(caption, label, "@{}llrrrrrr@{}", header, rows, size=r"\scriptsize", midrules=midrules))
    return "\n".join(out)


def _clustered_mean_ci(frame: pd.DataFrame, column: str, *, resamples: int = 10_000,
                       seed: int = 20261114) -> tuple[float, float, float]:
    """Macro mean over evaluation units with a dataset-clustered percentile bootstrap interval.

    Whole datasets are resampled with replacement; a KG dataset contributes all of its entity
    types together, as in the paired inference.
    """
    import numpy as np

    by_dataset = frame.groupby("dataset")[column].agg(["sum", "count"])
    sums, counts = by_dataset["sum"].to_numpy(), by_dataset["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(resamples, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(sums.sum() / counts.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def s17_mean_ci(dev: pd.DataFrame, kg: pd.DataFrame, bio: pd.DataFrame, selected: dict[str, str]) -> str:
    """Each method's mean with a dataset-clustered 95% interval, per tier (minor comment m9)."""
    rows: list[list[str]] = []
    midrules: set[int] = set()
    for name, frame in [("Development (24 pairs)", dev), ("KG Track, all units (5 datasets)", kg),
                        ("KG classes", kg[kg.entity_type == "class"]),
                        ("KG predicates", kg[kg.entity_type == "predicate"]),
                        ("KG instances", kg[kg.entity_type == "instance"]), ("Bio-ML Track (5 pairs)", bio)]:
        frame = selected_filter(frame, selected)
        if rows:
            midrules.add(len(rows))
        rows.append([f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{name}}}}}"])
        for m in [m for m in FIXED_ORDER if m in set(frame.method)]:
            sub = frame[frame.method == m]
            cells = []
            for column in ("mrr", "recall_at_50"):
                mean, lo, hi = _clustered_mean_ci(sub, column)
                cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            rows.append([f"\\quad {SHORT[m]}"] + cells)
    return table(
        "Mean MRR and Recall@50 of every method with dataset-clustered 95\\% bootstrap intervals~(10,000 "
        "resamples of whole pairs). Means are macro averages over evaluation units, as in Tables~2--5 of the "
        "paper; lexical methods use their selected configuration. With five pairs per held-out tier the "
        "intervals are approximate and probably too narrow; paired differences between methods, which are "
        "more precise because both methods answer the same queries, are in Table~6 of the paper.",
        "tab:s-mean-ci", "@{}lcc@{}", ["", "MRR [95\\% CI]", "Recall@50 [95\\% CI]"], rows,
        size=r"\small", midrules=midrules)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--old-vs-new", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scaling-csv", type=Path, default=None,
                        help="Output of scripts/scaling_benchmark.py; adds Table S17.")
    parser.add_argument("--selection-bootstrap-csv", type=Path, default=None,
                        help="Output of scripts/selection_bootstrap.py; adds Table S18.")
    parser.add_argument("--all-source-zero-share-csv", type=Path, default=None,
                        help="Summary CSV of scripts/all_source_zero_share.py; adds a column to Tables S3-S5.")
    args = parser.parse_args()
    run = Run(args.label)
    selected_path = run.path(lambda p: p.endswith("heldout_selected_settings.json"))
    selected = {m: e["hyperparameters_json"] for m, e in json.loads(selected_path.read_text())["selected_settings"].items()}
    dev = run.csv(lambda p: Path(p).name.startswith("result_") and p.endswith(".csv") and "_errors" not in p)
    kg = run.csv(lambda p: Path(p).name.startswith("heldout_result_") and "_errors" not in p)
    bio = run.csv(lambda p: Path(p).name.startswith("heldout_class_result_") and "_errors" not in p)
    parts = [
        f"% Generated by scripts/paper_supplement.py from run {args.label}; do not edit by hand.",
        s1_old_vs_new(pd.read_csv(args.old_vs_new)),
        s2_dev_pairs(dev, selected),
        s3_grids(run.csv(lambda p: p.endswith("/grid_diagnostics.csv")),
                 pd.read_csv(args.all_source_zero_share_csv) if args.all_source_zero_share_csv else None),
        s4_selection(run.csv(lambda p: p.endswith("/selection_regret.csv"))),
        s5_kg_units(kg, selected),
        s6_bioml(bio, selected),
        s7_reduction(kg, bio, selected),
        s8_efficiency(run.csv(lambda p: p.endswith("/efficiency.csv"))),
        s9_accounting(run.csv(lambda p: p.endswith("/dataset_accounting.csv"))),
        s10_labels(run.csv(lambda p: p.endswith("/dataset_accounting.csv"))),
        s11_provenance(ROOT / run.manifest["config"], ROOT / "config" / "oaei_manifest.csv"),
        s12_adaptive_k(run.csv(lambda p: Path(p).name.startswith("query_result_")),
                       run.csv(lambda p: Path(p).name.startswith("heldout_query_result_")),
                       run.csv(lambda p: p.endswith("/dataset_accounting.csv")), selected),
    ]
    if args.scaling_csv is not None:
        parts.append(s13_scaling(pd.read_csv(args.scaling_csv)))
    if args.selection_bootstrap_csv is not None:
        meta = json.loads(args.selection_bootstrap_csv.with_suffix(".json").read_text())
        parts.append(s14_selection_bootstrap(pd.read_csv(args.selection_bootstrap_csv), meta["resamples"]))
    parts.append(s15_any_gold(dev, kg, bio, selected))
    parts.append(s16_micro_macro(dev, kg, bio, selected))
    parts.append(s17_mean_ci(dev, kg, bio, selected))
    args.output.write_text("\n".join(parts))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
