"""Write the Supplementary Information tables of the revised paper as LaTeX.

    python scripts/old_vs_new.py results/canonical/v1.2-sncs-r1/manifest.json ovn.csv
    python scripts/paper_supplement.py --label v1.2-sncs-r1 --old-vs-new ovn.csv \
        --output <paper>/supplementary_tables.tex

Every number comes from a file listed in ``results/canonical/<label>/manifest.json``,
except the "submitted" column of Table S1, which ``old_vs_new.py`` computes from the
v1.0-submitted result files. Tables are numbered S1-S12 in the order written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SHORT = {
    "exact_match": "Exact", "tfidf": "TF-IDF", "bm25": "BM25", "char_ngram": "Char",
    "dense_minilm": "MiniLM", "dense_biolord": "BioLORD",
    "hybrid_char_minilm": "H-MiniLM", "hybrid_char_biolord": "H-BioLORD",
}
FIXED_ORDER = list(SHORT)
KG_ABBREV = {
    "marvelcinematicuniverse-marvel": "MCU--Marvel", "memoryalpha-memorybeta": "MA--MB",
    "memoryalpha-stexpanded": "MA--STX", "starwars-swg": "SW--SWG", "starwars-swtor": "SW--SWTOR",
}
BIO_ABBREV = {
    "ncit-doid": "NCIT--DOID", "omim-ordo": "OMIM--ORDO", "snomed.body-fma.body": "SNOMED--FMA (body)",
    "snomed.neoplas-ncit.neoplas": "SNOMED--NCIT (neo.)", "snomed.pharm-ncit.pharm": "SNOMED--NCIT (pharm.)",
}
TRACK_NAME = {"biodiv": "Biodiversity", "anatomy_track": "Anatomy", "conference": "Conference"}
CATEGORY = {"valid": "valid", "reduced_scoring": "reduced scoring", "feature_loss": "feature loss", "failed": "failed"}


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
        ("BioML MRR (macro)", "Table 5", "macro", "mrr"),
        ("BioML Recall@50 (macro)", "Table 5", "macro", "recall_at_50"),
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
        "Submitted values (v1.0-submitted) against this revision (v1.2-sncs-r1) for the lexical methods. "
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
            rows.append([TRACK_NAME[t], tt(dataset), f"{int(gold[(t, dataset)]):,}"] + [f3(values[m]) for m in methods])
    return table(
        "Development MRR per pair. BM25 and TF-IDF use the selected configuration. "
        "Gold is the number of gold pairs.",
        "tab:s-dev-pairs", "@{}llrcccccc@{}",
        ["Track", "Pair", "Gold"] + [SHORT[m] for m in methods], rows, size=r"\footnotesize", midrules=midrules)


def s3_grids(diagnostics: pd.DataFrame) -> str:
    out = []
    specs = [("bm25", "BM25", "tab:s-grid-bm25"), ("tfidf", "TF-IDF", "tab:s-grid-tfidf"),
             ("char_ngram_sensitivity", "character $n$-gram", "tab:s-grid-char")]
    for method, name, label in specs:
        frame = diagnostics[diagnostics.method == method].sort_values("mean_mrr", ascending=False)
        rows = [[describe(method, r.hyperparameters), f3(r.mean_mrr), f3(r.mean_recall_at_50),
                 f"{r.mean_zero_candidate_query_share:.3f}", str(int(r.datasets_completed)), CATEGORY[r.category]]
                for r in frame.itertuples()]
        out.append(table(
            f"Every {name} grid cell on the development tier, sorted by mean MRR over pairs. "
            "``No cand.'' is the mean share of queries with no candidate; ``Pairs'' is the number of the 24 "
            "pairs on which the cell could be run.",
            label, "@{}lccccl@{}", ["Configuration", "MRR", "R@50", "No cand.", "Pairs", "Category"], rows,
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
        "tab:s-selection", "@{}lllcc@{}", ["Method", "Rule", "Picked", "KG regret (rank)", "BioML regret (rank)"],
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
                         ["Pair", "Type", "Gold"] + [SHORT[m] for m in methods], rows,
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
    return table("BioML Track MRR and Recall@50 per pair. H-: hybrid of character $n$-grams with the named encoder.",
                 "tab:s-bioml", "@{}llcccccccc@{}", ["Pair", ""] + [SHORT[m] for m in FIXED_ORDER], rows,
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
        "returned per query~($k_{\\max}=50$). ``Dense'' covers MiniLM and the hybrids, whose list lengths are "
        "identical on every unit (BioLORD and its hybrid also match MiniLM on BioML).",
        "tab:s-reduction", "@{}llrccccc@{}", ["Pair", "Type", "$|\\mathcal{T}|$", "Exact", "TF-IDF", "BM25", "Char", "Dense"],
        rows, size=r"\footnotesize", midrules=midrules)


def s8_efficiency(efficiency: pd.DataFrame) -> str:
    frame = efficiency[efficiency.target_pool_size >= 1000].copy()
    frame["pool"] = frame.apply(
        lambda r: (KG_ABBREV.get(r.dataset) or BIO_ABBREV.get(r.dataset) or tt(r.dataset))
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
                         "@{}lrcccccccc@{}", ["Pool", "$|\\mathcal{T}|$"] + [SHORT[m] for m in FIXED_ORDER], rows,
                         size=r"\scriptsize"))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--old-vs-new", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
        s3_grids(run.csv(lambda p: p.endswith("/grid_diagnostics.csv"))),
        s4_selection(run.csv(lambda p: p.endswith("/selection_regret.csv"))),
        s5_kg_units(kg, selected),
        s6_bioml(bio, selected),
        s7_reduction(kg, bio, selected),
        s8_efficiency(run.csv(lambda p: p.endswith("/efficiency.csv"))),
    ]
    args.output.write_text("\n".join(parts))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
