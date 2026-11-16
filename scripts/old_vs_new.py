"""Old (submitted, v1.0-submitted / 6a34d4e) vs new (a canonical run) for every table and key claim.

Usage: python old_vs_new.py <canonical manifest.json> <output.csv>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
OLD = {
    "dev": R / "result_20260326_110453_6a34d4e.csv",
    "kg": R / "heldout_result_20260326_110937_6a34d4e.csv",
    "bio": R / "heldout_class_result_20260326_115627_6a34d4e.csv",
    "selected": R / "comparisons/result_20260326_110453_6a34d4e/heldout_selected_settings.json",
    "kg_inference": R / "comparisons/heldout_result_20260326_110937_6a34d4e/kg_heldout_pairwise_overall_inference.csv",
    "kg_inference_type": R / "comparisons/heldout_result_20260326_110937_6a34d4e/kg_heldout_pairwise_by_type_inference.csv",
}
METRICS = ["recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20", "recall_at_50", "mrr"]
LABEL = {"exact_match": "Exact match", "tfidf": "TF-IDF", "bm25": "BM25", "char_ngram": "Char n-gram",
         "dense_minilm": "Dense MiniLM", "dense_biolord": "Dense BioLORD",
         "hybrid_char_minilm": "Hybrid char+MiniLM", "hybrid_char_biolord": "Hybrid char+BioLORD"}
ORDER = ["exact_match", "tfidf", "bm25", "char_ngram", "dense_minilm", "dense_biolord", "hybrid_char_minilm", "hybrid_char_biolord"]


def canon(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(hyperparameters=frame.hyperparameters.map(
        lambda h: json.dumps(json.loads(h), sort_keys=True, separators=(",", ":"))))


def load(paths: dict[str, Path]) -> dict[str, object]:
    payload = json.loads(Path(paths["selected"]).read_text())
    selected = {m: e["hyperparameters_json"] for m, e in payload["selected_settings"].items()}
    data = {k: canon(pd.read_csv(paths[k])) for k in ("dev", "kg", "bio")}
    data["selected"] = selected
    return data


def pick(frame: pd.DataFrame, selected: dict[str, str]) -> pd.DataFrame:
    keep = ~frame.method.isin(["tfidf", "bm25", "char_ngram_sensitivity"])
    for method, hp in selected.items():
        keep |= (frame.method == method) & (frame.hyperparameters == hp)
    return frame[keep]


def quantities(data: dict[str, object]) -> dict[tuple, float]:
    q: dict[tuple, float] = {}
    dev, kg, bio, sel = data["dev"], data["kg"], data["bio"], data["selected"]
    sdev, skg, sbio = pick(dev, sel), pick(kg, sel), pick(bio, sel)
    # Table 1: dataset summary
    for name, g in dev.groupby("dataset"):
        q[("Table 1 (dataset summary)", f"dev {name}", "gold", "")] = g.gold_count.iloc[0]
    for (name, et), g in kg.groupby(["dataset", "entity_type"]):
        q[("Table 1 (dataset summary)", f"KG {name} {et}", "gold", "")] = g.gold_count.iloc[0]
        q[("Table 1 (dataset summary)", f"KG {name} {et}", "target pool", "")] = g.target_pool_size.iloc[0]
    for name, g in bio.groupby("dataset"):
        q[("Table 1 (dataset summary)", f"BioML {name}", "gold", "")] = g.gold_count.iloc[0]
    # Fig. 1 / Figs 2-3: grid means
    for (method, hp), g in dev[dev.method.isin(["bm25", "tfidf"])].groupby(["method", "hyperparameters"]):
        q[(f"Fig. {'1 (BM25' if method == 'bm25' else '2-3 (TF-IDF'} grid)", hp, "mean MRR", method)] = g.mrr.mean()
    for method in ("bm25", "tfidf"):
        grid = dev[dev.method == method].groupby("hyperparameters").mrr.mean()
        q[("§5.1-5.2 / §8 text", "grid span (max - min mean MRR)", "MRR", method)] = grid.max() - grid.min()
        q[("§5.1-5.2 / §8 text", "selected config mean MRR (24 dev pairs)", "MRR", method)] = (
            sdev[sdev.method == method].mrr.mean())
    # §5.1 depth gains
    ks = [1, 5, 10, 20, 50]
    for method, g in sdev.groupby("method"):
        for a, b in zip(ks, ks[1:]):
            q[("§5.1 text (depth gains)", f"Recall {a}->{b} mean gain", "recall", method)] = (
                g[f"recall_at_{b}"] - g[f"recall_at_{a}"]).mean()
    # Table 2: dev per track
    for (track, method), g in sdev.groupby(["track", "method"]):
        q[("Table 2 (dev per track)", track, "MRR", method)] = g.mrr.mean()
    for method, g in sdev.groupby("method"):
        q[("Table 2 (dev per track)", "all 24 pairs", "MRR", method)] = g.mrr.mean()
    # Tables 3-5: held-out
    for et, g in skg.groupby("entity_type"):
        table = "Table 3 (KG classes/predicates)" if et != "instance" else "Table 4 (KG instances)"
        for method, m in g.groupby("method"):
            for metric in METRICS:
                q[(table, et, metric, method)] = m[metric].mean()
    for method, m in sbio.groupby("method"):
        for metric in METRICS:
            q[("Table 5 (BioML)", "macro", metric, method)] = m[metric].mean()
    for tier, frame in (("KG", skg), ("BioML", sbio)):
        for method, m in frame.groupby("method"):
            q[("§6 text (aggregation)", f"{tier} macro MRR", "MRR", method)] = m.mrr.mean()
            q[("§6 text (aggregation)", f"{tier} micro MRR", "MRR", method)] = (m.mrr * m.gold_count).sum() / m.gold_count.sum()
    for method, m in skg[skg.dataset == "marvelcinematicuniverse-marvel"].groupby("method"):
        q[("§6.2 / §6.4 / §8 text", "Marvel instance pool", "recall_at_50", method)] = (
            m[m.entity_type == "instance"].recall_at_50.iloc[0])
    # Table 6: candidate reduction
    for (name, et, method), g in skg.groupby(["dataset", "entity_type", "method"]):
        q[("Table 6 (candidate reduction)", f"KG {name} {et}", "reduction ratio", method)] = g.candidate_reduction_ratio.iloc[0]
    for (name, method), g in sbio.groupby(["dataset", "method"]):
        q[("Table 6 (candidate reduction)", f"BioML {name}", "reduction ratio", method)] = g.candidate_reduction_ratio.iloc[0]
    return q


def main() -> int:
    manifest = json.loads(Path(sys.argv[1]).read_text())
    art = {Path(k).name: REPO / k for k in manifest["artifacts_sha256"]}
    by_prefix = lambda prefix: next(p for n, p in art.items() if n.startswith(prefix) and not n.endswith(("_errors.csv",)) and "query" not in n)
    new_paths = {
        "dev": by_prefix("result_"),
        "kg": by_prefix("heldout_result_"),
        "bio": by_prefix("heldout_class_result_"),
        "selected": next(p for n, p in art.items() if n == "heldout_selected_settings.json"),
    }
    old_q, new_q = quantities(load(OLD)), quantities(load(new_paths))
    rows = []
    for key in sorted(set(old_q) | set(new_q), key=lambda k: (k[0], k[1], k[2], ORDER.index(k[3]) if k[3] in ORDER else 9)):
        old, new = old_q.get(key), new_q.get(key)
        rows.append({"location": key[0], "quantity": key[1], "metric": key[2],
                     "method": LABEL.get(key[3], key[3]), "old": old, "new": new,
                     "change_new_minus_old": None if old is None or new is None else new - old})
    table = pd.DataFrame(rows)

    # Statistics: submitted pseudo-replicated tests vs clustered inference.
    stats = []
    old_inf = pd.read_csv(OLD["kg_inference"])
    old_inf = old_inf[(old_inf.metric == "mrr") & (old_inf.aggregation_scope == "overall_dataset_type_rows")]
    new_inf = pd.read_csv(art["paired_inference_heldout.csv"])
    new_inf = new_inf[(new_inf.metric == "mrr") & (new_inf.scope == "knowledgegraph:all")]
    for row in old_inf.itertuples():
        match = new_inf[(new_inf.method_a == row.method_a) & (new_inf.method_b == row.method_b)]
        n = match.iloc[0] if len(match) else None
        stats.append({"location": "§6.1 text (significance)", "quantity": f"{LABEL[row.method_a]} - {LABEL[row.method_b]}, KG MRR",
                      "metric": "delta [95% CI], p", "method": "",
                      "old": f"{row.paired_delta:+.3f} [{row.ci_lower:+.3f}, {row.ci_upper:+.3f}], p={row.p_value:.4f} (15 rows as independent)",
                      "new": "" if n is None else
                      f"macro {n.delta_macro:+.3f} [{n.ci_low_macro:+.3f}, {n.ci_high_macro:+.3f}], Holm p={n.p_macro_holm:.3f} (floor {n.p_floor:.4f}); "
                      f"micro {n.delta_micro:+.3f} [{n.ci_low_micro:+.3f}, {n.ci_high_micro:+.3f}]",
                      "change_new_minus_old": None})
    regret = pd.read_csv(art["selection_regret.csv"])
    for row in regret[regret.rule == "lambda=0.5"].itertuples():
        stats.append({"location": "§7.1 text (transfer gap -> selection regret)", "quantity": f"{row.tier} held-out",
                      "metric": "regret, macro MRR", "method": LABEL[row.method],
                      "old": "transfer gap 0.011 (BM25) / 0.149 (TF-IDF), not comparable",
                      "new": f"{row.regret_macro:.4f} (rank {row.rank_macro} of {row.configurations})",
                      "change_new_minus_old": None})
    out = pd.concat([table, pd.DataFrame(stats)], ignore_index=True)
    out.to_csv(sys.argv[2], index=False)
    print(f"{len(out)} rows -> {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
