"""Build the retriever for a fixed (pinned, never tuned) dense or hybrid baseline.

Imports are lazy so lexical-only runs never load torch.
"""

from __future__ import annotations

from src.config_loader import FixedBaselineEntry, HybridModelEntry


def is_fixed_baseline(method_name: str) -> bool:
    return method_name.startswith(("dense_", "hybrid_"))


def build_fixed_baseline(entry: FixedBaselineEntry, candidate_policy: str = "nonzero"):
    """Return an unfitted retriever exposing ``fit``, ``retrieve`` and ``retrieve_many``."""
    if isinstance(entry, HybridModelEntry):
        from src.retrieval.hybrid_retriever import HybridRetriever

        return HybridRetriever(
            model_name=entry.dense.model,
            revision=entry.dense.revision,
            device=entry.device,
            batch_size=entry.batch_size,
            rrf_k=entry.rrf_k,
            depth=entry.depth,
            candidate_policy=candidate_policy,  # type: ignore[arg-type]
        )
    from src.retrieval.dense_retriever import DenseRetriever

    return DenseRetriever(
        model_name=entry.model,
        revision=entry.revision,
        device=entry.device,
        batch_size=entry.batch_size,
        candidate_policy=candidate_policy,  # type: ignore[arg-type]
    )
