"""ragkit.retrieval.rerank — reranking tiers (Book Ch 7).

The cheap-recall → precise-rerank cascade. One ``Reranker`` facade over four tiers, so you can
start with the cross-encoder default and escalate only when measurement says the ceiling binds.

    cross_encoder.py   jina-reranker-v3 default; Cohere Rerank 4 as a managed swap
    listwise.py        RankZephyr-style listwise LLM reranking (largest lift, highest latency)
    colbert.py         ColBERT late-interaction tier (near-cross-encoder quality, cheaper)
    cascade.py         multi-stage funnel with per-stage candidate caps + latency budget

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class Reranker:
#     """Facade over the reranking tiers. `Reranker.default()` → jina-reranker-v3 (open)."""
#     @classmethod
#     def default(cls) -> "Reranker": ...
#     @classmethod
#     def from_provider(cls, provider: str) -> "Reranker":   # "jina" | "cohere"
#         ...
#     def rerank(self, query: str, candidates: list, top_k: int | None = None) -> list:
#         """Re-score candidates against the query; return them sorted best-first."""
#         ...

__all__ = ["Reranker"]  # populated in Phase 2
