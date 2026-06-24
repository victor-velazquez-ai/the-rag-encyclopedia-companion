"""ragkit.retrieval.context — turn a ranked shortlist into the tokens the model reads (Book Ch 8).

An ordered pipeline of cheap decisions — order, budget, compress, diversify, arbitrate, route — with
the best impact-to-cost ratio in the book, because most of its wins require no model call. The free
ones come first and clear most of the bar; the rest are conditional additions you justify on your own
data. One rule above all: a long context window does not replace careful assembly — it re-exposes
lost-in-the-middle at full scale and costs more.

    reorder.py      Lost-in-the-middle zipper fold — rank 1 first, rank 2 last, zippering inward, so
                    the strongest evidence sits on the two attention peaks. Free; do it first.
    budget.py       Rerank-then-truncate in *tokens* (not passages); reserve headroom for the system
                    prompt, the query, and the answer. Ship just above the accuracy knee.
    compress.py     LongLLMLingua (question-aware, quality) / LLMLingua-2 (distilled, fast) — only
                    when a passage must stay but can't be afforded whole, and only after "just
                    retrieve less" has failed. Never compress code/numbers/entities/clauses.
    mmr.py          Maximal Marginal Relevance on the reranked shortlist (O(k²), keep k small),
                    λ≈0.6–0.7. Kills redundancy; blind to contradiction by construction.
    arbitrate.py    Dedup freely; resolve inter-context contradiction as a *separate* stage —
                    claim decomposition + source-reliability + recency/authority precedence, powered
                    by metadata ingestion must have carried forward.
    self_route.py   Self-Route — answer from retrieved context and self-assess; escalate to a full
                    long-context read only when the model declares the context insufficient.

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class ContextBuilder:
#     """Assemble the final prompt context from a reranked shortlist.
#
#     order -> budget -> (compress) -> (diversify) -> (arbitrate) -> (route). The free stages
#     (fold, rerank-then-truncate, dedup) are always on; compress/mmr/self_route are opt-in.
#     """
#     def __init__(self, token_budget: int, mmr: float | None = None,
#                  compress: str | None = None, self_route: bool = False) -> None: ...
#     def build(self, query: str, passages: list) -> str:
#         """Fold (lost-in-the-middle), truncate to the token budget, then any opted-in stages."""
#         ...

__all__ = ["ContextBuilder"]  # populated in Phase 2
