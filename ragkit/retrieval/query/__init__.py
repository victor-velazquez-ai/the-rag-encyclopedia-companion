"""ragkit.retrieval.query — query transformation, router-gated (Book Ch 6).

Improve the *query itself* before it is matched. Every transform here puts one or more LLM calls on
the retrieval critical path, so each must earn that latency — none is turned on globally; they fire
only on the slice the routing layer sends them, and only after measurement says they still beat your
embedder.

    hyde.py          Hypothetical Document Embeddings — embed a fabricated answer, not the query.
                     Big lift (+16 to +32 nDCG@10) vs a weak/zero-shot embedder; shrinks and can
                     reverse against a modern fine-tuned one. A/B before shipping.
    step_back.py     Step-back prompting — abstract the question to its governing principle, then
                     retrieve. For reasoning-/principle-seeking corpora; skip on factoid lookups.
    multi_query.py   Generate paraphrases, retrieve each, fuse with RRF — buys recall when phrasing
                     variance is the measured recall problem.
    decompose.py     Split multi-hop / multi-part questions into atomic sub-queries — near-mandatory
                     when the answer is split across documents no single query retrieves.

Phase-1 scaffold: the surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class QueryTransform:
#     """Router-gated query transforms. Each adds an LLM call; route them on, don't globally enable."""
#     def hyde(self, query: str) -> str: ...           # -> hypothetical doc to embed
#     def step_back(self, query: str) -> str: ...       # -> abstracted question
#     def multi_query(self, query: str, n: int = 3) -> list[str]: ...   # -> paraphrases (RRF-fused)
#     def decompose(self, query: str) -> list[str]: ...                 # -> atomic sub-queries

__all__ = ["QueryTransform"]  # populated in Phase 2
