"""ragkit.retrieval.hybrid — BM25 + dense, fused with RRF (Book Ch 6).

The production default: run a sparse leg and a dense leg and fuse their *ranked lists* with
Reciprocal Rank Fusion (k=60), because the two retrievers fail on opposite inputs — dense on exact
identifiers and out-of-domain jargon, sparse on paraphrase — and ranks are scale-free, so RRF dodges
the BM25-unbounded-vs-cosine-bounded mismatch that makes linear blending fragile.

    bm25.py     sparse leg — exact-term overlap, no training, no GPU, robust out of domain
    dense.py    dense leg — single learned vector per query/doc (the Ch 4 embedder), cosine score
    splade.py   optional SPLADE-v3 learned-sparse leg — term expansion on one inverted index
    rrf.py      Reciprocal Rank Fusion (Cormack 2009); k=60 default, swept on your held-out set

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class HybridRetriever:
#     """BM25 + dense fused with RRF (k=60). Optional SPLADE-v3 sparse leg.
#
#     Pull a generous candidate depth from each leg before fusing — a doc ranked 80th by
#     dense but 2nd by BM25 must still enter fusion. Fusion assumes both legs are competent;
#     fix or drop a weak leg upstream rather than expecting RRF to launder it.
#     """
#     @classmethod
#     def from_config(cls, path: str, query_transform: str | None = None) -> "HybridRetriever":
#         ...
#     def search(self, query: str, top_k: int = 50, depth: int = 200) -> list:
#         """Run sparse + dense to `depth`, RRF-fuse, return the top_k fused candidates."""
#         ...

__all__ = ["HybridRetriever"]  # populated in Phase 2
