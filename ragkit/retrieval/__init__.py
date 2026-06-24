"""ragkit.retrieval — Part II: finding the right evidence and ordering it (Book Ch 6–8).

    hybrid/   BM25 + dense with RRF (k=60) fusion; optional SPLADE-v3 sparse leg   (Ch 6)
    query/    HyDE · step-back · multi-query · decomposition — router-gated         (Ch 6)
    routing/  semantic router + complexity classifier (the cost optimizer)          (Ch 6)
    rerank/   cross-encoder · listwise-LLM · ColBERT tier · cascade                  (Ch 7)
    context/  lost-in-the-middle reorder · rerank-then-truncate · MMR · compression (Ch 8)

Phase-1 scaffold. Phase 2 exports the top-level conveniences: HybridRetriever, Reranker.
"""

# Phase 2 will export: HybridRetriever (Ch 6), Reranker (Ch 7), ContextBuilder (Ch 8)
__all__ = ["HybridRetriever", "Reranker", "ContextBuilder"]
