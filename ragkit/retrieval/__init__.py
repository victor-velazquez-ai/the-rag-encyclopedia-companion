"""ragkit.retrieval — Part II: finding the right evidence and ordering it (Book Ch 6–8).

    hybrid/   BM25 + dense with RRF (k=60) fusion; optional SPLADE-v3 sparse leg   (Ch 6)
    query/    HyDE · step-back · multi-query · decomposition — router-gated         (Ch 6)
    routing/  semantic router + complexity classifier (the cost optimizer)          (Ch 6)
    rerank/   cross-encoder · listwise-LLM · ColBERT tier · cascade                  (Ch 7)
    context/  lost-in-the-middle reorder · rerank-then-truncate · MMR · compression (Ch 8)

Phase-1 scaffold. Phase 2 exports the top-level conveniences: HybridRetriever, Reranker.
"""

from ragkit.retrieval.context import ContextBuilder
from ragkit.retrieval.hybrid import HybridRetriever
from ragkit.retrieval.rerank import Reranker

__all__ = ["HybridRetriever", "Reranker", "ContextBuilder"]
