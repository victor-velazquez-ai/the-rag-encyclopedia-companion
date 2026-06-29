"""professional_rag_kit.retrieval.hybrid — BM25 + dense, fused with RRF (Book Ch 6).

The production default: run a sparse leg (BM25) and a dense leg (the Ch 4 embedder + Qdrant) and fuse
their *ranked lists* with Reciprocal Rank Fusion (k=60). The two retrievers fail on opposite inputs —
dense on exact identifiers and out-of-domain jargon, sparse on paraphrase — and ranks are scale-free,
so RRF dodges the BM25-unbounded-vs-cosine-bounded mismatch that makes linear score-blending fragile.

The pure pieces (``BM25``, ``rrf_fuse``) are dependency-free and unit-tested; the dense leg uses the
API-backed ``Embedder`` + ``VectorStore``, so a full hybrid search needs a key + a running Qdrant.
"""

from __future__ import annotations

from dataclasses import dataclass

from professional_rag_kit.retrieval.hybrid.bm25 import BM25, tokenize
from professional_rag_kit.retrieval.hybrid.fusion import rrf_fuse


@dataclass
class HybridRetriever:
    """BM25 + dense fused with RRF (k=60). Build with ``from_corpus`` (in-memory) for the walkthroughs.

    Pull a generous candidate depth from each leg before fusing — a doc ranked 80th by dense but
    2nd by BM25 must still enter fusion. RRF assumes both legs are competent; fix a weak leg upstream
    rather than expecting fusion to launder it.
    """

    bm25: BM25
    embedder: object | None = None  # professional_rag_kit.ingestion.embedding.Embedder
    store: object | None = None  # professional_rag_kit.ingestion.indexing.VectorStore
    k: int = 60

    @classmethod
    def from_corpus(cls, docs, *, embedder=None, store=None, k: int = 60) -> "HybridRetriever":
        """``docs`` is a sequence of (doc_id, text). The dense leg is optional (sparse-only if omitted)."""
        return cls(bm25=BM25(list(docs)), embedder=embedder, store=store, k=k)

    def search(self, query: str, top_k: int = 50, depth: int = 200) -> list[tuple[str, float]]:
        """Run sparse (+ dense if wired), RRF-fuse the ranked id lists, return the top_k."""
        rankings = [self.bm25.rank(query, depth)]
        if self.embedder is not None and self.store is not None:
            qvec = self.embedder.embed_query(query)
            dense_hits = self.store.search(qvec, top_k=depth)
            rankings.append([h.id for h in dense_hits])
        return rrf_fuse(rankings, k=self.k)[:top_k]


__all__ = ["HybridRetriever", "BM25", "rrf_fuse", "tokenize"]
