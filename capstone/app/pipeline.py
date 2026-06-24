"""RAGPipeline — the book's verdict stack, assembled (capstone).

    retrieve (hybrid BM25 + dense, RRF)  →  rerank  →  fold (lost-in-the-middle)  →  grounded generate

Every stage is the verdict-recommended component from its chapter, wired together. The dense leg
(Embedder + Qdrant) and an LLM reranker are optional — pass them in to enable them; omit them and the
pipeline runs sparse-only with no key, which is how the offline smoke test and tests exercise it.

The ``generator`` is any object with ``.generate(query, passages) -> GroundedAnswer``: the real
``GroundedGenerator`` (Claude/GPT) in production, or the offline ``ExtractiveAnswerer`` below when no
key is set — so the assembled pipeline always runs end-to-end.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragkit.production.generation import ABSTENTION, GroundedAnswer
from ragkit.retrieval.context import reorder_lost_in_middle
from ragkit.retrieval.hybrid import BM25, HybridRetriever


# A tiny stopword set so the offline answerer abstains on *content*, not on "the"/"of" overlap.
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "to", "in", "on", "at", "for",
    "and", "or", "with", "as", "by", "it", "its", "that", "this", "from", "what", "how", "do",
    "does", "within", "you", "your", "may", "can", "will", "i", "we", "my",
}


class ExtractiveAnswerer:
    """A no-API offline 'generator': returns the passage with the most query-term overlap, cited.

    Not a real generator — it extracts rather than synthesizes — but it gives the assembled pipeline
    a runnable, key-free default so you can exercise the whole flow before wiring in Claude/GPT. It
    abstains when no *content* term (stopwords removed) overlaps, mirroring the grounded prompt's rule.
    """

    def generate(self, query: str, passages: Sequence[str], **_) -> GroundedAnswer:
        from ragkit.retrieval.hybrid.bm25 import tokenize

        q = {t for t in tokenize(query) if t not in _STOP}
        scored = [(len(q & {t for t in tokenize(p) if t not in _STOP}), i, p) for i, p in enumerate(passages)]
        scored.sort(key=lambda t: (-t[0], t[1]))
        if not scored or scored[0][0] == 0:
            return GroundedAnswer(text=ABSTENTION, citations=[], abstained=True)
        overlap, idx, passage = scored[0]
        return GroundedAnswer(text=f"{passage} [{idx + 1}]", citations=[idx + 1], abstained=False)


@dataclass
class RAGPipeline:
    retriever: object  # has .search(query, top_k, depth) -> list[(doc_id, score)]
    doc_text: dict[str, str]
    generator: object  # has .generate(query, passages) -> GroundedAnswer
    reranker: object | None = None  # has .rerank(query, passages, top_k)
    top_k: int = 5
    depth: int = 20

    @classmethod
    def from_corpus(cls, docs: Sequence[tuple[str, str]], generator, *, embedder=None, store=None,
                    reranker=None, top_k: int = 5, depth: int = 20) -> "RAGPipeline":
        """``docs`` is (doc_id, text). Dense leg (embedder+store) and reranker are optional."""
        retriever = HybridRetriever.from_corpus(docs, embedder=embedder, store=store)
        return cls(retriever=retriever, doc_text=dict(docs), generator=generator,
                   reranker=reranker, top_k=top_k, depth=depth)

    def answer(self, query: str) -> GroundedAnswer:
        ranked = self.retriever.search(query, top_k=self.depth, depth=self.depth)
        passages = [self.doc_text[i] for i, _ in ranked if i in self.doc_text]
        if not passages:
            return GroundedAnswer(text=ABSTENTION, citations=[], abstained=True)
        if self.reranker is not None:
            passages = self.reranker.rerank(query, passages, top_k=self.top_k)
        else:
            passages = passages[: self.top_k]
        folded = reorder_lost_in_middle(passages)  # strongest evidence on the attention peaks
        return self.generator.generate(query, folded)


__all__ = ["RAGPipeline", "ExtractiveAnswerer", "BM25"]
