"""professional_rag_kit.retrieval.context — turn a ranked shortlist into the tokens the model reads (Book Ch 8).

An ordered pipeline of cheap decisions with the best impact-to-cost ratio in the book — most of its
wins require no model call. The free stages are always on (lost-in-the-middle fold, rerank-then-
truncate, MMR dedup); compression and self-route are opt-in. One rule above all: a long context
window does not replace careful assembly — it re-exposes lost-in-the-middle at full scale.

Implemented here (pure): the zipper fold and MMR. Compression/arbitration/self-route are Phase 2+.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def reorder_lost_in_middle(passages: Sequence[str]) -> list[str]:
    """Zipper fold (Ch 8): strongest passages land on the two attention peaks (start and end).

    Input is best-first (rank 1 strongest). Output places rank 1 first, rank 2 last, rank 3 second,
    rank 4 second-to-last, ... so the weakest evidence sits in the low-attention middle.
    """
    out: list[str | None] = [None] * len(passages)
    lo, hi = 0, len(passages) - 1
    for i, p in enumerate(passages):
        if i % 2 == 0:
            out[lo] = p
            lo += 1
        else:
            out[hi] = p
            hi -= 1
    return [p for p in out if p is not None]


def mmr(
    query_vec: Sequence[float],
    doc_vecs: Sequence[Sequence[float]],
    k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance (Carbonell & Goldstein 1998). Returns selected doc *indices*.

    Greedy, O(k·n): each step picks the doc maximizing
        λ·sim(doc, query) − (1−λ)·max sim(doc, already-selected).
    λ≈0.7 keeps relevance dominant while breaking up near-duplicates (Ch 8). Cosine similarity;
    run it on a small reranked shortlist, not the whole index.
    """
    import numpy as np

    q = np.asarray(query_vec, dtype=float)
    d = np.asarray(doc_vecs, dtype=float)
    if d.ndim != 2 or len(d) == 0:
        return []

    def _norm(m):
        n = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(n == 0, 1, n)

    qn = _norm(q.reshape(1, -1))[0]
    dn = _norm(d)
    rel = dn @ qn  # cosine to query
    sim = dn @ dn.T  # pairwise cosine

    selected: list[int] = []
    candidates = set(range(len(d)))
    k = min(k, len(d))
    while len(selected) < k:
        best_i, best_score = None, -np.inf
        for i in candidates:
            redundancy = max((sim[i][j] for j in selected), default=0.0)
            score = lambda_ * rel[i] - (1 - lambda_) * redundancy
            if score > best_score:
                best_i, best_score = i, score
        selected.append(best_i)
        candidates.remove(best_i)
    return selected


@dataclass
class ContextBuilder:
    """Assemble the final context: fold (lost-in-the-middle) → truncate to a token budget.

    MMR diversification is opt-in (pass ``mmr`` as λ and provide vectors to ``build``). Token counts
    use a word-count proxy; swap a real tokenizer for production budgeting.
    """

    token_budget: int = 4000
    mmr: float | None = None

    def build(
        self,
        query: str,
        passages: Sequence[str],
        *,
        query_vec: Sequence[float] | None = None,
        passage_vecs: Sequence[Sequence[float]] | None = None,
    ) -> str:
        items = list(passages)

        # opt-in MMR dedup on the shortlist (needs vectors)
        if self.mmr is not None and query_vec is not None and passage_vecs is not None:
            order = mmr(query_vec, passage_vecs, k=len(items), lambda_=self.mmr)
            items = [items[i] for i in order]

        # rerank-then-truncate in tokens, reserving nothing fancy (proxy budget)
        kept, used = [], 0
        for p in items:
            t = len(p.split())
            if used + t > self.token_budget and kept:
                break
            kept.append(p)
            used += t

        # lost-in-the-middle fold last, so the strongest kept passages sit on the peaks
        folded = reorder_lost_in_middle(kept)
        return "\n\n".join(folded)


__all__ = ["ContextBuilder", "reorder_lost_in_middle", "mmr"]
