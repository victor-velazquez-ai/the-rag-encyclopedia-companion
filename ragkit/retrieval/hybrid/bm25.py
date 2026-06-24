"""Okapi BM25 — the sparse leg of hybrid retrieval (Book Ch 6), pure Python.

BM25 scores a query against a document by exact term overlap, weighted by term frequency (with
saturation) and inverse document frequency, normalized by document length. It is the brutally strong
lexical baseline that catches exact identifiers, codes, and rare entities a dense vector blurs away —
which is why hybrid (BM25 + dense, fused with RRF) is the production default.

    score(q, d) = sum_t  idf(t) · ( f(t,d)·(k1+1) ) / ( f(t,d) + k1·(1 − b + b·|d|/avgdl) )
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25:
    """A tiny in-memory BM25 index over (doc_id, text) pairs."""

    def __init__(self, docs: Sequence[tuple[str, str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self.tokens: list[list[str]] = []
        self.tf: list[Counter] = []
        df: Counter = Counter()
        for doc_id, text in docs:
            toks = tokenize(text)
            self.doc_ids.append(doc_id)
            self.tokens.append(toks)
            tf = Counter(toks)
            self.tf.append(tf)
            df.update(tf.keys())
        self.n = len(self.doc_ids)
        self.avgdl = (sum(len(t) for t in self.tokens) / self.n) if self.n else 0.0
        # BM25 idf with the standard +0.5 smoothing (clamped at 0 to avoid negatives on common terms)
        self.idf = {
            term: max(0.0, math.log((self.n - n_t + 0.5) / (n_t + 0.5) + 1.0))
            for term, n_t in df.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return ``[(doc_id, score), ...]`` for the top_k docs, best-first."""
        q_terms = tokenize(query)
        scores: list[tuple[str, float]] = []
        for i, doc_id in enumerate(self.doc_ids):
            tf, dl = self.tf[i], len(self.tokens[i])
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / denom
            if s > 0:
                scores.append((doc_id, s))
        scores.sort(key=lambda kv: (-kv[1], kv[0]))
        return scores[:top_k]

    def rank(self, query: str, top_k: int = 10) -> list[str]:
        """Just the ranked doc ids (convenient for RRF fusion)."""
        return [doc_id for doc_id, _ in self.search(query, top_k)]


__all__ = ["BM25", "tokenize"]
