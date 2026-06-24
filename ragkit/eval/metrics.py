"""Retrieval metrics (Book Ch 14) — pure, dependency-free.

The three the chapter commits to, in increasing richness:

- ``recall_at_k``  — fraction of all relevant docs found in the top k. Order-agnostic; the floor.
- ``mrr``          — 1 / rank of the *first* relevant doc. Rank-aware; right when one good hit suffices.
- ``ndcg_at_k``    — DCG / IDCG. The only one that handles *graded* relevance and full-ordering quality.
                      DCG@k = sum_i  rel_i / log2(i + 1)   (i is 1-based rank).

Each takes a ranked list of doc ids and a relevance signal, so they compose with any retriever.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of ``relevant`` docs appearing in the top ``k`` of ``ranked``."""
    if not relevant:
        return 0.0
    topk = set(ranked[:k])
    return len(topk & relevant) / len(relevant)


def mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant doc (0.0 if none retrieved)."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Discounted cumulative gain over the top ``k`` (graded relevance via ``gains``)."""
    total = 0.0
    for i, doc_id in enumerate(ranked[:k], start=1):
        rel = gains.get(doc_id, 0.0)
        if rel:
            total += rel / math.log2(i + 1)
    return total


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """nDCG@k = DCG@k / ideal-DCG@k. Returns 0.0 when there is no relevant gain."""
    ideal_order = sorted(gains, key=lambda d: gains[d], reverse=True)
    idcg = dcg_at_k(ideal_order, gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(ranked, gains, k) / idcg


__all__ = ["recall_at_k", "mrr", "dcg_at_k", "ndcg_at_k"]
