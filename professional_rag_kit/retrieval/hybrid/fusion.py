"""Reciprocal Rank Fusion (Book Ch 6) — pure, no dependencies.

RRF (Cormack, Clarke & Buettcher, SIGIR 2009) fuses several ranked lists by summing reciprocal
ranks. It uses *rank only*, never the raw scores — which is exactly why it beats a linear score
blend: BM25 scores are unbounded and corpus-dependent while cosine sits in ~[-1, 1], so a weighted
sum lets one scale swamp the other. ``k`` damps the influence of any single list's top hit; 60 is
the paper's empirical best-average on TREC, not a theoretical optimum — re-tune per corpus.

    RRF_score(d) = sum over lists of  1 / (k + rank_in_list(d))     # rank is 1-based
"""

from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of document ids into one list, best-first.

    Args:
        rankings: each inner sequence is one retriever's doc ids, ordered best-first.
        k: the RRF constant (default 60).

    Returns:
        ``[(doc_id, score), ...]`` sorted by descending fused score. Ties broken by doc id for
        determinism.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


__all__ = ["rrf_fuse"]
