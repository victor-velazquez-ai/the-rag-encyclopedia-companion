"""ragkit.eval.suite — the reproduction runner behind `make reproduce` (Book Ch 14).

`python -m ragkit.eval.suite` re-runs the book's head-to-head comparisons on the version-controlled
golden set, printing a *quality* number and a *cost* number for each, because the book's argument is
never quality alone. This makes every verdict falsifiable on your own corpus.

This offline reproduction needs no API key: it compares a body-only BM25 retriever against a BM25
body+title fusion (RRF) and scores both with the harness. The point is the *method* — name two
configs, score them on the same golden set, read the delta — which extends directly to the keyed
comparisons (semantic-vs-recursive chunking, reranker on/off, GraphRAG-vs-vector) that live in the
chapter `reproduce.py` files and use real embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ragkit.eval import GoldenSet, Harness
from ragkit.retrieval.hybrid import BM25, rrf_fuse

_DATA = Path(__file__).resolve().parents[2] / "data"


def _load_corpus() -> list[dict]:
    path = _DATA / "corpus_small" / "docs.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_retrievers(corpus: list[dict], depth: int = 10):
    body = BM25([(d["id"], d["text"]) for d in corpus])
    title = BM25([(d["id"], d["title"]) for d in corpus])

    def bm25_body(query: str) -> list[str]:
        return body.rank(query, depth)

    def bm25_fusion(query: str) -> list[str]:
        # fuse body + title rankings — recovers docs whose query vocabulary lives in the title
        fused = rrf_fuse([body.rank(query, depth), title.rank(query, depth)], k=60)
        return [doc_id for doc_id, _ in fused]

    return {"BM25 (body only)": bm25_body, "BM25 body + title (RRF)": bm25_fusion}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ragkit reproduction suite")
    ap.add_argument("--all", action="store_true", help="run all offline reproductions")
    ap.add_argument("--k", type=int, default=5, help="cutoff k for the metrics")
    args = ap.parse_args(argv)

    corpus = _load_corpus()
    golden = GoldenSet.from_jsonl(_DATA / "golden" / "qa.jsonl")
    cards = Harness.default().compare(_build_retrievers(corpus), golden, k=args.k)

    print(f"\nRetrieval reproduction on the golden set (n={len(golden)}, k={args.k}):\n")
    for c in cards:
        print("  " + c.row())

    best = cards[0]
    base = next(c for c in cards if c.name == "BM25 (body only)")
    delta = best.ndcg - base.ndcg
    print(
        f"\n  delta nDCG@{args.k} (best - body-only): {delta:+.3f} - the title-fusion leg "
        f"{'helps' if delta > 0 else 'does not help'} on this corpus.\n"
        "  The lesson is the method, not the number: measure the delta on YOUR golden set before\n"
        "  shipping the extra leg. Keyed reproductions (chunking, rerankers, graph) need an API key\n"
        "  -- see each chapter's reproduce.py.\n"
    )


if __name__ == "__main__":
    main()
