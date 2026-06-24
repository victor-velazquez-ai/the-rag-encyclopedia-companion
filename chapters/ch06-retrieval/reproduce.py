"""Reproduce the headline experiments of Book Chapter 6 — Retrieval & Query Understanding.

Two head-to-head comparisons on the version-controlled golden set, each printing a quality number
*and* a cost number (the chapter's argument is never quality alone):

  1. hybrid+RRF vs. dense-only — does fusing BM25 + dense with Reciprocal Rank Fusion (k=60) beat
     either leg alone? Watch the identifier-bearing queries (order numbers, error strings, SKUs)
     where the single pooled dense vector blurs the join key and BM25 carries the recall.
       metrics: nDCG@10, Recall@k (quality) · added latency, $/1k queries (cost)

  2. HyDE on a strong vs. a weak embedder — the +16 to +32 nDCG@10 lift HyDE shows against an
     unsupervised embedder, and how it shrinks toward zero (and can reverse) against a modern
     fine-tuned one — while still paying a full pre-retrieval LLM call in latency.
       metrics: nDCG@10 delta (quality) · pre-retrieval LLM-call latency (cost)

Phase 2 wires these to ragkit.retrieval.{hybrid,query,routing} and ragkit.eval against data/golden/.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
