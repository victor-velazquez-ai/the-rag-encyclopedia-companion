"""Chapter 10 experiment — hierarchy and adaptive control, each forced to prove its delta.

Reproduces the master-map comparisons for this chapter on the golden set:

  * RAPTOR collapsed-tree — the synthesis-question lift over a flat baseline, and collapsed-tree
    vs. tree-traversal (the paper's mode wins), measured on questions that require integration, not
    lookup (where alone the gain appears).
  * Adaptive-RAG — accuracy plotted against average retrieval steps per query (near-full accuracy at
    ~half the steps); the cost optimizer's frontier moving up-and-left.
  * CRAG — a lift on top of a base pipeline (the Self-CRAG deltas), read as the additive grade-and-
    correct improvement on the queries where retrieval is shaky.
  * Self-RAG — a grounding delta (claims that trace to evidence) with vs. without the support loop.
  * FLARE — a theta sweep of answer quality against retrievals-per-answer.

Every comparison prints quality AND cost; each policy is judged on the delta it produces, not on
aggregate accuracy.

Phase 1: this is a stub. The runnable experiment lands in Phase 2.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
