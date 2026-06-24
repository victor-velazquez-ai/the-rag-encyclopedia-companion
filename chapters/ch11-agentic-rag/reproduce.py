"""Chapter 11 experiment — single-shot vs. agentic multi-hop, and the entity-resolution lift.

Reproduces the master-map comparisons for this chapter on the golden set:

  * Single-shot vs. agentic multi-hop on a genuinely compositional question set (MultiHop-RAG /
    FRAMES-style). Beyond final-answer EM/F1, reports per-hop intermediate accuracy (the chain's
    ceiling is the product of its hops), hop-count distribution, and non-termination rate — and
    licenses the loop on correctness-lift-per-dollar over single-shot on the routed subset, not on
    aggregate accuracy (which the complexity gate's routing confounds).
  * Entity resolution — cross-source retrieval completeness for entity-centric queries with the
    canonical-ID layer on vs. off (the lift that pays for the pipeline), plus matcher pairwise
    precision/recall and the false-merge rate, since over-merging's cost is asymmetric.

Every comparison prints quality AND cost.

Phase 1: this is a stub. The runnable experiment lands in Phase 2.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
