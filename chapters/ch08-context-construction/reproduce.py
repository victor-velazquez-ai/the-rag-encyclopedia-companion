"""Reproduce the headline experiments of Book Chapter 8 — Context Construction & Assembly.

Two head-to-head comparisons on the version-controlled golden set, each printing a quality number
*and* a cost number (most of this chapter's wins require no model call at all):

  1. lost-in-the-middle ordering — hold retrieval and the passage *set* fixed and vary only the
     assembly order: rank-order vs. the zipper fold (rank 1 first, rank 2 last, zippering inward)
     vs. a random shuffle. The delta between rank-order and the fold is the lost-in-the-middle tax
     you reclaim for free — the cheapest point on the chapter's price-performance curve.
       metrics: exact-match / LLM-judge faithfulness (quality) · zero added latency (cost)

  2. compression vs. retrieve-less — does a LLMLingua compressor (LongLLMLingua / LLMLingua-2) beat
     the honest baseline of "just send fewer, better passages" once you charge the compressor for
     its *own* call? Measured at matched output-token budgets, compressor cost included — the
     accounting the vendor numbers omit.
       metrics: end-task accuracy (quality) · total latency + tokens, compressor included (cost)

Phase 2 wires these to ragkit.retrieval.context and ragkit.eval against data/golden/.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
