"""Chapter 9 experiment — does a graph earn its cost?

Reproduces the master-map comparison for this chapter: GraphRAG vs. LightRAG vs. plain vector RAG on
local-fact traffic, plus an answer-quality head-to-head on a set of genuinely global questions
(comprehensiveness / diversity win rates, never nDCG). For each system it reports the index-time
token bill and update cost alongside the quality number, and states which baseline each win rate is
measured against (GraphRAG's vs. vector RAG; LightRAG's and LazyGraphRAG's vs. GraphRAG itself) —
because in this chapter the baselines are not the same and must never be stacked on one axis.

Phase 1: this is a stub. The runnable experiment lands in Phase 2.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
