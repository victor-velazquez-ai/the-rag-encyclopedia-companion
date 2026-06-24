"""ragkit.retrieval.routing — the cost optimizer: spend expensive paths only where they pay (Book Ch 6).

Not every query deserves the same machinery. A cheap classifier inspects the incoming query *before*
any expensive work and picks its path. Framed correctly, routing is a cost optimizer that frequently
improves quality as a side effect — by keeping heavy machinery off the easy queries it would only
overcomplicate.

    semantic.py     Semantic router — embed the query + k-NN against curated per-route utterances.
                    No LLM call in the decision: ~5000 ms (LLM router) -> ~100 ms, scales to
                    thousands of routes. As good as its curated utterances; re-seed as traffic drifts.
    complexity.py   Adaptive-RAG complexity classifier — no-retrieval / single-step / multi-step.
                    A simple query skips retrieval entirely; only true multi-hop pays for the Part III
                    loops. Bias toward retrieving when uncertain (silent failure is worse than waste).

Phase-1 scaffold: the surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class SemanticRouter:
#     """Embed + k-NN intent routing over curated utterances (no LLM call in the decision)."""
#     def route(self, query: str) -> str: ...           # -> route name (nearest cluster)
#
# class ComplexityClassifier:
#     """Adaptive-RAG: classify retrieval complexity. Bias toward retrieving when uncertain."""
#     def classify(self, query: str) -> str: ...        # -> "none" | "single" | "multi"

__all__ = ["SemanticRouter", "ComplexityClassifier"]  # populated in Phase 2
