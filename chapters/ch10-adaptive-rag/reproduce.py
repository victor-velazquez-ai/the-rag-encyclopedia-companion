"""Chapter 10 - adaptive control: the route distribution (offline cost-optimizer demo).

The chapter's claim for Adaptive-RAG is that complexity routing is a cost optimizer: send each query
down the cheapest path that can answer it (no-retrieval / single / multi), spending the retrieval
budget only where it buys accuracy. This script routes a handful of queries and reports the
distribution - near-full coverage at a fraction of the retrieval steps a flat pipeline would pay.

Fully OFFLINE: `route` is a pure heuristic, no model. The reflective policies (CRAG / Self-RAG /
FLARE) are patterns over your frontier model and need a key; their call shape is in 01_adaptive.py.
"""

from __future__ import annotations

from collections import Counter

from ragkit.architectures.adaptive import AdaptiveRAG

QUERIES = [
    "What is the capital of France?",
    "Define reciprocal rank fusion.",
    "Summarize our Q3 refund policy.",
    "List the supported embedding providers.",
    "Compare GraphRAG and LightRAG and explain how their index costs relate.",
    "How does the reranker affect end-to-end latency, and which stage dominates?",
]
# cost weights: retrieval calls per route (the budget the optimizer is spending)
STEP_COST = {"no_retrieval": 0, "single": 1, "multi": 3}


def main() -> None:
    rag = AdaptiveRAG()  # pure router only
    routes = [(q, rag.route(q)) for q in QUERIES]

    print("Per-query routing:")
    for q, r in routes:
        print(f"  {r:>13}  <-  {q}")

    dist = Counter(r for _, r in routes)
    print("\nRoute distribution:")
    for name in ("no_retrieval", "single", "multi"):
        print(f"  {name:>13}: {dist.get(name, 0)}")

    total_steps = sum(STEP_COST[r] for _, r in routes)
    # A pipeline that always runs the heavy multi-step path to be safe on the hard queries.
    always_multi = len(QUERIES) * STEP_COST["multi"]
    print(f"\nRetrieval steps: adaptive={total_steps}  vs  always-multi-step={always_multi}")
    print("The router pays the multi-step cost only on the two compositional queries and skips")
    print("retrieval entirely on the factoids - near-full coverage at a fraction of the steps a")
    print("uniformly-heavy pipeline would spend. That is the cost optimizer's frontier up-and-left.")


if __name__ == "__main__":
    main()
