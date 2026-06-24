"""Chapter 7 — Reranking: the headline experiment.

Reproduces the chapter's central claim on the golden set:
  1. Reranker ON vs. OFF  — does a cross-encoder over the top-50 candidates lift nDCG@10 and
     answer correctness, and by how much p95 latency?
  2. Listwise LLM vs. distilled cross-encoder — the largest lift vs. the cheapest acceptable one.

Prints a QUALITY delta and a COST delta (added p95 latency, $/1k queries) for each — because the
chapter's point is that the reranker is the highest-leverage quality lever *and* often 60-84% of
retrieval-pipeline latency. You decide with both numbers in front of you.

Phase-1 scaffold: implemented in Phase 2 against ragkit.eval + ragkit.retrieval.rerank.
"""


def main() -> None:
    print("Phase 2 — see README.md")


if __name__ == "__main__":
    main()
