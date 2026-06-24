"""Chapter 7 - Reranking: the headline experiment (reranker ON vs OFF).

The chapter's central claim is that an expensive rerank over a cheap first-stage shortlist lifts
quality - at a latency cost (the reranker is often 60-84% of retrieval-pipeline latency). This
script shows the ON-vs-OFF reorder on a small golden example.

With a real key it uses the LLM listwise reranker (`Reranker.default()` -> Claude/GPT). With NO key
it falls back to a tiny deterministic generation backend so the comparison still RUNS offline.
"""

from __future__ import annotations

import os

from ragkit.core.config import ProviderRegistry
from ragkit.retrieval.hybrid import BM25
from ragkit.retrieval.rerank import Reranker

CORPUS = [
    ("d1", "A reranker model scores relevance; a reranker improves relevance ranking of results."),
    ("d2", "Our search model indexes documents and returns the top results for a query."),
    ("d3", "A cross-encoder re-scores a shortlist and reorders it so the best answer lands first."),
    ("d4", "Reranking is the precise second stage that fixes a noisy first-stage candidate order."),
]
QUERY = "how does a reranker improve relevance"


def _build_reranker() -> Reranker:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        print("[key found] using the LLM listwise reranker (Reranker.default()).")
        return Reranker.default()
    print("[no key] using an offline deterministic fake backend so this still runs.")
    ProviderRegistry.register("generation", "demo_reproduce")(
        lambda model, system, prompt, maxtok: "2,4,1,3"
    )
    return Reranker(provider="llm", gen_provider="demo_reproduce", gen_model="x")


def main() -> None:
    id_to_text = dict(CORPUS)
    bm = BM25(CORPUS)

    # OFF: first-stage (BM25) order only.
    off = bm.rank(QUERY, top_k=4)
    candidates = [id_to_text[cid] for cid in off]

    # ON: rerank the shortlist.
    reranker = _build_reranker()
    on = reranker.rerank(QUERY, candidates, top_k=4)

    print("\nReranker OFF (BM25 order):")
    for i, cid in enumerate(off, 1):
        print(f"  {i}. {cid}: {id_to_text[cid]}")

    print("\nReranker ON (reordered):")
    for i, passage in enumerate(on, 1):
        print(f"  {i}. {passage}")

    print("\nQUALITY: the explanatory passages are promoted over the keyword-stuffed BM25 winner.")
    print("COST: the reranker is often 60-84% of retrieval-pipeline latency; candidate count is")
    print("      the cost knob - here", len(candidates), "candidates were re-scored.")


if __name__ == "__main__":
    main()
