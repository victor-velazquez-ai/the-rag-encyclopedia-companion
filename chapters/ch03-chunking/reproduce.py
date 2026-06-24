"""Chapter 3 experiment — semantic vs. tuned ~200-token recursive; Contextual Retrieval on/off.

Reproduces the chapter's central claim on the golden set. First, runs embedding-breakpoint
semantic chunking head-to-head against a tuned ~200-token structure-aware recursive splitter,
to show the cheap baseline holds (or wins) on structured documents. Second, ablates Anthropic
Contextual Retrieval — baseline → +contextual embeddings → +contextual BM25 → +rerank — to
reproduce the halved top-20 retrieval-failure rate (5.7% → 2.9% → 1.9%). Prints a quality delta
(recall, nDCG@10, top-20 failure rate) and a cost delta (index-time LLM $/M tokens, extra
embedding passes), so the spend decision is made with both numbers in front of you.

Phase 2 wires this to ragkit.ingestion.chunking and the shared eval harness.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
