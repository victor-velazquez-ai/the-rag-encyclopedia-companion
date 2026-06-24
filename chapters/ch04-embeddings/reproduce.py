"""Chapter 4 experiment — self-host Qwen3 vs. managed; int8/binary + rescore vs. full float.

Reproduces the chapter's central claims on the golden set. First, embeds the corpus with the
self-host default (Qwen3-Embedding-8B, Apache-2.0) and a managed embedder, comparing retrieval
quality against the per-token / GPU cost. Second, sweeps the compression ladder — full float vs.
int8+rescore vs. binary+rescore, optionally stacked with MRL truncation — to reproduce the
"~128× smaller at near-float quality" result and expose the two silent traps (a query/document
prefix mismatch, and MRL truncation without re-normalization). Prints a quality delta (nDCG@10,
recall@k) and a cost delta ($/1M tokens or GPU-seconds, bytes/vector, search latency).

Phase 2 wires this to ragkit.ingestion.embedding and the shared eval harness.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
