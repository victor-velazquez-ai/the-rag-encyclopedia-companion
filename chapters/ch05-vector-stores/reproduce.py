"""Chapter 5 experiment — HNSW vs. quantized recall; the filtered-ANN recall collapse.

Reproduces the chapter's central claims on the golden set, all against Qdrant. First, sweeps the
in-index quantization ladder — full HNSW vs. int8 vs. binary vs. RaBitQ, each with a full-precision
rescore over the top candidates — to show how much recall the rescore recovers and what each tier
costs in RAM and QPS. Second, runs the filtered-ANN failure demo: naive post-filtering vs. Qdrant's
filterable HNSW under a deliberately adversarial, negatively-correlated filter, measuring recall
against an *exhaustive filtered* ground truth so the silent collapse becomes visible. Prints a
quality delta (recall@k vs. exhaustive baseline) and a cost delta (QPS, p99 latency, bytes/vector).

Phase 2 wires this to ragkit.ingestion.indexing and the shared eval harness.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
