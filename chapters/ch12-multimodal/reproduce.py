"""Chapter 12 experiment — retrieve over pixels vs. a lossy text projection.

Reproduces the master-map comparisons for this chapter on the golden set of visually-rich pages:

  * ColPali / ColQwen visual document retrieval vs. the OCR-then-embed text pipeline — page-level
    nDCG@5 on the same query set (the delta is the entire business case), alongside indexing speed
    (seconds/page).
  * Native multimodal object embedding vs. image-to-text summaries on number-dense charts and tables
    (mAP@5 / nDCG@5), the regime where the text-conversion step most often fabricates or flattens.

Each quality number is paired with the storage estimate — pages x ~257.5 KB (or /3 with token
pooling) vs. pages x ~8.6 KB for a text chunk vector — because the rule is: ship visual retrieval
only when the quality delta is real AND the storage line is affordable. Any public-benchmark sanity
check runs on ViDoRe V2, never the near-saturated V1.

Phase 1: this is a stub. The runnable experiment lands in Phase 2.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
