"""Chapter 14 reproduction — the orthogonality demonstration and LLM-judge calibration.

Runs the chapter's spine claims on the golden set:

  * the orthogonality demonstration — score the same answers end-to-end vs. with separate
    retrieval and generation numbers, showing how a single accuracy figure hides which half is
    broken (the FRAMES 0.40 -> 0.66 no-retrieval-to-retrieval delta in miniature, isolating how
    much "correctness" was parametric), and
  * an LLM-judge calibration pass — position-swap consistency gating, printing how often raw
    verdicts flip when nothing changes but the order of the two candidates.

Prints a quality number (nDCG@k, faithfulness) AND a cost number (judge calls per metric per
query), because even the harness must price itself. This is the experiment that makes every other
verdict in the book falsifiable on your own corpus.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
