"""Chapter 13 reproduction — grounding on vs. off, and the counterfactual context probe.

Runs the chapter's central comparison on the golden set:

  * grounding ON (context-faithful prompting + Chain-of-Note + CAD) vs. a naive
    "answer using the context" prompt, and
  * the counterfactual context test — feed deliberately wrong context and confirm CAD now
    follows it, quantifying the retrieval-quality dependency this chapter keeps returning to.

Prints a quality delta (faithfulness, answer correctness, ALCE-style citation recall/precision)
AND a cost delta (added tokens/latency for the note step and CAD's second forward pass), because
faithfulness is not correctness: every technique that makes the model trust the context more makes
it trust bad context more too.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
