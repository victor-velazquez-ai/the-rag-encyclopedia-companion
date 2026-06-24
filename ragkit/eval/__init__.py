"""ragkit.eval — the evaluation harness: the book's spine, made runnable (Book Ch 14).

The thesis is operational here: *measurement licenses complexity.* No technique earns a place in
the pipeline because it is sophisticated — only because a number on a held-out set moved, and stayed
moved, when you turned it on. This module is the instrument that produces that number, and every
chapter's `reproduce.py` calls into it.

The one non-negotiable conceptual move: *retrieval failure and generation failure are orthogonal,*
so the harness scores them SEPARATELY against their own ground truth and never debugs from
end-to-end accuracy alone (a high retrieval / suspiciously-high generation pair means the model
answered from parametric memory, not your evidence — the dangerous case).

    golden_set.py   the version-controlled query set with known-good outputs (relevant docs and/or
                    reference answers) — the unit-test fixture of RAG. Representative of real
                    traffic, stable across weeks, grown from production failures. Synthetic
                    generation (RAGAS knowledge-graph evolutions: single-hop / multi-context /
                    reasoning; ARES with PPI) solves cold start — always spot-checked against humans.
    metrics.py      retrieval metrics — nDCG@k (the RAG default; graded relevance + full ordering,
                    k set to the REAL context depth), Recall@k (the floor alarm), MRR (single-hit
                    tasks) — and generation metrics kept separate: faithfulness, answer relevance,
                    answer correctness. Faithful is not correct: track them apart.
    ragas.py        RAGAS-style reference-free scoring (Faithfulness, Answer Relevance, Context
                    Precision@K, Context Recall; Noise Sensitivity / Answer Correctness where
                    references exist). The production default — good enough to ship, cheap enough to
                    run every CI build — treated as the LLM-judge scores they are.
    judge.py        LLM-as-judge calibration — ~85% human agreement licenses automation, but never
                    raw: POSITION-SWAP consistency gating first (judge twice, swap order, count only
                    agreements), a CROSS-FAMILY judge against self-enhancement (+10% to +25%), and
                    PPI over a few hundred human anchor labels for defensible confidence intervals.
    suite.py        the `python -m ragkit.eval.suite --all` runner the Makefile's `make reproduce`
                    target invokes — re-runs every chapter's head-to-head on the golden set.

`Harness` runs a config against a `GoldenSet` and returns the stage-wise scorecard plus a cost
number; the book's argument is never quality alone.

Phase-1 scaffold: the surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class GoldenSet:
#     """A version-controlled set of queries with relevant-doc and/or reference-answer ground truth."""
#     @classmethod
#     def load(cls, name: str = "default") -> "GoldenSet": ...
#     def __iter__(self): ...    # yields (query, relevant_docs, reference_answer)
# class Harness:
#     """Runs a pipeline config over a GoldenSet; scores retrieval and generation SEPARATELY,
#     plus a cost number. Calibrates any LLM judge before trusting it."""
#     @classmethod
#     def default(cls) -> "Harness": ...
#     def run(self, pipeline, golden: "GoldenSet") -> "Scorecard":
#         """Return stage-wise quality (nDCG@k, Recall@k; faithfulness, relevance, correctness) + cost."""
#         ...

__all__ = ["Harness", "GoldenSet"]  # populated in Phase 2
