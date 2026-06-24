"""ragkit.production.generation — grounded generation, citation, and abstention (Book Ch 13).

The last component in the pipeline, and the one that inherits every upstream decision. Its job is
narrow and committed: make the generator *use* the retrieved context, *attribute* each claim to the
span that supports it, and *stop* — abstain — when the evidence runs out. A four-layer defense,
cheapest and most general first:

    prompting.py    context-faithful prompting (opinion-based framing + counterfactual demos) —
                    the free default every generator starts from; weakest, never the only line
    chain_of_note.py  per-document reading notes before answering — the noisy-retrieval filter
                      (+7.9 EM in fully-noisy settings); also lifts unanswerable rejection
    cad.py          Context-Aware Decoding — training-free, logit-level contrastive decoding that
                    amplifies the context (+14.3% factuality); needs self-hosted logit access, and
                    amplifies *wrong* context as faithfully as right — a retrieval-quality multiplier
    citation.py     span-level citation, scored ALCE-style (recall + precision) via an NLI judge
                    read as a high-precision, LOW-RECALL instrument (recall is a lower bound)
    abstention.py   sufficiency gate → first-class "the sources do not support an answer" output,
                    calibrated as a product decision, evaluated under a hallucination-penalty scheme
    conflict.py     conflicting/insufficient evidence: source-reliability weighting (counter the
                    base model's coherent-but-wrong bias), surface disagreement, else abstain

The cross-cutting rule the facade encodes: *faithfulness is not correctness.* Measure them
separately — the technique that raises one (CAD most starkly) can lower the other.

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class GroundedGenerator:
#     """Facade over the grounding stack. `default()` = context-faithful prompting + span citation
#     + sufficiency-gated abstention; CAD and Chain-of-Note layered on by config."""
#     @classmethod
#     def default(cls) -> "GroundedGenerator": ...
#     def generate(self, query: str, passages: list, *, cite: bool = True) -> "GroundedAnswer":
#         """Answer from `passages`; attach span-level citations; abstain if support is insufficient."""
#         ...
# class GroundedAnswer:
#     text: str
#     citations: list          # span -> supporting passage span
#     abstained: bool          # True when the sufficiency gate fired
#     conflict: str | None     # set when sources disagreed and were surfaced, not silently resolved

__all__ = ["GroundedGenerator", "GroundedAnswer"]  # populated in Phase 2
