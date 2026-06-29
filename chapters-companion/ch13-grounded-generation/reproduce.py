"""Chapter 13 reproduction — grounding and abstention (offline fake backend, no API key).

The chapter's central claim is that grounded generation is three behaviors of one component: use the
context, cite the supporting passage, and ABSTAIN when the evidence is insufficient. This runnable
check exercises all three offline by registering a deterministic generation backend in the provider
registry - no key, no SDK call - so the grounding/citation/abstention *logic* is verified end to end.

The keyed version (faithfulness vs. correctness deltas, the counterfactual-context probe) swaps the
backend for Claude; the offline slice proves the wiring those deltas ride on.

    python chapters-companion/ch13-grounded-generation/reproduce.py
"""

from __future__ import annotations

from professional_rag_kit.core.config import ProviderRegistry
from professional_rag_kit.production.generation import ABSTENTION, GroundedGenerator

# A deterministic backend: when the *question* asks about the tax rate it cites [1]; otherwise it
# abstains. We key on the "Question:" line (the prompt also embeds the passages, which mention the
# rate). Signature matches a real provider: (model, system, prompt, max_tokens) -> str.
def _repro_backend(model: str, system: str, prompt: str, maxtok: int) -> str:
    question = prompt.rsplit("Question:", 1)[-1].lower()
    return "The corporate tax rate is 21% [1]." if "tax rate" in question else ABSTENTION


ProviderRegistry.register("generation", "repro")(_repro_backend)

_PASSAGES = [
    "The standard corporate tax rate is 21% as of the 2017 reform.",
    "State-level surtaxes vary and are assessed separately.",
]


def main() -> None:
    gen = GroundedGenerator(provider="repro", model="x")

    print("Chapter 13 - grounding + citation + abstention (offline)")
    print("-" * 56)

    # 1) supported question -> grounded, cited, not abstained
    supported = gen.generate("What is the corporate tax rate?", _PASSAGES)
    print(f"supported   -> '{supported.text}'")
    print(f"             citations={supported.citations} abstained={supported.abstained}")
    assert supported.citations == [1]
    assert supported.abstained is False

    # 2) unsupported question -> abstains, no citation
    unsupported = gen.generate("What is the dividend yield?", _PASSAGES)
    print(f"unsupported -> '{unsupported.text}'")
    print(f"             citations={unsupported.citations} abstained={unsupported.abstained}")
    assert unsupported.abstained is True
    assert unsupported.citations == []

    print("-" * 56)
    print("PASS - grounds + cites when supported, abstains when not.")
    print("Reminder: faithfulness is not correctness; the keyed suite measures both.")


if __name__ == "__main__":
    main()
