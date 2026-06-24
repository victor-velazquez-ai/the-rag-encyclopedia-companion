# %% [markdown]
# # Chapter 13 — Grounded generation: ground, cite, abstain
#
# The last component in the pipeline. Its job is narrow and committed: make the generator *use* the
# retrieved context, *attribute* each claim to the passage that supports it, and **abstain** when the
# evidence runs out (say "I don't know" rather than fill the gap from memory). The cross-cutting rule
# the chapter keeps returning to: **faithfulness is not correctness** - a perfectly grounded answer to
# bad context is confidently wrong.
#
# To run this fully offline (no API key), we register a *deterministic* generation backend in the
# provider registry and point `GroundedGenerator` at it. The real Claude-backed call shape is shown at
# the end in a guarded cell.
#
# Production code: [`ragkit.production.generation`](../../ragkit/production/generation/__init__.py)
# (`GroundedGenerator`). Book section 13: "Grounded generation".

# %%
from ragkit.core.config import ProviderRegistry
from ragkit.production.generation import GroundedGenerator

# Register a fake "demo" generation backend: signature is (model, system, prompt, max_tokens) -> str.
# It returns a fixed grounded answer with a citation, so the whole pipeline runs with no key.
ProviderRegistry.register("generation", "demo")(
    lambda model, system, prompt, maxtok: "The rate is 21% [1]."
)

passages = [
    "The standard corporate tax rate is 21% as of the 2017 reform.",
    "State-level surtaxes vary and are assessed separately.",
]

# %% [markdown]
# ## Grounding + citation
# `GroundedGenerator` formats the passages as a numbered context, calls the backend with the grounded
# system prompt (use ONLY the passages, cite passage numbers, abstain if unsupported), and parses the
# `[n]` citations out of the returned text. The citation list is what makes an answer auditable.

# %%
gen = GroundedGenerator(provider="demo", model="x")
ans = gen.generate("What is the corporate tax rate?", passages)

print("text:     ", ans.text)
print("citations:", ans.citations)
print("abstained:", ans.abstained)
assert ans.citations == [1]
assert ans.abstained is False

# %% [markdown]
# ## Abstention - the sufficiency gate
# When the passages do not contain the answer, the right output is not a plausible guess; it is a
# refusal. We swap in a backend that returns the exact abstention sentinel, and the generator flags
# `abstained=True`. In production this is the single most valuable behavior: a wrong answer costs more
# than no answer in every enterprise setting.

# %%
from ragkit.production.generation import ABSTENTION

ProviderRegistry.register("generation", "demo_abstain")(
    lambda model, system, prompt, maxtok: ABSTENTION
)
abstainer = GroundedGenerator(provider="demo_abstain", model="x")
ans2 = abstainer.generate("What is the dividend yield?", passages)  # not in the passages

print("text:     ", ans2.text)
print("citations:", ans2.citations)
print("abstained:", ans2.abstained)
assert ans2.abstained is True
assert ans2.citations == []

# %% [markdown]
# ## The real call (guarded)
# `GroundedGenerator.default()` uses Claude (`claude-opus-4-8`) via the Anthropic SDK. It needs
# `pip install anthropic` and `ANTHROPIC_API_KEY` set, so we guard it - the offline cells above prove
# the grounding/citation/abstention *logic*; this proves the wiring. The pipeline code is identical;
# only the provider field changes (one-line swap to OpenAI GPT).

# %%
import os

if os.environ.get("ANTHROPIC_API_KEY"):
    real = GroundedGenerator.default()  # provider="anthropic", model="claude-opus-4-8"
    real_ans = real.generate("What is the corporate tax rate?", passages)
    print("text:     ", real_ans.text)
    print("citations:", real_ans.citations)
else:
    print("ANTHROPIC_API_KEY not set - skipping the live Claude call.")
    print("Set it and run GroundedGenerator.default().generate(query, passages) for the real thing.")

# %% [markdown]
# ## The lesson
# Grounding, citation, and abstention are three behaviors of one component, and all three are
# *measured separately from correctness*. Every technique that makes the model trust the context more
# (Chain-of-Note, Context-Aware Decoding - the chapter's deeper layers) also makes it trust **bad**
# context more. So faithfulness without retrieval quality is a sharper knife pointed the wrong way:
# report faithfulness AND answer correctness, never one as a proxy for the other.

# %%
print("\nground -> cite -> abstain: but faithfulness is not correctness. Measure both (Ch 14).")
