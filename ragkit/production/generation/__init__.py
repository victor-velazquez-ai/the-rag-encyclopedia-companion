"""ragkit.production.generation — grounded generation, citation, and abstention (Book Ch 13).

The last component in the pipeline. Its job is narrow and committed: make the generator *use* the
retrieved context, *attribute* each claim to the passage that supports it, and *abstain* when the
evidence runs out (say "I don't know" rather than fill the gap from memory). The cross-cutting rule
the facade encodes is the chapter's: *faithfulness is not correctness* — measure them separately.

Bring-your-own-key: defaults to Claude (``claude-opus-4-8``) via the Anthropic SDK; OpenAI GPT is a
one-line provider swap. SDK imports are lazy, so this module imports fine without an SDK installed.
The book's deeper layers (Chain-of-Note, Context-Aware Decoding) build on this base — see the
chapter; CAD in particular needs self-hosted logit access and is out of scope for the API stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragkit.core.config import ProviderRegistry, load_config

ABSTENTION = "I don't have enough information to answer that."

# Ch 13: context-faithful prompting + span citation + sufficiency-gated abstention, in one prompt.
GROUNDED_SYSTEM = (
    "You answer strictly from the provided context passages. Rules:\n"
    "1. Use ONLY the numbered passages. Never add facts from prior knowledge.\n"
    "2. Cite the supporting passage number(s) in square brackets after each claim, e.g. [2].\n"
    f'3. If the passages do not contain the answer, reply exactly: "{ABSTENTION}" Do not guess.\n'
    "4. If passages conflict, say so explicitly and prefer the more authoritative or recent source."
)


@dataclass
class GroundedAnswer:
    text: str
    citations: list[int] = field(default_factory=list)  # passage numbers cited in the text
    abstained: bool = False


def _format_context(passages: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))


def _parse_citations(text: str) -> list[int]:
    import re

    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})


@dataclass
class GroundedGenerator:
    """Facade over the generation providers. ``default()`` → Claude (claude-opus-4-8)."""

    provider: str = "anthropic"
    model: str = "claude-opus-4-8"

    @classmethod
    def default(cls) -> "GroundedGenerator":
        return cls()

    @classmethod
    def from_provider(cls, provider: str, model: str = "") -> "GroundedGenerator":
        model = model or {"anthropic": "claude-opus-4-8", "openai": "gpt-4o"}.get(provider, "")
        return cls(provider=provider, model=model)

    @classmethod
    def from_config(cls, path: str | None = None) -> "GroundedGenerator":
        c = load_config(path).generation
        return cls.from_provider(c.provider, c.model)

    def generate(self, query: str, passages: list[str], *, max_tokens: int = 2048) -> GroundedAnswer:
        """Answer ``query`` grounded in ``passages``; cite spans; abstain if support is insufficient."""
        backend = ProviderRegistry.get("generation", self.provider)
        prompt = f"Context:\n{_format_context(passages)}\n\nQuestion: {query}"
        text = backend(self.model, GROUNDED_SYSTEM, prompt, max_tokens).strip()
        return GroundedAnswer(
            text=text,
            citations=_parse_citations(text),
            abstained=ABSTENTION.rstrip(".").lower() in text.lower(),
        )


# --- Backends (registered with the provider registry) ------------------------
@ProviderRegistry.register("generation", "anthropic")
def _anthropic_generate(model: str, system: str, prompt: str, max_tokens: int) -> str:
    import anthropic  # lazy — needs `pip install anthropic` + ANTHROPIC_API_KEY

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model or "claude-opus-4-8",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


@ProviderRegistry.register("generation", "openai")
def _openai_generate(model: str, system: str, prompt: str, max_tokens: int) -> str:
    import openai  # lazy — needs `pip install openai` + OPENAI_API_KEY

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model or "gpt-4o",
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


__all__ = ["GroundedGenerator", "GroundedAnswer", "GROUNDED_SYSTEM", "ABSTENTION"]
