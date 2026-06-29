"""professional_rag_kit.retrieval.query — query transformation, router-gated (Book Ch 6).

Improve the *query itself* before it is matched. Every transform here puts one or more LLM calls on
the retrieval critical path, so each must earn that latency — none is turned on globally; they fire
only on the slice the routing layer sends them, and only after measurement says they still beat your
embedder.

    hyde         Hypothetical Document Embeddings — write a fabricated answer and embed *that*, not
                 the terse query. Big lift (+16 to +32 nDCG@10) vs a weak/zero-shot embedder; shrinks
                 and can reverse against a modern fine-tuned one. A/B before shipping.
    step_back    Step-back prompting — abstract the question to its governing principle, then
                 retrieve on the abstraction. For reasoning-/principle-seeking corpora; skip factoids.
    multi_query  Generate paraphrases, retrieve each, fuse with RRF — buys recall when phrasing
                 variance is the measured recall problem.
    decompose    Split multi-hop / multi-part questions into atomic sub-queries — near-mandatory when
                 the answer is split across documents no single query retrieves.

The LLM call is made through the generation provider (the same ``GroundedGenerator``-style backend
the rest of professional_rag_kit uses), so swapping Claude↔GPT is a one-line provider change and no vendor SDK is
imported at module top level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Prompts kept faithful to the chapter's descriptions of each transform.
_HYDE_SYSTEM = (
    "You write a single hypothetical passage that, if it existed in a corpus, would directly answer "
    "the user's question. Write it as a confident, factual answer document — not a question, not a "
    "preamble. It is used only as a search probe (its embedding), never shown to a user, so plausible "
    "detail is fine. Output ONLY the passage, 2-4 sentences."
)
_STEP_BACK_SYSTEM = (
    "You are an expert at step-back prompting. Given a specific question, produce a single more "
    "general 'step-back' question about the underlying principle, concept, or relationship that the "
    "specific question is an instance of. Retrieving on the general question surfaces the governing "
    "principle. Output ONLY the step-back question."
)
_MULTI_QUERY_SYSTEM = (
    "You generate alternative phrasings of a search query to improve retrieval recall. Given a query "
    "and a count N, output N diverse paraphrases that preserve the information need but vary the "
    "vocabulary and phrasing. Output each paraphrase on its own line, no numbering, nothing else."
)
_DECOMPOSE_SYSTEM = (
    "You decompose a complex, multi-part or multi-hop question into the minimal set of atomic "
    "sub-questions, each answerable by a single retrieval. If the question is already atomic, return "
    "it unchanged as the only line. Output each sub-question on its own line, no numbering, nothing "
    "else."
)


def _split_lines(text: str) -> list[str]:
    """Parse a one-item-per-line LLM list, stripping bullets/numbering and blanks."""
    out: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line:
            out.append(line)
    return out


@dataclass
class QueryTransform:
    """Router-gated query transforms (Ch 6). Each adds an LLM call; route them on, don't enable globally.

    LLM-backed via the generation provider. ``default()`` uses the configured generation provider;
    construct with an explicit ``provider``/``model`` to pin one. The generation backend is resolved
    and called lazily inside each method, so this module imports with no SDK installed.
    """

    provider: str = ""  # "" → use the configured generation provider
    model: str = ""

    @classmethod
    def default(cls) -> "QueryTransform":
        return cls()

    @classmethod
    def from_provider(cls, provider: str, model: str = "") -> "QueryTransform":
        return cls(provider=provider, model=model)

    def _gen(self):
        """Resolve (backend, model) from the generation registry, falling back to the config."""
        from professional_rag_kit.core.config import ProviderRegistry, load_config

        cfg = load_config().generation
        provider = self.provider or cfg.provider
        model = self.model or cfg.model
        return ProviderRegistry.get("generation", provider), model

    def _call(self, system: str, prompt: str, max_tokens: int = 512) -> str:
        backend, model = self._gen()
        return backend(model, system, prompt, max_tokens).strip()

    def hyde(self, query: str) -> str:
        """Write a hypothetical answer document to embed in place of the query (Ch 6: HyDE).

        Best when the embedder is weak/zero-shot; A/B it against a strong fine-tuned embedder, where
        the advantage shrinks and can reverse. The returned text is a *search probe* (embed it); it
        is never shown to the user.
        """
        return self._call(_HYDE_SYSTEM, f"Question: {query}")

    def step_back(self, query: str) -> str:
        """Abstract the question to its governing principle, then retrieve on it (Ch 6: step-back).

        Helps reasoning-/principle-seeking corpora; skip on factoid lookups (gate it on a classifier).
        """
        return self._call(_STEP_BACK_SYSTEM, f"Specific question: {query}")

    def multi_query(self, query: str, n: int = 3) -> list[str]:
        """Generate ``n`` paraphrases to retrieve for and fuse with RRF (Ch 6: multi-query).

        Buys recall when phrasing variance is the measured recall problem; cost scales with ``n``.
        The original query is always included so fusion never loses the user's own phrasing.
        """
        if n <= 1:
            return [query]
        out = self._call(_MULTI_QUERY_SYSTEM, f"N = {n}\nQuery: {query}")
        variants = _split_lines(out)
        merged = [query] + [v for v in variants if v.lower() != query.lower()]
        return merged[:n] if len(merged) > n else merged

    def decompose(self, query: str) -> list[str]:
        """Split a multi-hop / multi-part question into atomic sub-queries (Ch 6: decomposition).

        Near-mandatory for genuine multi-hop traffic, pure overhead for single-fact lookups (route it
        on). Returns ``[query]`` unchanged when the model judges the question already atomic.
        """
        out = self._call(_DECOMPOSE_SYSTEM, f"Question: {query}")
        subs = _split_lines(out)
        return subs or [query]


__all__ = ["QueryTransform"]
