"""ragkit.retrieval.rerank — reranking tiers (Book Ch 7).

The cheap-recall → precise-rerank cascade: a first stage casts a wide net, an expensive model
re-scores only the survivors. One ``Reranker`` facade; escalate tiers only when measurement says the
ceiling binds — and remember the chapter's warning that the reranker is often 60–84% of
retrieval-pipeline latency, so its candidate count is your main cost knob.

Bring-your-own-key default: **LLM listwise reranking** (RankZephyr-style) using your generation
provider — it reuses the Claude/GPT key you already have, no extra model to host. **Cohere Rerank**
is a one-line swap. The book's self-host cross-encoder (jina-reranker-v3) is a ``[selfhost]`` tier.

SDK imports are lazy; this module imports without any SDK installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragkit.core.config import ProviderRegistry, load_config

_LISTWISE_SYSTEM = (
    "You are a search reranker. Given a query and numbered passages, output the passage numbers "
    "ordered from most to least relevant to the query, as a comma-separated list (e.g. 3,1,2). "
    "Include every number exactly once. Output ONLY the list."
)


@dataclass
class Reranker:
    """Facade over the reranking tiers. ``default()`` → LLM listwise (reuses the generation provider)."""

    provider: str = "llm"
    gen_provider: str = ""  # for the llm tier; defaults to the configured generation provider
    gen_model: str = ""

    @classmethod
    def default(cls) -> "Reranker":
        return cls()

    @classmethod
    def from_provider(cls, provider: str) -> "Reranker":  # "llm" | "cohere" | "jina"
        return cls(provider=provider)

    def _gen(self):
        gp = self.gen_provider or load_config().generation.provider
        gm = self.gen_model or load_config().generation.model
        return ProviderRegistry.get("generation", gp), gm

    def rerank(self, query: str, candidates: list[str], top_k: int | None = None) -> list[str]:
        """Re-score ``candidates`` against ``query``; return them best-first (top_k if given)."""
        if not candidates:
            return []
        if self.provider == "llm":
            order = self._rerank_llm(query, candidates)
        elif self.provider == "cohere":
            order = _rerank_cohere(query, candidates)
        elif self.provider == "jina":
            order = _rerank_jina(query, candidates)
        else:
            raise ValueError(f"Unknown reranker provider '{self.provider}'")
        ranked = [candidates[i] for i in order]
        return ranked[:top_k] if top_k else ranked

    def _rerank_llm(self, query: str, candidates: list[str]) -> list[int]:
        backend, model = self._gen()
        listing = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(candidates))
        prompt = f"Query: {query}\n\nPassages:\n{listing}"
        out = backend(model, _LISTWISE_SYSTEM, prompt, 256)
        nums = [int(n) - 1 for n in re.findall(r"\d+", out)]
        # keep valid, de-duplicate, then append any the model dropped (stable fallback)
        seen, order = set(), []
        for i in nums:
            if 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                order.append(i)
        order += [i for i in range(len(candidates)) if i not in seen]
        return order


def _rerank_cohere(query: str, candidates: list[str]) -> list[int]:
    import cohere  # lazy

    client = cohere.ClientV2()
    res = client.rerank(model="rerank-v3.5", query=query, documents=candidates)
    return [r.index for r in res.results]


def _rerank_jina(query: str, candidates: list[str]) -> list[int]:
    from sentence_transformers import CrossEncoder  # lazy ([selfhost])

    ce = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)
    scores = ce.predict([(query, c) for c in candidates])
    return sorted(range(len(candidates)), key=lambda i: -scores[i])


__all__ = ["Reranker"]
