"""professional_rag_kit.production.serving — throughput, latency, and cost levers (Book Ch 16).

The generation stage dominates the latency budget (1-5 s) and the bill (retrieved context is most of
the input tokens), so every lever here points at it: skip the call entirely (semantic cache), make a
cheaper call where the query allows (model routing), or make the call cacheable by the provider
(stable-prefix prompt discipline).

    SemanticCache  embedding-proximity cache in front of the whole pipeline — a hit skips retrieval,
                   rerank, AND the LLM call. Budget the speedup at 2-10x on a hit, never the marketing
                   100x, and size savings by hit rate (target 30%+). The pure ``lookup`` is numpy
                   cosine; ``get``/``put`` embed lazily via ``Embedder``.
    ModelRouter    route each query to a cheap vs strong model by a complexity heuristic (RouteLLM /
                   FrugalGPT family). Most queries are easy; the strong tier must *earn* its place
                   per query. A *cost* decision about which generator (retrieval routing is Ch 10).
    stable_prefix  the one architectural rule of prompt caching — STABLE PREFIX FIRST (system + tools
                   + recurring reference docs), per-query suffix last. Get the order backwards and you
                   cache nothing, because the cache keys on the prompt prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence


# --- Semantic cache ----------------------------------------------------------
def lookup(
    query_vec: Sequence[float],
    entries: Sequence[tuple[Sequence[float], Any]],
    threshold: float,
) -> Optional[Any]:
    """Pure cosine-proximity cache lookup (Ch 16). Return the best entry's response or ``None``.

    ``entries`` is ``[(vec, response), ...]``; return the response of the entry whose cached vector is
    most cosine-similar to ``query_vec`` *iff* that similarity is at least ``threshold``, else
    ``None`` (a miss).

    vCache caveat (arXiv:2502.03771): a single *static* threshold has NO correctness guarantee. The
    similarity distribution of pairs where the cached answer is still correct OVERLAPS the
    distribution where it is wrong — a grey zone where one global number cannot separate a safe hit
    from a confidently-wrong one. The production fix is a LEARNED, PER-PROMPT, error-bounded threshold
    (vCache), paired with a TTL and event-based invalidation. This pure helper is the static baseline;
    treat its ``threshold`` as a correctness control, not a tuning knob.
    """
    import numpy as np

    if not entries:
        return None
    q = np.asarray(query_vec, dtype=float)
    qn = np.linalg.norm(q)
    if qn == 0:
        return None
    best_sim = -1.0
    best_response: Any = None
    for vec, response in entries:
        v = np.asarray(vec, dtype=float)
        vn = np.linalg.norm(v)
        if vn == 0:
            continue
        sim = float(q @ v / (qn * vn))
        if sim > best_sim:
            best_sim, best_response = sim, response
    return best_response if best_sim >= threshold else None


@dataclass
class SemanticCache:
    """Embedding-proximity cache in front of the pipeline (Ch 16). ``get``/``put`` embed lazily.

    A hit skips retrieval, reranking, and the (dominant-cost) LLM call at once. Ship only at a measured
    hit rate of 30%+; below ~15% the operational surface is not worth it. Budget 2-10x on a hit, never
    the marketing 100x.

    The threshold is a CORRECTNESS control, not a tuning knob: a single static value (this default)
    sits in a grey zone where correct and incorrect similarity distributions overlap, so it can serve
    a confidently-wrong cached answer. The reference fix is vCache's learned, per-prompt, error-bounded
    threshold (arXiv:2502.03771); always pair the cache with a TTL plus event-based invalidation where
    answer-to-source provenance is traceable, since a cached answer right in March is wrong in June.

    The embedder is lazy: construct without one and a default ``Embedder`` (OpenAI text-embedding-3-
    large, bring-your-own-key) is built on first ``get``/``put``. So this module imports with no SDK
    installed; only an actual embed call needs a key.
    """

    threshold: float = 0.85
    embedder: Any = None
    _entries: list[tuple[list[float], Any]] = field(default_factory=list)

    def _embed(self, query: str) -> list[float]:
        if self.embedder is None:
            from professional_rag_kit.ingestion.embedding import Embedder  # lazy: no key needed to import

            self.embedder = Embedder.default()
        return self.embedder.embed_query(query)

    def get(self, query: str) -> Optional[Any]:
        """Embed ``query`` (lazy) and return a cached response within ``threshold``, else ``None``."""
        return lookup(self._embed(query), self._entries, self.threshold)

    def put(self, query: str, response: Any) -> None:
        """Embed ``query`` (lazy) and store ``(vec, response)`` for future proximity hits."""
        self._entries.append((self._embed(query), response))


# --- Model routing -----------------------------------------------------------
@dataclass
class ModelRouter:
    """Route a query to a cheap vs strong model by a complexity heuristic (Ch 16, RouteLLM/FrugalGPT).

    Most RAG queries are easy and a small model answers them correctly; a minority are hard and need
    the frontier tier. Sending everything to the strong model pays frontier prices for questions the
    cheap model would have nailed. Practitioners report 45-85% cost cuts at ~95% quality from routing;
    the 98% headline (FrugalGPT) is a benchmark ceiling, not the routine result.

    Pure heuristic (no model call, so it is the *upfront-router* shape, not the cascade): score
    complexity from length, multi-part structure, and reasoning/comparison cues, and route to the
    strong tier when the score crosses ``threshold``. This is a *cost* decision about which generator;
    retrieval routing (which index/strategy) is Ch 10's concern. Swap in a learned router for
    production — the heuristic is the dependency-free default.
    """

    cheap_model: str = "claude-haiku-4-5"
    strong_model: str = "claude-opus-4-8"
    threshold: float = 1.0

    _REASONING_CUES = (
        "why", "how", "compare", "contrast", "explain", "analyze", "analyse",
        "trade-off", "tradeoff", "implications", "step by step", "step-by-step",
        "difference between", "pros and cons", "evaluate", "derive", "prove",
    )

    def complexity(self, query: str) -> float:
        """A pure 0-up complexity score: longer, multi-part, reasoning-laden queries score higher."""
        q = query.lower()
        words = query.split()
        score = 0.0
        if len(words) > 25:
            score += 1.0
        elif len(words) > 12:
            score += 0.5
        # multi-part questions (several '?' or conjunctions) are harder
        if query.count("?") > 1:
            score += 0.5
        if " and " in q or "; " in query:
            score += 0.25
        score += sum(0.6 for cue in self._REASONING_CUES if cue in q)
        return score

    def route(self, query: str) -> str:
        """Return the model id to use: ``strong_model`` if complex enough, else ``cheap_model``."""
        return self.strong_model if self.complexity(query) >= self.threshold else self.cheap_model


# --- Prompt-cache prefix discipline ------------------------------------------
def stable_prefix(system: str, tools: Sequence[str] | None = None) -> str:
    """Assemble the prompt's STABLE PREFIX — system instructions then tools (Ch 16, prompt caching).

    The one architectural rule of prompt caching: put everything that is constant across requests
    FIRST (system instructions, tool definitions, recurring reference docs), and the per-query suffix
    (the user question + per-request retrieved chunks) LAST. The cache keys on the prompt prefix, so a
    request-varying token anywhere in the prefix busts the cache for everything after it — get the
    order backwards and you cache nothing.

    On a RAG workload the cached prefix is most of the input tokens and a cache read is steeply
    discounted (Anthropic ~0.1x; re-pull provider discounts live and dated — they move), so this
    one-line ordering decision is the biggest single cost lever in the chapter. This helper returns the
    prefix string; concatenate the per-query suffix *after* it, never before.
    """
    blocks = [system.strip()]
    for tool in tools or ():
        blocks.append(tool.strip())
    return "\n\n".join(b for b in blocks if b)


__all__ = ["SemanticCache", "ModelRouter", "lookup", "stable_prefix"]
