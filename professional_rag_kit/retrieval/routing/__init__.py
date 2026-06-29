"""professional_rag_kit.retrieval.routing — the cost optimizer: spend expensive paths only where they pay (Book Ch 6).

Not every query deserves the same machinery. A cheap classifier inspects the incoming query *before*
any expensive work and picks its path. Framed correctly, routing is a cost optimizer that frequently
improves quality as a side effect — by keeping heavy machinery off the easy queries it would only
overcomplicate.

    SemanticRouter        Embed the query + k-NN against curated per-route utterances. No LLM call in
                          the decision: the chapter's ~5000 ms (LLM router) -> ~100 ms (embed + k-NN),
                          scaling to thousands of routes. As good as its curated utterances.
    ComplexityClassifier  Adaptive-RAG complexity routing: no_retrieval / single / multi. A simple
                          query skips retrieval; only true multi-hop pays for the Part III loops. The
                          chapter's rule: BIAS TOWARD RETRIEVING when uncertain — a hard question
                          misrouted to no_retrieval fails silently, while an easy one misrouted to
                          multi merely wastes spend.

``ComplexityClassifier`` and ``route_by_vectors`` are PURE (no API/network) and unit-tested;
``SemanticRouter.route`` lazily uses the embedding ``Embedder`` so this module imports with no SDK.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

# --- ComplexityClassifier (PURE heuristic, Ch 6 / Adaptive-RAG) ---------------
# Markers that signal a query spans several facts/hops and so deserves the multi-step path.
_MULTI_HOP_MARKERS = (
    "and then",
    "compare",
    "comparison",
    "difference between",
    "versus",
    " vs ",
    "vs.",
    "relationship between",
    "both",
    "each of",
    "as well as",
    "after which",
    "before that",
    "trace the",
    "step by step",
)
_QUESTION_WORDS = ("who", "what", "when", "where", "which", "why", "how", "whom", "whose")
# Cues that the model likely already knows the answer with no retrieval (chit-chat / pure compute /
# definitional small talk). Kept deliberately narrow because the chapter biases toward retrieving.
_NO_RETRIEVAL_CUES = (
    "hello",
    "hi there",
    "thank you",
    "thanks",
    "good morning",
    "how are you",
    "who are you",
    "what can you do",
    "tell me a joke",
)
_WORD = re.compile(r"\w+")
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")


@dataclass
class ComplexityClassifier:
    """Adaptive-RAG complexity routing (Ch 6), PURE heuristic over surface cues.

    Maps a query to ``"no_retrieval" | "single" | "multi"``. Cues: chit-chat/greeting markers →
    ``no_retrieval``; multi-hop markers (``"and then"``, ``"compare"``, ...), several capitalized
    entities, or multiple coordinated clauses → ``"multi"``; otherwise ``"single"``. Per the
    chapter, ambiguity resolves toward *more* retrieval, never less (silent failure beats waste).
    """

    multi_entity_threshold: int = 2  # >= this many distinct capitalized entities hints multi-hop

    def classify(self, query: str) -> str:
        q = query.strip()
        low = q.lower()
        if not q:
            return "no_retrieval"

        # 1) no_retrieval: only for unambiguous chit-chat/greetings, and only when the query is short
        #    and carries no question word (bias is toward retrieving, so keep this gate tight).
        words = _WORD.findall(low)
        has_question = low.endswith("?") or any(w in _QUESTION_WORDS for w in words[:3]) or any(
            w in _QUESTION_WORDS for w in words
        )
        if len(words) <= 6 and not has_question and any(cue in low for cue in _NO_RETRIEVAL_CUES):
            return "no_retrieval"

        # 2) multi: explicit multi-hop markers, several entities, or several coordinated clauses.
        if any(marker in low for marker in _MULTI_HOP_MARKERS):
            return "multi"
        entities = {e.lower() for e in _ENTITY.findall(q)}
        if len(entities) >= self.multi_entity_threshold:
            return "multi"
        # multiple "and"/"," coordinations + question words suggest several sub-questions in one.
        conjunctions = low.count(" and ") + low.count(";") + low.count(" as well as ")
        question_count = sum(low.count(w) for w in ("what", "who", "when", "where", "how", "why"))
        if conjunctions >= 1 and question_count >= 2:
            return "multi"

        # 3) default: a normal single-step retrieval (the safe, common case).
        return "single"


# --- route_by_vectors (PURE cosine k-NN, Ch 6 semantic routing) ---------------
def route_by_vectors(
    query_vec: Sequence[float],
    route_vecs: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    k: int = 1,
) -> str:
    """Route a query to the nearest labeled route by cosine k-NN (Ch 6 semantic router, PURE numpy).

    ``route_vecs[i]`` is a curated-utterance embedding with intent ``labels[i]`` (utterances may
    repeat a label). Returns the label of the single nearest utterance (``k=1``), or the majority
    label among the top-``k`` nearest (ties broken by closest-neighbor). No LLM call — this is the
    decision the chapter measures at ~100 ms vs an LLM router's ~5000 ms.
    """
    import numpy as np

    if len(route_vecs) == 0:
        raise ValueError("route_by_vectors needs at least one route vector.")
    if len(route_vecs) != len(labels):
        raise ValueError("route_vecs and labels must be the same length.")

    q = np.asarray(query_vec, dtype=float)
    r = np.asarray(route_vecs, dtype=float)

    def _norm(m, axis):
        n = np.linalg.norm(m, axis=axis, keepdims=True)
        return m / np.where(n == 0, 1, n)

    qn = _norm(q.reshape(1, -1), 1)[0]
    rn = _norm(r, 1)
    sims = rn @ qn  # cosine similarity to every utterance

    order = list(np.argsort(-sims))  # nearest first
    if k <= 1:
        return labels[order[0]]

    top = order[: min(k, len(order))]
    votes: dict[str, float] = {}
    for rank, i in enumerate(top):
        # majority vote; use descending sim as a tiny tiebreak weight so the nearest wins ties.
        votes[labels[i]] = votes.get(labels[i], 0.0) + 1.0 + (len(top) - rank) * 1e-6
    return max(votes, key=votes.get)


@dataclass
class SemanticRouter:
    """Embed-and-k-NN intent router over curated utterances (Ch 6), no LLM call in the decision.

    Seed with ``add_route(label, utterances)``; ``route(query)`` lazily embeds via the configured
    ``Embedder`` (Ch 4) and delegates to the PURE ``route_by_vectors``. The embedding step is the
    only model-dependent part — import-verified only — and the utterance index is cached after the
    first ``route`` call so re-seeding requires a fresh router or ``reset_index()``.
    """

    k: int = 1
    _labels: list[str] = field(default_factory=list)
    _utterances: list[str] = field(default_factory=list)
    _vectors: list[list[float]] | None = field(default=None, repr=False)

    def add_route(self, label: str, utterances: Sequence[str]) -> "SemanticRouter":
        """Register a route as a label plus representative example utterances (Ch 6)."""
        for u in utterances:
            self._labels.append(label)
            self._utterances.append(u)
        self._vectors = None  # invalidate any cached index
        return self

    def reset_index(self) -> None:
        self._vectors = None

    def _ensure_index(self) -> None:
        if self._vectors is not None:
            return
        if not self._utterances:
            raise ValueError("SemanticRouter has no routes; call add_route(...) first.")
        from professional_rag_kit.ingestion.embedding import Embedder  # lazy: no SDK at import time

        self._vectors = Embedder.default().embed_documents(self._utterances)

    def route(self, query: str) -> str:
        """Embed ``query`` and return the nearest route's label (Ch 6, LLM-free decision)."""
        self._ensure_index()
        from professional_rag_kit.ingestion.embedding import Embedder

        query_vec = Embedder.default().embed_query(query)
        assert self._vectors is not None
        return route_by_vectors(query_vec, self._vectors, self._labels, k=self.k)


__all__ = ["ComplexityClassifier", "SemanticRouter", "route_by_vectors"]
