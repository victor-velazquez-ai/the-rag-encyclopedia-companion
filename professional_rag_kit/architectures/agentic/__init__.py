"""professional_rag_kit.architectures.agentic — multi-hop orchestration across hops and sources (Book Ch 11).

Three layers most RAG content conflates, kept deliberately separate here. *Loop* (agentic
multi-hop) only for the compositional minority, gated by the complexity classifier and capped by a
turn budget. *Wrap* (MCP) only when sources or consumers exceed one. *Resolve* (canonical-ID entity
resolution) *always* once the corpus crosses source systems — it is the precision substrate under
multi-source and graph RAG, not an optional nicety.

This module surfaces three classes:

    EntityResolver  the canonical-ID layer (PURE, fully tested). Deterministic/exact match first
                    (normalize → exact alias lookup), THEN fuzzy (difflib SequenceMatcher ratio ≥
                    threshold) — the chapter's "deterministic-then-Fellegi-Sunter" ordering, with a
                    review band routing the uncertain middle to humans. Optional first-token blocking.
    AgenticRAG      the plan-act-reflect loop (lazy, generator-driven). Decompose → retrieve →
                    reflect("enough? / turn budget?") → loop or synthesize, capped at a hard turn
                    budget. Self-Ask and IRCoT multi-hop helpers. Gate it behind a complexity check
                    (conceptually the Adaptive-RAG classifier of Ch 10) — never the default path.
    MCPSource       a thin, uniform ``search(query) -> list`` boundary over an external source — the
                    MCP tool-boundary idea (one client-host-server contract) without a hard MCP dep.

Pure logic (EntityResolver) is fully implemented and tested; the loop/MCP surfaces are lazy and
inject their retriever + generator, so this module imports with no heavy deps and no SDK installed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# --- EntityResolver: the canonical-ID layer (PURE) ---------------------------

# Legal suffixes stripped during normalization so "Acme Inc." and "Acme" collide deterministically
# (Ch 11: "normalization lives in the deterministic stage — it turns many fuzzy cases into exact ones").
_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "llc", "llp", "lp", "plc", "gmbh", "ag", "sa", "nv", "bv", "pty", "pte", "srl", "spa",
}


def normalize(name: str) -> str:
    """Canonicalize a surface form for deterministic matching (Ch 11, the deterministic stage).

    Lowercase, strip punctuation, drop trailing legal suffixes ("Inc.", "Corp.", "GmbH"), and
    collapse whitespace — so "Acme, Inc.", "ACME Corporation" and "  acme  corp " all collapse to a
    single comparable key ("acme"). Acronyms that differ only in spacing/dots ("I.B.M." vs "IBM")
    survive as distinct keys here ("i b m" vs "ibm") and are reconciled by the fuzzy stage.
    """
    s = name.casefold()
    s = re.sub(r"[^\w\s]", " ", s)  # punctuation → space ("I.B.M." → "i b m")
    tokens = s.split()
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _fuzzy_ratio(a: str, b: str) -> float:
    """Similarity for the fuzzy stage: max of the spaced ratio and the despaced ratio.

    Comparing both forms makes the matcher robust to the acronym spacing the deterministic stage
    leaves behind — "i b m" vs "ibm" scores 1.0 despaced, so "I.B.M." resolves to "IBM".
    """
    spaced = SequenceMatcher(None, a, b).ratio()
    da, db = a.replace(" ", ""), b.replace(" ", "")
    despaced = SequenceMatcher(None, da, db).ratio()
    return max(spaced, despaced)


@dataclass
class ResolutionResult:
    """Outcome of a resolve call: the canonical id (or None), how it matched, and the score.

    ``method`` is one of "deterministic", "fuzzy", "review", or "none"; ``review=True`` flags a
    match in the uncertain band that the Fellegi-Sunter three-way decision routes to a human.
    """

    canonical_id: str | None
    method: str
    score: float = 0.0
    review: bool = False
    matched_alias: str | None = None


class EntityResolver:
    """Canonical-ID crosswalk: map every surface form of an entity to one canonical id (Ch 11).

    The matcher runs the chapter's non-negotiable ordering — *deterministic / exact first, then
    fuzzy / probabilistic for the remainder* — a Fellegi-Sunter-style stack:

    1. **Deterministic.** Normalize the mention (case/space/punct, legal-suffix stripping) and look
       it up against normalized aliases. Exact agreement is cheap, exact, and auditable.
    2. **Fuzzy.** For what deterministic leaves unresolved, score normalized similarity with
       ``difflib.SequenceMatcher`` and accept when the ratio ≥ ``threshold`` (default 0.85).
    3. **Review band.** Ratios in ``[review_low, threshold)`` are returned as *uncertain* (the
       three-way match / non-match / review decision) rather than forced into a guess — the lever
       the chapter says to tune toward fearing over-merging more than under-merging.

    Optional **blocking** by first normalized token mimics the ER pipeline's scalability stage: only
    aliases sharing the mention's first token are compared in the fuzzy pass.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        review_low: float = 0.75,
        *,
        blocking: bool = False,
    ) -> None:
        if not 0.0 <= review_low <= threshold <= 1.0:
            raise ValueError("require 0 <= review_low <= threshold <= 1")
        self.threshold = threshold
        self.review_low = review_low
        self.blocking = blocking
        # normalized alias -> canonical_id (the crosswalk)
        self._alias_to_id: dict[str, str] = {}
        # canonical_id -> its set of normalized aliases (kept for introspection)
        self._id_to_aliases: dict[str, set[str]] = {}

    # -- building the crosswalk ------------------------------------------------
    def add_canonical(self, canonical_id: str, *aliases: str) -> "EntityResolver":
        """Register ``canonical_id`` and its surface forms (the id itself is always an alias)."""
        forms = [canonical_id, *aliases]
        bucket = self._id_to_aliases.setdefault(canonical_id, set())
        for form in forms:
            key = normalize(form)
            if not key:
                continue
            self._alias_to_id[key] = canonical_id
            bucket.add(key)
        return self

    @property
    def canonical_ids(self) -> list[str]:
        return sorted(self._id_to_aliases)

    # -- resolving -------------------------------------------------------------
    def resolve(self, name: str) -> str | None:
        """Map a surface form to its canonical id, or ``None`` if no confident match (review excluded).

        Convenience over :meth:`resolve_detail`: returns the id for deterministic and fuzzy hits and
        ``None`` for misses *and* for review-band matches (an uncertain match is not a decision).
        """
        res = self.resolve_detail(name)
        return res.canonical_id if res.method in ("deterministic", "fuzzy") else None

    def resolve_detail(self, name: str) -> ResolutionResult:
        """Full resolution outcome: id, method, score, and review flag (Ch 11 three-way decision)."""
        key = normalize(name)
        if not key:
            return ResolutionResult(None, "none")

        # 1) deterministic / exact: normalized alias hit
        hit = self._alias_to_id.get(key)
        if hit is not None:
            return ResolutionResult(hit, "deterministic", 1.0, matched_alias=key)

        # 2) fuzzy: best SequenceMatcher ratio over (optionally blocked) candidate aliases
        block = key.split()[0] if self.blocking else None
        best_id: str | None = None
        best_alias: str | None = None
        best_score = 0.0
        for alias, cid in self._alias_to_id.items():
            if block is not None and not alias.startswith(block):
                continue
            score = _fuzzy_ratio(key, alias)
            if score > best_score:
                best_id, best_alias, best_score = cid, alias, score

        if best_id is None:
            return ResolutionResult(None, "none")
        if best_score >= self.threshold:
            return ResolutionResult(best_id, "fuzzy", best_score, matched_alias=best_alias)
        if best_score >= self.review_low:
            # uncertain middle — surface the candidate but do not commit (route to a human)
            return ResolutionResult(best_id, "review", best_score, review=True, matched_alias=best_alias)
        return ResolutionResult(None, "none", best_score)


# --- AgenticRAG: the plan-act-reflect loop (lazy, generator-driven) ----------

# A retriever is any callable mapping a query string to a ranked list of passage strings; a
# generator is the GroundedGenerator-style callable (system, prompt) -> text. Both are injected, so
# this module never imports an SDK and never assumes a concrete retrieval backend.
Retriever = Callable[[str], Sequence[str]]
Generator = Callable[[str, str], str]

_DECOMPOSE_SYSTEM = (
    "You are a query planner for multi-hop retrieval. Given a question, output the next single "
    "retrieval sub-query whose answer you most need before you can answer the original question. "
    "Output ONLY that sub-query, on one line. If the question needs no further retrieval, output the "
    "original question unchanged."
)
_REFLECT_SYSTEM = (
    "You decide whether the evidence gathered so far is sufficient to answer the question. "
    'Reply with exactly "ENOUGH" if it is, or "MORE" if another retrieval hop is needed. Output only '
    "one of those two words."
)
_SYNTH_SYSTEM = (
    "Answer the question using ONLY the gathered evidence passages. Cite passage numbers in square "
    "brackets. If the evidence is insufficient, say so plainly rather than guessing."
)
_SELF_ASK_SYSTEM = (
    "Decompose the question into explicit follow-up sub-questions (Self-Ask, Press et al. 2022). "
    "Output one follow-up question per line, in the order they must be answered. If no decomposition "
    "is needed, output the original question."
)


def _looks_compositional(query: str) -> bool:
    """Cheap stand-in for the Adaptive-RAG complexity classifier (Ch 10) — the gate, not the loop.

    A real system routes with a trained classifier; here we approximate "genuinely multi-hop" by
    cheap signals (multiple clauses, comparison/superlative/chaining cues, length). The point is the
    *gate exists*: most traffic is local fact lookup and must never enter the loop.
    """
    q = query.lower()
    cues = (
        " and ", " same ", " compared", " than ", " both ", " which of ", " each ",
        "headquarter", "largest", "most ", "least ", " versus ", " vs ",
    )
    if any(c in q for c in cues):
        return True
    return len(query.split()) > 18


@dataclass
class AgenticRAG:
    """Plan-act-reflect loop over an injected retriever + generator, capped by a hard turn budget.

    Each iteration: decompose the current state into the next retrieval query, *act* (retrieve),
    append to the evidence scratchpad, then *reflect* ("enough? / budget?") and either loop or
    synthesize (Ch 11). The loop is gated by a complexity check — simple queries take a single-shot
    path and never pay the multiplicative latency/token cost of looping.

    Inject ``retriever`` (query -> passages) and ``generator`` ((system, prompt) -> text); both are
    lazy, so importing this module pulls in no SDK. ``run`` returns a dict with the answer, the hop
    trace, and accumulated evidence so the loop stays inspectable.
    """

    retriever: Retriever
    generator: Generator
    max_hops: int = 5
    top_k: int = 5
    gate: Callable[[str], bool] = _looks_compositional

    @classmethod
    def default(cls, retriever: Retriever, generator: Generator, max_hops: int = 5) -> "AgenticRAG":
        return cls(retriever=retriever, generator=generator, max_hops=max_hops)

    def run(self, query: str) -> dict:
        """Decompose → retrieve → reflect (enough? budget?) → loop or synthesize (Ch 11)."""
        # the gate: simple queries skip the loop entirely (single-shot path)
        if not self.gate(query):
            passages = list(self.retriever(query))[: self.top_k]
            return {
                "answer": self._synthesize(query, passages),
                "evidence": passages,
                "hops": [{"subquery": query, "passages": passages}],
                "looped": False,
                "terminated": "single_shot",
            }

        evidence: list[str] = []
        trace: list[dict] = []
        state = query
        for hop in range(self.max_hops):
            subquery = self._decompose(query, state, evidence)
            retrieved = list(self.retriever(subquery))[: self.top_k]
            evidence.extend(retrieved)
            trace.append({"subquery": subquery, "passages": retrieved})
            # reflect: enough? (the budget cap is the for-loop bound itself)
            if self._reflect(query, evidence):
                terminated = "enough"
                break
            state = subquery
        else:
            terminated = "turn_budget"  # hit the hard cap without converging (non-termination guard)

        return {
            "answer": self._synthesize(query, evidence),
            "evidence": evidence,
            "hops": trace,
            "looped": True,
            "terminated": terminated,
        }

    # -- multi-hop helpers (lazy, generator-driven) ----------------------------
    def self_ask(self, query: str) -> dict:
        """Self-Ask (Press et al., arXiv:2210.03350): explicit, auditable follow-up sub-questions.

        Decompose the question into a legible list of follow-ups, retrieve + answer each in order
        (entity-normalizable per sub-question), then synthesize. The structure makes every hop a
        discrete, inspectable, individually-guardrailable unit — the chapter's default multi-hop pick.
        """
        subqs = self._self_ask_decompose(query)
        evidence: list[str] = []
        trace: list[dict] = []
        for subq in subqs[: self.max_hops]:
            retrieved = list(self.retriever(subq))[: self.top_k]
            evidence.extend(retrieved)
            trace.append({"subquery": subq, "passages": retrieved})
        return {
            "answer": self._synthesize(query, evidence),
            "evidence": evidence,
            "hops": trace,
            "method": "self_ask",
        }

    def ircot(self, query: str) -> dict:
        """IRCoT (Trivedi et al., arXiv:2212.10509): interleave retrieval with chain-of-thought.

        Generate one reasoning step, use it as the next retrieval query, append the documents, and
        repeat until reflection says enough or the step cap hits. Free-form reasoning steers
        retrieval where discrete decomposition is awkward — the flexible, harder-to-inspect sibling
        of Self-Ask. Mechanically this is :meth:`run`'s loop, exposed under its paper name.
        """
        out = self.run(query)
        out["method"] = "ircot"
        return out

    # -- internal generator calls (one place each, all lazy) -------------------
    def _decompose(self, original: str, state: str, evidence: Sequence[str]) -> str:
        ev = _format_passages(evidence)
        prompt = f"Original question: {original}\nCurrent sub-query: {state}\nEvidence so far:\n{ev}"
        out = self.generator(_DECOMPOSE_SYSTEM, prompt).strip()
        return out or original

    def _reflect(self, original: str, evidence: Sequence[str]) -> bool:
        prompt = f"Question: {original}\nEvidence:\n{_format_passages(evidence)}"
        out = self.generator(_REFLECT_SYSTEM, prompt).strip().upper()
        return out.startswith("ENOUGH")

    def _synthesize(self, query: str, evidence: Sequence[str]) -> str:
        prompt = f"Question: {query}\nEvidence:\n{_format_passages(evidence)}"
        return self.generator(_SYNTH_SYSTEM, prompt).strip()

    def _self_ask_decompose(self, query: str) -> list[str]:
        out = self.generator(_SELF_ASK_SYSTEM, query).strip()
        subqs = [line.strip(" -*\t") for line in out.splitlines() if line.strip()]
        return subqs or [query]


def _format_passages(passages: Iterable[str]) -> str:
    items = list(passages)
    if not items:
        return "(none yet)"
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(items))


# --- MCPSource: a uniform tool boundary over an external source --------------


@dataclass
class MCPSource:
    """A thin, uniform ``search(query) -> list`` boundary over one external source (Ch 11, MCP).

    The Model Context Protocol's value is a single client-host-server contract: each source is
    wrapped once as a server exposing a ``search`` *Tool*, and the agent sees one uniform,
    runtime-discoverable retrieval interface instead of N bespoke integrations. This wrapper models
    that boundary — *interface, not semantics* — without a hard MCP dependency:

    - ``search_fn``: an in-process callable (the common, dependency-free case — wrap any retriever).
    - ``server``/``tool``: name an MCP server + tool to call lazily over the ``mcp`` SDK instead.

    The chapter's caveats apply: two sources' ``search`` tools can mean subtly different things, and a
    uniform boundary is also a uniform attack surface (hardening is Ch 15). This wrapper standardizes
    the *call*, not the relevance behind it.
    """

    name: str
    search_fn: Retriever | None = None
    server: str | None = None
    tool: str = "search"
    extra: dict = field(default_factory=dict)

    def search(self, query: str, top_k: int = 5) -> list:
        """Uniform retrieval call. Uses the in-process ``search_fn`` if given, else a lazy MCP call."""
        if self.search_fn is not None:
            return list(self.search_fn(query))[:top_k]
        if self.server is not None:
            return self._mcp_search(query, top_k)
        raise ValueError(
            f"MCPSource '{self.name}' has neither a search_fn nor an MCP server configured."
        )

    def _mcp_search(self, query: str, top_k: int) -> list:
        # Lazy: only the MCP-backed path needs the SDK. Kept thin and uncalled in tests/import.
        from mcp.client.session import ClientSession  # type: ignore  # lazy, optional dep

        raise NotImplementedError(  # pragma: no cover
            "Connect an MCP ClientSession to "
            f"server '{self.server}' and invoke tool '{self.tool}'. "
            "Wire your transport (stdio / Streamable HTTP) here; ClientSession is "
            f"{ClientSession.__name__}."
        )


__all__ = ["EntityResolver", "ResolutionResult", "normalize", "AgenticRAG", "MCPSource"]
