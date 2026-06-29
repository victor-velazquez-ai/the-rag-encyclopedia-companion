"""professional_rag_kit.architectures.adaptive — composable runtime control policies (Book Ch 10).

Flat RAG retrieves the same way every time: always, once, and trusting whatever came back. The
policies here are guards over *different decision points* on one pipeline, so they stack (Ch 10).
Each ships as a **pattern** — a control graph over the frontier model you already run — not as the
papers' 7B–13B checkpoints: the policy is the durable asset, the model is yours.

``AdaptiveRAG`` wires the four chapter policies over a ``retriever`` and a ``generator`` you supply:

- ``route(query) -> "no_retrieval" | "single" | "multi"`` — Adaptive-RAG (arXiv:2403.14403)
  complexity routing, the cost optimizer. A **PURE** heuristic (tested); swap in an LLM/classifier
  router by overriding ``route`` or passing ``router=``.
- ``crag_grade(query, passages) -> "correct" | "ambiguous" | "incorrect"`` — CRAG (arXiv:2401.15884):
  grade the retrieval (lazy LLM), with a web-search fallback *hook* for the weak grades.
- ``self_rag(query, ...)`` — Self-RAG (arXiv:2310.11511) retrieve-then-support-critique reflection.
- ``flare(query, ...)`` — FLARE (arXiv:2305.06983) generator-driven, retrieve on low-confidence spans.

The grading/reflection calls are honest **patterns** over your frontier model (the chapter's
load-bearing caveat), routed lazily through ``ProviderRegistry`` — no vendor SDK at import time.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from professional_rag_kit.core.config import ProviderRegistry, load_config

# Ch 10 (Adaptive-RAG): route by query complexity to no-retrieval / single-step / multi-step.
# Pure heuristic boundary — the cheap, dependency-free default the chapter explicitly allows
# ("the classifier can be the paper's small T5, a cheaper LLM call ... or a heuristic").

# Markers a knowledgeable model answers from parametric memory — branch A (no retrieval).
_NO_RETRIEVAL = re.compile(
    r"\b(what is|what are|define|definition of|who wrote|capital of|how many|"
    r"convert|translate|spell|meaning of)\b",
    re.IGNORECASE,
)
# Markers of a compositional / multi-hop question — branch C (multi-step).
_MULTI_HOP = re.compile(
    r"\b(compare|difference between|relationship between|how does .* (affect|relate|impact)|"
    r"both|and then|after that|step by step|which .* and .*|"
    r"trace|chain|connected to|exposed to|across (all|these|the))\b",
    re.IGNORECASE,
)

CRAG_GRADE_SYSTEM = (
    "You are a retrieval-quality grader. Given a query and the retrieved passages, judge whether the "
    "passages are sufficient to answer the query. Reply with exactly one word:\n"
    "  correct    — the passages clearly answer the query\n"
    "  ambiguous  — partial or uncertain support\n"
    "  incorrect  — the passages do not answer the query\n"
    "Output only that one word."
)

SUPPORT_SYSTEM = (
    "You critique whether a drafted answer is fully supported by the provided passages (Self-RAG "
    "[IsSup]). Reply with one word: 'supported' if every claim traces to the passages, otherwise "
    "'unsupported'. Output only that word."
)


@dataclass
class AdaptiveRAG:
    """Composable control layer over a retriever + generator (Book Ch 10).

    ``retriever`` is a callable ``query -> list[str]`` (passages); ``generator`` is a callable
    ``(query, passages) -> str``. Both are optional so the *pure* ``route`` heuristic and the
    LLM-graders can be used standalone. ``run`` dispatches on the route.
    """

    retriever: Optional[Callable[[str], list]] = None
    generator: Optional[Callable[[str, list], str]] = None
    # optional router override: callable query -> "no_retrieval"|"single"|"multi"
    router: Optional[Callable[[str], str]] = None
    provider: str = "anthropic"
    model: str = ""
    # CRAG fallback hook: callable query -> list[str]
    web_search: Optional[Callable[[str], list]] = None
    flare_threshold: float = 0.0  # placeholder confidence gate; documented hook

    # --- Adaptive-RAG routing (PURE) ------------------------------------------
    def route(self, query: str) -> str:
        """Route by complexity → ``"no_retrieval" | "single" | "multi"`` — PURE heuristic (Ch 10).

        Order matters: multi-hop cues win over no-retrieval cues (a compositional question always
        needs the heavy path), and the default for a plain factoid is single-step retrieval.
        An explicit ``router`` override (LLM/classifier) takes precedence if supplied.
        """
        if self.router is not None:
            return self.router(query)
        q = query.strip()
        if _MULTI_HOP.search(q):
            return "multi"
        if _NO_RETRIEVAL.search(q):
            return "no_retrieval"
        return "single"

    def run(self, query: str, *, max_steps: int = 3) -> dict:
        """Route, then retrieve+generate accordingly (Ch 10). Returns a small trace dict.

        - ``no_retrieval`` → answer from the generator's parametric knowledge (no passages).
        - ``single`` → one retrieve-then-generate pass.
        - ``multi`` → iterative multi-step retrieval, accumulating passages over ``max_steps``.
        """
        decision = self.route(query)
        if decision == "no_retrieval":
            answer = self._generate(query, [])
            return {"route": decision, "passages": [], "answer": answer}
        if decision == "single":
            passages = list(self._retrieve(query))
            return {"route": decision, "passages": passages, "answer": self._generate(query, passages)}
        # multi-step: naive iterative accumulation (the multi-hop path)
        passages: list[str] = []
        for _ in range(max(1, max_steps)):
            new = [p for p in self._retrieve(query) if p not in passages]
            if not new:
                break
            passages.extend(new)
        return {"route": decision, "passages": passages, "answer": self._generate(query, passages)}

    # --- CRAG: grade the retrieval, then correct it (lazy LLM + web hook) ------
    def crag_grade(self, query: str, passages: list[str], *, max_tokens: int = 8) -> str:
        """Grade retrieval → ``"correct" | "ambiguous" | "incorrect"`` (CRAG, Ch 10), lazy LLM."""
        if not passages:
            return "incorrect"
        backend = ProviderRegistry.get("generation", self.provider)
        listing = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        prompt = f"Query: {query}\n\nPassages:\n{listing}"
        out = backend(self._model(), CRAG_GRADE_SYSTEM, prompt, max_tokens).strip().lower()
        for label in ("incorrect", "ambiguous", "correct"):
            if label in out:
                return label
        return "ambiguous"

    def crag(self, query: str, *, max_tokens: int = 2048) -> dict:
        """CRAG control flow: retrieve → grade → correct (refine / web-fallback / both), then generate.

        On ``incorrect``/``ambiguous`` the ``web_search`` hook (if wired) augments or replaces the
        internal passages — the chapter's "fall back to the web" branch. Returns a trace dict.
        """
        passages = list(self._retrieve(query))
        grade = self.crag_grade(query, passages)
        if grade != "correct" and self.web_search is not None:
            web = list(self.web_search(query))
            passages = web if grade == "incorrect" else passages + web
        return {"grade": grade, "passages": passages, "answer": self._generate(query, passages)}

    # --- Self-RAG: retrieve + segment-level support critique (lazy) -----------
    def self_rag(self, query: str, *, max_redrafts: int = 1, max_tokens: int = 2048) -> dict:
        """Self-RAG pattern: draft, critique support ([IsSup]), re-draft if unsupported (Ch 10), lazy.

        A pattern over your frontier model, not the paper's 7B checkpoint. Returns the (re)drafted
        answer plus the final support verdict.
        """
        passages = list(self._retrieve(query))
        answer = self._generate(query, passages)
        verdict = self._support_verdict(answer, passages)
        redrafts = 0
        while verdict == "unsupported" and redrafts < max_redrafts:
            answer = self._generate(query, passages)
            verdict = self._support_verdict(answer, passages)
            redrafts += 1
        return {"passages": passages, "answer": answer, "support": verdict, "redrafts": redrafts}

    def _support_verdict(self, answer: str, passages: list[str], *, max_tokens: int = 8) -> str:
        if not passages:
            return "unsupported"
        backend = ProviderRegistry.get("generation", self.provider)
        listing = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        prompt = f"Passages:\n{listing}\n\nDraft answer: {answer}"
        out = backend(self._model(), SUPPORT_SYSTEM, prompt, max_tokens).strip().lower()
        return "supported" if "support" in out and "unsupport" not in out else "unsupported"

    # --- FLARE: forward-looking active retrieval (generator-driven, lazy) -----
    def flare(self, query: str, *, max_sentences: int = 8, max_tokens: int = 256) -> dict:
        """FLARE pattern: generate sentence-by-sentence, retrieve on low-confidence spans (Ch 10).

        The generator drives retrieval: each drafted sentence whose ``_low_confidence`` hook fires
        triggers a retrieval keyed on that sentence, then a regeneration with the new context. The
        confidence gate is a documented hook (real FLARE needs token logprobs vs threshold θ).
        """
        context: list[str] = list(self._retrieve(query))
        sentences: list[str] = []
        for _ in range(max(1, max_sentences)):
            draft = self._generate_next(query, context, sentences, max_tokens)
            if not draft:
                break
            if self._low_confidence(draft):
                context += [p for p in self._retrieve(draft) if p not in context]
                draft = self._generate_next(query, context, sentences, max_tokens)
            sentences.append(draft)
        return {"passages": context, "answer": " ".join(sentences).strip()}

    def _low_confidence(self, sentence: str) -> bool:
        """Confidence gate hook for FLARE. Default: never fires (override for logprob-based θ)."""
        return False

    def _generate_next(self, query, context, sentences, max_tokens):
        prefix = " ".join(sentences)
        prompt = f"{query}\n\nAnswer so far: {prefix}" if prefix else query
        return self._generate(prompt, context, max_tokens=max_tokens).strip()

    # --- generator / retriever plumbing ---------------------------------------
    def _retrieve(self, query: str):
        if self.retriever is None:
            return []
        return self.retriever(query)

    def _generate(self, query: str, passages, *, max_tokens: int = 2048) -> str:
        if self.generator is not None:
            return self.generator(query, passages)
        # default generator: route through the grounded generation backend, lazily
        backend = ProviderRegistry.get("generation", self.provider)
        ctx = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        system = "Answer the question using the context passages if present; otherwise answer directly."
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}" if passages else query
        return backend(self._model(), system, prompt, max_tokens).strip()

    def _model(self) -> str:
        return self.model or load_config().generation.model


__all__ = ["AdaptiveRAG", "CRAG_GRADE_SYSTEM", "SUPPORT_SYSTEM"]
