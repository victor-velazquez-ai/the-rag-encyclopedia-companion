"""professional_rag_kit.production.observability — tracing, drift detection, feedback loop (Book Ch 16).

You cannot operate what you cannot see. An un-traced RAG pipeline is unoperable, and retrofitting
tracing after an incident is the expensive path — so instrument from day one. The trace is the
substrate every other lever in the chapter measures against.

    psi / psi_alert    Population Stability Index for drift on a *scalar* signal, with the standard
                       0.1 / 0.2 thresholds. Pure. NEVER apply univariate tests (PSI per-dimension,
                       KS) to embeddings — oversensitive at scale, blind to joint structure; for
                       embedding drift use centroid-cosine / MMD / a classifier-based detector.
    trace              a lazy context manager that emits an OpenTelemetry / OpenInference span if the
                       SDK is installed, else a silent no-op. Lazy import; never a required dependency,
                       so this module imports with no tracing SDK present.
    feedback_to_golden shape thumbs-down traces for promotion into the eval golden set — the
                       production loop as plumbing (thumbs -> annotation queue -> eval dataset), so
                       every real-world failure becomes a permanent regression gate on the next deploy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator


# --- Drift: Population Stability Index (pure) ---------------------------------
def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    """Population Stability Index between a reference (``expected``) and current (``actual``) sample.

    PSI = sum_b (a_b - e_b) * ln(a_b / e_b), over ``bins`` quantile buckets cut on the *expected*
    distribution, where e_b / a_b are the fraction of each sample falling in bucket b. ~0 means no
    shift; it grows as the distributions diverge.

    Standard operating thresholds (Ch 16): *under 0.1 = no significant shift; 0.1-0.2 = moderate shift
    (investigate); over 0.2 = significant shift (act)* — see ``psi_alert``. PSI is the right tool for a
    *single scalar* feature only. Do NOT apply it per-dimension (or a univariate KS test) to
    embeddings: that is oversensitive at large N and blind to the joint structure that matters — use
    centroid-cosine drift, MMD, or a classifier-based detector for high-dimensional embedding drift.
    """
    exp = sorted(float(x) for x in expected)
    act = [float(x) for x in actual]
    n_exp, n_act = len(exp), len(act)
    if n_exp == 0 or n_act == 0 or bins < 1:
        return 0.0

    # Quantile bucket edges from the expected distribution (interior edges only).
    edges: list[float] = []
    for i in range(1, bins):
        pos = i * n_exp / bins
        lo = int(pos)
        frac = pos - lo
        if lo >= n_exp - 1:
            edges.append(exp[-1])
        else:
            edges.append(exp[lo] * (1 - frac) + exp[lo + 1] * frac)

    def _bucket(x: float) -> int:
        # rightmost bucket whose left edge x exceeds; edges are non-decreasing
        b = 0
        for e in edges:
            if x > e:
                b += 1
            else:
                break
        return b

    n_buckets = len(edges) + 1
    exp_counts = [0] * n_buckets
    act_counts = [0] * n_buckets
    for x in exp:
        exp_counts[_bucket(x)] += 1
    for x in act:
        act_counts[_bucket(x)] += 1

    # Laplace-smoothed proportions so empty buckets never blow up the log.
    eps = 1e-6
    total = 0.0
    for ec, ac in zip(exp_counts, act_counts):
        e_frac = max(ec / n_exp, eps)
        a_frac = max(ac / n_act, eps)
        total += (a_frac - e_frac) * math.log(a_frac / e_frac)
    return total


def psi_alert(value: float) -> str:
    """Map a PSI value to an operating band: ``"none"`` / ``"moderate"`` / ``"significant"``.

    Thresholds (Ch 16): < 0.1 none, 0.1-0.2 moderate (investigate), > 0.2 significant (act).
    """
    if value < 0.1:
        return "none"
    if value <= 0.2:
        return "moderate"
    return "significant"


# --- Tracing: lazy OTel / OpenInference span, else no-op ---------------------
@contextmanager
def trace(name: str, **attributes: Any) -> Iterator[Any]:
    """Open one tracing span named ``name`` if an OpenTelemetry SDK is installed, else a no-op (Ch 16).

    Lazy import: ``opentelemetry`` is never a required dependency, so this module imports fine without
    it and the context manager becomes a silent no-op (yields ``None``). When the SDK *is* present this
    emits a span carrying the given attributes — set ``gen_ai.*`` (OTel GenAI conventions) or
    OpenInference span-kind attributes here so a RETRIEVER -> RERANKER -> LLM trace tells you whether a
    bad answer was a retrieval or a generation failure. OTel's GenAI conventions are pre-release; pin a
    version and expect attribute churn.
    """
    try:
        from opentelemetry import trace as otel_trace  # lazy, optional
    except Exception:
        yield None
        return

    tracer = otel_trace.get_tracer("professional_rag_kit")
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            try:
                span.set_attribute(key, value)
            except Exception:
                pass
        yield span


# --- Feedback loop: thumbs-down -> golden-set candidates (pure) --------------
def feedback_to_golden(thumbs_down_items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Shape flagged (thumbs-down) traces into eval golden-set candidates (Ch 16: the feedback loop).

    The production loop is plumbing: thumbs-down -> annotation queue -> eval dataset, so every
    real-world failure becomes a permanent regression gate on the next deploy. This light, pure helper
    is the *promotion* step: it normalizes each flagged item into the same ``{query, relevant, gains,
    note}`` shape ``professional_rag_kit.eval.GoldenSet.from_items`` consumes, ready for a human in the annotation
    queue to confirm the ``relevant`` labels.

    Each input item is a mapping with at least a ``query``; optional ``relevant`` (the doc ids that
    *should* have been retrieved), ``gains`` (graded relevance), and a free-text ``note`` (what went
    wrong: bad retrieval / hallucination / stale answer). Items without a query are skipped.
    """
    candidates: list[dict[str, Any]] = []
    for item in thumbs_down_items:
        query = item.get("query")
        if not query:
            continue
        candidate: dict[str, Any] = {
            "query": query,
            "relevant": list(item.get("relevant", []) or []),
        }
        if item.get("gains"):
            candidate["gains"] = dict(item["gains"])
        note = item.get("note") or item.get("reason")
        if note:
            candidate["note"] = note
        candidates.append(candidate)
    return candidates


__all__ = ["psi", "psi_alert", "trace", "feedback_to_golden"]
