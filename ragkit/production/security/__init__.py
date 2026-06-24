"""ragkit.production.security — access control, PII, injection defense (Book Ch 15).

The properties a RAG system needs to survive an enterprise security review — none of which show up
in an offline metric, all of which show up in an incident report. The committed posture is defense
in depth: no single control is load-bearing, least of all "a smarter model" (BIPIA found more-capable
models are often *more* vulnerable to injection). These are pure, dependency-free guards meant to
*compose*, not to stand alone.

What lands here, faithful to the chapter's three implementable controls:

    PIIRedactor       two-sided redaction (OWASP LLM02:2025): redact *before indexing* AND
                      *re-redact retrieved chunks* at query time — the retrieved side is the half
                      teams forget and the half that leaks (retrieved docs usually carry more PII
                      than the query). Regex/checksum-free patterns for the common identifiers;
                      Microsoft Presidio (NER + checksum recognizers + anonymizers) is the
                      production upgrade and stays an optional/lazy dependency.
    InjectionDetector pattern-based *indirect* prompt-injection screen (OWASP LLM01:2025) — one
                      inline-detector layer of defense in depth, NOT a silver bullet. Static patterns
                      catch opportunistic payloads; an adaptive attacker optimizing against the
                      detector erodes it (the StruQ/SecAlign and adaptive-attack literature), so this
                      is one wall among several (permission-aware retrieval, data/instruction
                      separation, least-privilege tools), never the load-bearing one.
    enforce_acl       app-layer retrieval-time access trimming: keep only chunks whose stamped
                      ``allowed_groups`` intersect the caller's groups (OWASP LLM08:2025 — the leak
                      happens the instant an unentitled chunk enters the prompt). The vector store
                      enforces this server-side at query time; this is the belt-and-suspenders guard
                      in the app, applied *before* the model ever sees the chunk.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# --- PII redaction (pure) ----------------------------------------------------
# Order matters: more specific / structured patterns (SSN, credit card) run before the looser
# phone pattern so a card number is not partially eaten as a phone number. Each (label, regex).
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # SSN: 3-2-4 grouped by - or space (avoid matching inside a longer digit run).
    ("SSN", re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b")),
    # Credit-card-like: 13-16 digits, optionally grouped in 4s by - or space.
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    # Phone: optional +country, separators, 7-15 digits total in a plausible shape.
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{2,4}\b")),
    ("IPV4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
]


class PIIRedactor:
    """Pure regex redaction of common PII (Ch 15: two-sided redaction).

    Covers emails, phone numbers, US SSNs, credit-card-like numbers, and IPv4 addresses by replacing
    each match with a ``[REDACTED:<TYPE>]`` token. ``redact`` is deliberately the *same call* used on
    both sides of the pipeline — run it before indexing AND again on retrieved chunks at query time,
    since retrieved chunks usually carry more PII than the query and the retrieved side is the one
    teams forget (OWASP LLM02:2025).

    This is a coverage floor, not a compliance guarantee: regex misses unstructured PII (names,
    addresses, health details). The production upgrade is *Microsoft Presidio* — NER + checksum
    recognizers + configurable anonymizers — which stays an optional/lazy dependency (``use_presidio``
    is a documented hook, not wired here to keep the module import-light).
    """

    def __init__(self, *, patterns: Sequence[tuple[str, re.Pattern[str]]] | None = None) -> None:
        self.patterns = list(patterns) if patterns is not None else list(_PII_PATTERNS)

    def redact(self, text: str) -> tuple[str, list[str]]:
        """Return ``(clean_text, found_types)``. ``found_types`` lists each PII *kind* detected.

        Use this on BOTH sides: before indexing a document and again on every retrieved chunk before
        it reaches the model or your logs/traces.
        """
        found: list[str] = []
        clean = text
        for label, pattern in self.patterns:
            if pattern.search(clean):
                found.append(label)
                clean = pattern.sub(f"[REDACTED:{label}]", clean)
        return clean, found

    def redact_chunks(self, chunks: Iterable[str]) -> list[str]:
        """Convenience: re-redact a list of retrieved chunks (the query-time, retrieved side)."""
        return [self.redact(c)[0] for c in chunks]


# --- Indirect prompt-injection detection (pure) ------------------------------
# Pattern-based screen for the indirect-injection payloads retrieval uniquely introduces. Each entry
# is (weight, regex); risk is a saturating sum so several weak hits or one strong hit raise the flag.
_INJECTION_PATTERNS: list[tuple[float, re.Pattern[str]]] = [
    (0.6, re.compile(r"\bignore\s+(?:all\s+|the\s+)?(?:previous|prior|above|earlier)\b", re.I)),
    (0.6, re.compile(r"\bdisregard\s+(?:all\s+|the\s+|your\s+|any\s+)?\w*\s*instruction", re.I)),
    (0.5, re.compile(r"\b(?:forget|override)\s+(?:all\s+|the\s+|your\s+|previous\s+)", re.I)),
    (0.5, re.compile(r"\bsystem\s+prompt\b", re.I)),
    (0.5, re.compile(r"\byou\s+are\s+now\b", re.I)),
    (0.4, re.compile(r"\b(?:act|behave)\s+as\b", re.I)),
    (0.4, re.compile(r"\bnew\s+instructions?\b", re.I)),
    (0.4, re.compile(r"\b(?:reveal|print|repeat|leak|exfiltrate)\s+(?:the\s+|your\s+)?(?:system\s+prompt|prompt|instructions|secret|api[\s_-]?key)", re.I)),
    (0.3, re.compile(r"</?(?:system|assistant|user|instruction)\b", re.I)),  # role-override markers
    (0.3, re.compile(r"\bdo\s+anything\s+now\b|\bDAN\b")),
]


class InjectionDetector:
    """Pattern-based indirect-prompt-injection screen (Ch 15: defense in depth, OWASP LLM01:2025).

    ``scan`` returns ``(risk, matched)`` where ``risk`` is a saturating score in ``[0, 1]`` and
    ``matched`` lists the literal substrings that tripped patterns. Use it as an *inline detector* on
    retrieved chunks and on model output — one layer in a stack that also includes permission-aware
    retrieval, data/instruction separation (spotlighting), an alignment-hardened model (the
    StruQ/SecAlign line), and least-privilege tools.

    This is explicitly NOT a silver bullet: static patterns catch opportunistic payloads, but an
    adaptive attacker who optimizes against the detector erodes it, and more-capable models are often
    *more* vulnerable to injection (BIPIA). The durable defense is the layered one; treat a low risk
    score as "no *known-pattern* injection," never as "safe."
    """

    def __init__(self, *, threshold: float = 0.5, patterns: Sequence[tuple[float, re.Pattern[str]]] | None = None) -> None:
        self.threshold = threshold
        self.patterns = list(patterns) if patterns is not None else list(_INJECTION_PATTERNS)

    def scan(self, text: str) -> tuple[float, list[str]]:
        """Return ``(risk in [0,1], matched_substrings)``. Risk saturates so it never exceeds 1.0."""
        score = 0.0
        matched: list[str] = []
        for weight, pattern in self.patterns:
            m = pattern.search(text)
            if m:
                score += weight
                matched.append(m.group(0))
        return min(1.0, score), matched

    def is_suspicious(self, text: str) -> bool:
        """True if the scanned risk meets the configured threshold."""
        return self.scan(text)[0] >= self.threshold


# --- App-layer ACL trimming (pure) -------------------------------------------
def _groups_of(chunk: Any) -> set[str]:
    """Extract a chunk's ``allowed_groups`` whether it is a mapping or an attributed object."""
    if isinstance(chunk, Mapping):
        groups = chunk.get("allowed_groups", ())
    else:
        groups = getattr(chunk, "allowed_groups", ())
    return set(groups or ())


def enforce_acl(chunks: Sequence[Any], allowed_groups: Iterable[str]) -> list[Any]:
    """Keep only chunks whose stamped ``allowed_groups`` intersect the caller's (Ch 15, OWASP LLM08).

    Each chunk is a mapping or object carrying an ``allowed_groups`` collection stamped at index time;
    ``allowed_groups`` is the caller's resolved group/role set. A chunk survives iff the two sets
    overlap. A chunk with no stamped groups fails closed (it is dropped) — an unlabeled chunk is not
    world-readable by default.

    The vector store enforces this server-side at query time (the non-optional, primary control); this
    is the belt-and-suspenders app-layer guard applied *before* the model sees any chunk, because once
    an unentitled chunk is in the prompt the leak has already happened — filtering citations after
    generation is theater.
    """
    caller = set(allowed_groups)
    if not caller:
        return []
    return [c for c in chunks if _groups_of(c) & caller]


__all__ = ["PIIRedactor", "InjectionDetector", "enforce_acl"]
