"""ragkit.ingestion.parsing — documents into faithful structured text (Book Ch 2).

The most error-prone stage in RAG: nothing downstream recovers what parsing drops, scrambles, or
invents. The chapter's fork is classical-OCR vs. pure-VLM vs. hybrid, and the verdict is a small
decision tree, not one model. The default for scanned/complex pages is a *detector-first hybrid*
(crop the page, then a VLM transcribes each block) because cropping leashes the VLM's hallucination
while keeping its quality.

This module is the *text-extraction entry point* for the formats a bring-your-own-key stack can
parse with no GPU: plain text, Markdown, simple HTML (including tables), and born-digital PDFs (via
a lazy ``pypdf`` import). The chapter's frontier — detector-first hybrid VLM / OCR for scans and
dense layouts (MinerU 2.5, PaddleOCR-VL, dots.ocr) — is an *external service*, not a dependency we
bundle; ``register_parser`` is the extension point to wire one in for a given content type.

    parse(source, *, content_type=None) -> str
        Dispatch on content type (sniffed from the path/bytes if not given) and return
        reading-order text. HTML tables are serialized to Markdown by ``html_table_to_markdown``.

Two chapter rules shape this surface. *Tables:* HTML's ``rowspan``/``colspan`` faithfully encode
merged cells; Markdown cannot express a merged cell, so ``html_table_to_markdown`` is a lossy
convenience for *rectangular* tables only (it flattens spans by repeating the spanned value) — keep
the source HTML when a number's exact place matters. *Charts/figures:* a VLM-read chart number is
wrong 40-60% of the time and fails silently, so this layer never asserts one; figure handling lives
behind the external hybrid parser, not here.
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Callable

# Extension registry: content_type -> parser(source) -> str. The hybrid VLM/OCR frontier (Ch 2's
# detector-first default for scans) is an EXTERNAL service; register it here for, e.g., "image/png".
_PARSERS: dict[str, Callable[[object], str]] = {}


def register_parser(content_type: str, fn: Callable[[object], str]) -> Callable[[object], str]:
    """Register an external parser for a content type (the hybrid-VLM/OCR extension point, Ch 2)."""
    _PARSERS[content_type] = fn
    return fn


# --- content-type sniffing ---------------------------------------------------
def _sniff(source: object) -> str:
    """Best-effort content type from a path suffix or a bytes/str payload. Defaults to plain text."""
    if isinstance(source, bytes):
        head = source[:1024].lstrip()
        if source[:5] == b"%PDF-":
            return "application/pdf"
        if head[:1] == b"<" and (b"<table" in head.lower() or b"<html" in head.lower() or b"<body" in head.lower()):
            return "text/html"
        return "text/plain"
    text = str(source)
    low = text.lower()
    if low.endswith(".pdf"):
        return "application/pdf"
    if low.endswith((".html", ".htm")):
        return "text/html"
    if low.endswith((".md", ".markdown")):
        return "text/markdown"
    if low.endswith((".txt",)):
        return "text/plain"
    # inline payloads (not a path): detect HTML by a leading tag
    stripped = text.lstrip()
    if stripped[:1] == "<" and ("<table" in low or "<html" in low or "<body" in low or "<p" in low):
        return "text/html"
    return "text/plain"


def _read_path_or_text(source: object) -> str:
    """If ``source`` is an existing file path, read it; otherwise treat it as the content itself."""
    import os

    if isinstance(source, (bytes, bytearray)):
        return bytes(source).decode("utf-8", errors="replace")
    s = str(source)
    if len(s) < 4096 and os.path.exists(s):
        with open(s, encoding="utf-8", errors="replace") as f:
            return f.read()
    return s


# --- HTML table -> Markdown (pure, stdlib regex; no bs4 required) ------------
_TAG = re.compile(r"<[^>]+>")
_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.I | re.S)
_CELL = re.compile(r"<(t[hd])\b([^>]*)>(.*?)</\1>", re.I | re.S)
_SPAN = re.compile(r"colspan\s*=\s*[\"']?(\d+)", re.I)


def _cell_text(raw: str) -> str:
    """Strip inner tags and unescape entities, collapsing whitespace to a single Markdown-safe line."""
    text = _TAG.sub(" ", raw)
    text = _html.unescape(text)
    text = text.replace("|", r"\|")
    return re.sub(r"\s+", " ", text).strip()


def html_table_to_markdown(html: str) -> str:
    """Serialize a *simple, rectangular* HTML ``<table>`` to a Markdown pipe table (Ch 2, pure).

    The chapter is explicit that this is lossy: Markdown cannot express a merged cell, so a
    ``colspan``/``rowspan`` is *flattened* (a spanned value is repeated across the columns it covers)
    to keep the grid rectangular. Use this only for trivially rectangular tables; keep the source
    HTML whenever a merged header or an exact in-cell number must survive into retrieval.

    Pure stdlib (regex) so it needs no ``bs4`` — but if ``bs4`` is installed you may prefer it for
    gnarly real-world HTML; this function deliberately handles only the simple case.
    """
    rows: list[list[str]] = []
    for row_html in _ROW.findall(html):
        cells: list[str] = []
        for m in _CELL.finditer(row_html):
            attrs, body = m.group(2), m.group(3)
            text = _cell_text(body)
            span_match = _SPAN.search(attrs)
            span = int(span_match.group(1)) if span_match else 1
            cells.extend([text] * max(1, span))  # flatten colspan by repetition (lossy)
        if cells:
            rows.append(cells)
    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body_rows = rows[0], rows[1:]

    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in body_rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _inline_html_tables(html: str) -> str:
    """Replace each ``<table>...</table>`` with its Markdown rendering; strip remaining tags."""
    def _repl(m: "re.Match[str]") -> str:
        return "\n\n" + html_table_to_markdown(m.group(0)) + "\n\n"

    out = re.sub(r"<table\b[^>]*>.*?</table>", _repl, html, flags=re.I | re.S)
    # crude block-level breaks so stripped paragraphs don't run together
    out = re.sub(r"</(p|div|h[1-6]|li|tr|br\s*/?)>", "\n", out, flags=re.I)
    out = _TAG.sub("", out)
    out = _html.unescape(out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _parse_pdf(source: object) -> str:
    """Born-digital PDF text via a lazy ``pypdf`` import (Ch 2: scans need the external hybrid VLM)."""
    try:
        import pypdf  # lazy — needs `pip install pypdf`
    except ImportError as e:  # pragma: no cover - import-guarded message
        raise ImportError(
            "Parsing PDFs needs pypdf: `pip install pypdf`. Note (Ch 2): pypdf extracts text from "
            "BORN-DIGITAL PDFs only; scanned/image PDFs and dense layouts need the detector-first "
            "hybrid VLM/OCR parser (an external service) — register it via register_parser()."
        ) from e

    import io
    import os

    if isinstance(source, (bytes, bytearray)):
        stream: object = io.BytesIO(bytes(source))
    elif isinstance(source, str) and os.path.exists(source):
        stream = source
    else:
        raise ValueError("PDF parsing needs a file path or raw bytes, not inline text.")

    reader = pypdf.PdfReader(stream)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()


def parse(source: object, *, content_type: str | None = None) -> str:
    """Parse ``source`` into faithful, reading-order text (Book Ch 2 entry point).

    ``source`` is a file path, raw bytes, or inline content. ``content_type`` (e.g. ``"text/html"``,
    ``"application/pdf"``) is sniffed from the suffix/payload when omitted. Returns plain text with
    HTML tables rendered as Markdown.

    Handled here (no GPU, BYO-key stack): plain text and Markdown (returned as-is, only outer
    whitespace stripped — Markdown *is* the target format and must not be mangled), simple HTML
    (tables → Markdown, other tags stripped), and born-digital PDF (lazy ``pypdf``). The frontier —
    hybrid VLM/OCR for scans, dense layouts, and figures — is an external service; attach one with
    ``register_parser(content_type, fn)``.
    """
    ctype = content_type or _sniff(source)

    if ctype in _PARSERS:  # external/registered parser wins (the hybrid-VLM extension point)
        return _PARSERS[ctype](source)

    if ctype == "application/pdf":
        return _parse_pdf(source)

    text = _read_path_or_text(source)

    if ctype == "text/html":
        return _inline_html_tables(text)

    # text/plain and text/markdown: already the target format — return cleaned, do not mangle.
    return text.strip()


__all__ = ["parse", "html_table_to_markdown", "register_parser"]
