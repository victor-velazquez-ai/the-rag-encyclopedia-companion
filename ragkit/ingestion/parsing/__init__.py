"""ragkit.ingestion.parsing — documents into faithful structured text (Book Ch 2).

The most error-prone stage in RAG: nothing downstream recovers what parsing drops, scrambles, or
invents. The chapter's fork is classical-OCR vs. pure-VLM vs. hybrid, and the verdict is a small
decision tree, not one model. The default is a *detector-first hybrid* (crop the page, then a VLM
transcribes each block) because cropping leashes the VLM's hallucination while keeping its quality.

    hybrid.py     detector-first hybrid: MinerU 2.5 default; PaddleOCR-VL for multilingual/throughput
    vlm.py        pure end-to-end VLM (dots.ocr) — one model, no cascade, for non-dense pages
    classical.py  zero-hallucination transcription (Tesseract/Textract) for legal/regulatory/finance
    tables.py     table extraction; serialize as HTML so rowspan/colspan survive (never Markdown)
    figures.py    chart/figure handling: typed blocks + image ref + a *hedged* surrogate, never a hard number

Two rules carry most of the chapter. Tables: serialize to HTML — Markdown cannot express a merged
cell, and the structure-vs-content gap (~97-99 S-TEDS vs ~91-93 end-to-end TEDS) is in-cell digit
error, so measure in-cell accuracy where numbers matter. Charts: assume a VLM-read number is wrong
(frontier models lose 40-60% numeric accuracy on complex charts, and fail *silently*) — verify it
against a source table or store it as an image plus a hedged surrogate, never as an asserted fact.

Phase-1 scaffold: the parsing entry point's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# def parse(path, *, backend: str = "hybrid", tables: str = "html") -> "ParsedDoc":
#     """Parse a document into structured, reading-order-correct text.
#
#     backend: "hybrid" (MinerU 2.5 — default) | "vlm" (dots.ocr) | "classical" (zero-hallucination)
#     tables:  "html" (default; preserves merged cells) | "json" (programmatic) | "markdown" (rectangular only)
#
#     Returns a ParsedDoc carrying text in reading order, tables as HTML, and figures as
#     (image_ref, textual_surrogate) pairs — chart numbers flagged untrusted until verified.
#     """
#     ...
#
# class ParsedDoc:
#     """Faithful structured output: reading-order text + HTML tables + linked figure surrogates."""
#     ...

__all__ = ["parse", "ParsedDoc"]  # populated in Phase 2
