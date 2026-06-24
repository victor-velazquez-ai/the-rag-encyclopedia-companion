# %% [markdown]
# # Chapter 2 - Parsing documents into faithful, structured text
#
# Parsing is the most error-prone stage in RAG: nothing downstream recovers what the parser drops,
# scrambles, or invents. This walkthrough runs the *offline* part of `ragkit` - plain text, Markdown,
# and simple-HTML-table parsing with no GPU and no key - and points at where the frontier (hybrid
# VLM / OCR for scans) lives. The book's rule throughout: **preserve structure**, and **never trust
# an unverified VLM chart number.**
#
# Production code: [`ragkit/ingestion/parsing`](../../ragkit/ingestion/parsing/__init__.py).
# Book section: Ch 2, "The three architectures" and "Table extraction and serialization".

# %%
# `parse` and `html_table_to_markdown` are the offline entry points (no GPU, no key).
from ragkit.ingestion.parsing import html_table_to_markdown, parse

# %% [markdown]
# ## Plain text and Markdown pass through cleanly
# Markdown *is* the target format, so `parse` returns it unmangled (only outer whitespace stripped).
# The content type is sniffed from the payload when you don't pass one. This is the cheap, lossless
# path: born-digital text needs no model at all.

# %%
plain = "Retrieval-augmented generation grounds a model in retrieved text.\n\nParsing comes first."
md = "# Methods\n\nWe evaluate **recall@k** and report `nDCG@10` per query."

print("--- text/plain ---")
print(parse(plain))
print("\n--- text/markdown (structure preserved) ---")
print(parse(md, content_type="text/markdown"))

# %% [markdown]
# ## HTML tables serialize to Markdown - structure survives
# A table is a fact grid; flattening it into a wall of prose severs every cell from its row and
# column. `html_table_to_markdown` renders a *rectangular* HTML table to a Markdown pipe table so the
# grid survives into chunking and retrieval. This is Ch 2's "preserve structure" point made concrete.

# %%
TABLE_HTML = """
<table>
  <tr><th>Model</th><th>MTEB</th><th>License</th></tr>
  <tr><td>Qwen3-Embedding-8B</td><td>70.6</td><td>Apache-2.0</td></tr>
  <tr><td>NV-Embed-v2</td><td>72.3</td><td>non-commercial</td></tr>
</table>
"""

print(html_table_to_markdown(TABLE_HTML))

# %% [markdown]
# ## Tables inside a page are inlined automatically
# `parse` on an HTML document replaces each `<table>` with its Markdown rendering and strips the
# remaining tags, so a mixed prose + table page comes out as clean reading-order text with the grid
# intact. Note this is *sniffed* as HTML from the leading tag - no `content_type` needed.

# %%
PAGE_HTML = """
<html><body>
<p>The 2026 embedding shortlist, scored on our own held-out queries:</p>
<table>
  <tr><th>Provider</th><th>Dims</th></tr>
  <tr><td>OpenAI text-embedding-3-large</td><td>3072</td></tr>
  <tr><td>Voyage voyage-3-large</td><td>1024</td></tr>
</table>
<p>Read the license before the leaderboard score.</p>
</body></html>
"""

print(parse(PAGE_HTML))

# %% [markdown]
# ## The lossy edge: merged cells
# Markdown cannot express a merged cell, so a `colspan`/`rowspan` is *flattened* - the spanned value
# is repeated across the columns it covers. That keeps the grid rectangular but is lossy. The chapter's
# rule: when a merged header or an exact in-cell number must survive, **keep the source HTML**.

# %%
MERGED = """
<table>
  <tr><th colspan="2">Q4 Revenue</th></tr>
  <tr><td>Region</td><td>USD (M)</td></tr>
  <tr><td>NA</td><td>412</td></tr>
</table>
"""

print(html_table_to_markdown(MERGED))
print("\nNote: 'Q4 Revenue' is repeated across 2 columns (the colspan was flattened).")

# %% [markdown]
# ## The frontier is an external service - and VLM chart numbers are untrusted
# Scanned PDFs and dense layouts need the chapter's *detector-first hybrid* (crop the page, then a VLM
# transcribes each block - MinerU 2.5, PaddleOCR-VL, dots.ocr). That is an **external service**, not a
# dependency `ragkit` bundles; you wire it in with `register_parser(content_type, fn)`. And a VLM-read
# **chart number is wrong 40-60% of the time and fails silently**, so this layer never asserts one.

# %%
from ragkit.ingestion.parsing import register_parser

# Sketch only (not called): this is how you'd attach an external hybrid VLM/OCR parser for scans.
def _hybrid_vlm_parse(source):  # pragma: no cover - external service in production
    raise NotImplementedError("wire in MinerU 2.5 / PaddleOCR-VL here (external GPU service)")

register_parser("image/png", _hybrid_vlm_parse)
print("registered an external parser slot for image/png (scans -> hybrid VLM/OCR).")
print("RULE: a VLM chart number must be VERIFIED before it becomes a fact - never assert it blind.")

# %%
print("\nparsing done. Faithful structured text is what Ch 3 (chunking) splits next.")
