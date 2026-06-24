# %% [markdown]
# # Chapter 12 — Visual document retrieval (ColPali): embed the page, skip OCR
#
# One decision, applied to visually-structured documents: do you retrieve over a *lossy text
# projection* (OCR-then-embed) or over the **page image itself**? On layout-heavy pages - forms,
# tables, charts, slides - the answer is decisively the image. ColPali/ColQwen render each page to an
# image, emit *one vector per patch* from a vision-language model, and score queries with ColBERT-style
# **MaxSim late interaction**. That deletes the entire brittle parse-and-chunk front end of Chapters
# 2-3.
#
# This walkthrough demonstrates the `VisualRetriever` API. It needs the optional **`[multimodal]`**
# extra (colpali-engine + a GPU), so the code is *guarded*: if the extra is absent we print the
# install hint and stop before any model load. Read it as an API tour, not a laptop demo.
#
# Production code: [`ragkit.architectures.multimodal`](../../ragkit/architectures/multimodal/__init__.py)
# (`VisualRetriever`). Book section 12: "Visual document retrieval".

# %%
import importlib.util

from ragkit.architectures.multimodal import VisualRetriever

# Construction is free and lazy - no model touches until you index/search.
retriever = VisualRetriever.default()  # ColQwen2.5 + token pooling, the chapter's committed default
print("backbone:      ", retriever.backbone)
print("token_pooling: ", retriever.token_pooling, "(applied from day one: -66.7% vectors, -2.2% quality)")

# %% [markdown]
# ## The extra guard
# Page encoding is a VLM forward pass, so it needs `colpali-engine`. We check for it *before* calling
# any model method. If it is missing, print the one-line install command and stop here - everything
# below this cell assumes the extra is present.

# %%
HAVE_MULTIMODAL = importlib.util.find_spec("colpali_engine") is not None

if not HAVE_MULTIMODAL:
    print("[multimodal] extra not installed. To run the cells below:")
    print("    pip install -e '.[multimodal]'")
    print("and run on a GPU-backed host (page encoding is a VLM forward pass).")
    print("\nStopping here - the API tour below is illustrative and does not execute.")
else:
    print("colpali-engine present: the cells below will load a VLM and run for real.")

# %% [markdown]
# ## Indexing pages - the page is the unit
# No OCR, no layout detection, no chunker. `index` renders each page to an image, encodes it to a
# per-patch multi-vector (with token pooling), and stores it. The input is PIL images of rendered
# pages - you would get them from `pdf2image` or similar.
#
# ```python
# from pdf2image import convert_from_path
# pages = convert_from_path("annual_report.pdf", dpi=150)   # list of PIL images
# retriever.index(pages)                                    # embed + store, no text path anywhere
# ```

# %%
if HAVE_MULTIMODAL:
    from pdf2image import convert_from_path  # type: ignore

    pages = convert_from_path("annual_report.pdf", dpi=150)
    retriever.index(pages)
    print(f"indexed {len(pages)} pages as per-patch multi-vectors")
else:
    print("(skipped: needs the [multimodal] extra + a PDF + a GPU)")

# %% [markdown]
# ## Searching - MaxSim late interaction
# The query encodes to *per-token* vectors; each query token is matched to its best page patch and the
# scores summed (MaxSim). This is the exact late-interaction operation dissected in Chapter 4 - the
# multi-vector / MaxSim scoring path, applied to vision. `search` returns `(page_image, score)`
# best-first, page images you hand straight to a *multimodal* generator (text-only generation has not
# escaped extraction, only deferred it).
#
# ```python
# hits = retriever.search("What was Q3 free cash flow?", top_k=3)   # [(page_image, score), ...]
# ```

# %%
if HAVE_MULTIMODAL:
    hits = retriever.search("What was Q3 free cash flow?", top_k=3)
    for page_ref, score in hits:
        print(f"score {score:.3f}  ->  {page_ref}")
else:
    print("(skipped: search runs the same VLM forward pass on the query)")

# %% [markdown]
# ## The number that justifies the price, and the price
# On the **ViDoRe** benchmark, visual retrieval scores **81.3 nDCG@5** against **67.0** for the
# strongest OCR-then-embed text pipeline - and indexes ~18x faster (no parse step). That delta is the
# entire business case.
#
# The price is **storage**: per-page multi-vectors run ~30x a single text embedding, and the index
# needs a *late-interaction* search path, not a plain ANN top-k. That is the multi-vector storage
# tradeoff of **Chapter 4** - budget for it before you commit. Apply token pooling from day one
# (`token_pooling=True`, on by default) or you have only deferred the extraction cost into your
# storage bill.

# %%
print("ship visual retrieval only when the quality delta is real AND the storage line is affordable.")
print("ViDoRe: 81.3 vs 67.0 nDCG@5; storage ~30x; token pooling is non-optional. Cross-ref Ch 4.")
