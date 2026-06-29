# Chapter 12 — Multimodal RAG

> 📖 Companion to **Book Chapter 12: Multimodal RAG**. Read the chapter, then run these
> top-to-bottom. Everything for this chapter is in this folder — walkthroughs, the experiment, and
> links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Ask one question of your corpus — *is the meaning in the layout, or only in the words?* — and act on
it. When it's in the layout, you'll retrieve over page *images* with ColPali / ColQwen2.5 (no OCR, no
chunker), apply token pooling from day one to blunt the ~30× storage tax, and embed individual
charts and tables *natively* rather than as lossy text summaries. When it's only in the words, you'll
stay with text RAG and not pay for vision.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_ocr_then_embed_loses_information.ipynb` | Why the OCR/layout/chunk cascade leaks — and what it costs (§ "Why OCR-then-embed loses information") | [`professional_rag_kit/architectures/multimodal/`](../../professional_rag_kit/architectures/multimodal/) |
| 2 | `02_colpali_colqwen.ipynb` · `02_colpali_colqwen.py` | Visual document retrieval — page image + MaxSim late interaction; ColQwen2.5 backbone (§ "Visual document retrieval") | [`professional_rag_kit/architectures/multimodal/colpali.py`](../../professional_rag_kit/architectures/multimodal/) |
| 3 | `03_token_pooling.ipynb` · `03_token_pooling.py` | Token pooling — −66.7% vectors, −2.2% quality, the day-one storage mitigation (§ "The tradeoff and where it breaks") | [`professional_rag_kit/architectures/multimodal/colpali.py`](../../professional_rag_kit/architectures/multimodal/) |
| 4 | `04_unified_vs_late_interaction.ipynb` | CLIP/SigLIP unified space vs. multi-vector late interaction — pick by what the query points at (§ "Unified versus separate multimodal embedding spaces") | [`professional_rag_kit/architectures/multimodal/unified.py`](../../professional_rag_kit/architectures/multimodal/) |
| 5 | `05_native_object_embedding.ipynb` | Charts/tables/figures embedded natively vs. image-to-text summaries (§ "Retrieving images, tables, and charts as first-class objects") | [`professional_rag_kit/architectures/multimodal/objects.py`](../../professional_rag_kit/architectures/multimodal/) |

## The experiment — `reproduce.py`

```bash
make ch12            # launch the notebooks
python chapters-companion/ch12-multimodal/reproduce.py   # or just the headline experiment
```

Reproduces this chapter's master-map comparison: **ColPali visual retrieval vs. OCR-then-embed** on
visually-rich page images — the nDCG@5 delta that is the entire business case — plus the
**native-object vs. image-to-text** comparison on number-dense charts and tables. Crucially it pairs
the quality number with the **storage line**: pages × ~257.5 KB (or ÷3 with token pooling) vs. pages
× ~8.6 KB for text. The book's rule made runnable — ship visual retrieval only when the quality delta
is real *and* the storage is affordable. Benchmark on ViDoRe V2, never the saturated V1.

## Lift it into your project

```python
from professional_rag_kit.architectures.multimodal import VisualRetriever

retriever = VisualRetriever.default()                 # ColQwen2.5 + token pooling from day one
retriever.index(pages, token_pooling=True)
top5 = retriever.search("Q3 APAC revenue from the regional bar chart", top_k=5)
# the retrieved unit is a PAGE IMAGE — requires a multimodal generator downstream, or you have
# only deferred extraction, not escaped it.
```

## Ship-this verdict (from the book)

> The multimodal decision reduces to a single question asked of your corpus: *is the meaning in the
> layout, or only in the words?* When it is in the layout — visually-structured documents,
> number-dense charts, slides — retrieve over the pixels (*ColPali/ColQwen* for pages, native
> multimodal embedding for objects) and accept the multi-vector storage tax, mitigated with token
> pooling; the quality and indexing-speed wins are large and the brittle OCR/chunking front end
> disappears. When the meaning is only in the words — clean text, talk-driven audio — stay with text
> RAG and do not pay for vision. Name the corpus's visual structure first; let it, not the benchmark
> leaderboard, decide whether you go multimodal.

## Prerequisites

`make setup && make up` (installs `professional_rag_kit`, loads sample data, starts Qdrant). The MaxSim late-
interaction embedding mechanism and the per-patch storage economics are [Chapter 4](../ch04-embeddings/);
the document parsing that visual retrieval lets you skip is [Chapter 2](../ch02-document-processing/);
the page-too-coarse trimming problem this can introduce is [Chapter 8](../ch08-context-construction/).
