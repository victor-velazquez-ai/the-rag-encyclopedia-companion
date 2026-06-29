# Chapter 3 — Chunking and Segmentation

> 📖 Companion to **Book Chapter 3: Chunking and Segmentation**. Read the chapter, then run these
> top-to-bottom. Everything for this chapter is in this folder — walkthroughs, the experiment, and
> links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Cut documents into the atomic units everything downstream embeds, retrieves, and reads — without
severing facts from the context that makes them answerable. You'll ship the strong, cheap baseline
(a tuned ~200-token recursive splitter), and know exactly which expensive method — semantic, late,
contextual, propositional, parent-child — earns its index-time cost *on your data*, with the
numbers to prove it before you spend an LLM budget.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_recursive_baseline.ipynb` · `01_recursive_baseline.py` | Structure-aware recursive split, ~200 tok + overlap — the baseline (§ "The baseline") | [`professional_rag_kit/ingestion/chunking/recursive.py`](../../professional_rag_kit/ingestion/chunking/) |
| 2 | `02_semantic_chunking.ipynb` | Embedding-breakpoint semantic chunking (§ "Semantic chunking") | [`professional_rag_kit/ingestion/chunking/semantic.py`](../../professional_rag_kit/ingestion/chunking/) |
| 3 | `03_propositional.ipynb` | Propositional / Dense-X atomic chunking (§ "Propositional (Dense-X) chunking") | [`professional_rag_kit/ingestion/chunking/propositional.py`](../../professional_rag_kit/ingestion/chunking/) |
| 4 | `04_late_chunking.ipynb` | Late Chunking — embed whole doc, then segment + pool (§ "Late Chunking") | [`professional_rag_kit/ingestion/chunking/late.py`](../../professional_rag_kit/ingestion/chunking/) |
| 5 | `05_contextual_retrieval.ipynb` · `05_contextual_retrieval.py` | Anthropic Contextual Retrieval — LLM context + BM25 (§ "Anthropic Contextual Retrieval") | [`professional_rag_kit/ingestion/chunking/contextual.py`](../../professional_rag_kit/ingestion/chunking/) |
| 6 | `06_small_to_big.ipynb` | Small-to-big / sentence-window / auto-merging (§ "Hierarchical retrieval") | [`professional_rag_kit/ingestion/chunking/hierarchical.py`](../../professional_rag_kit/ingestion/chunking/) |
| 7 | `07_metadata.ipynb` | Chunk-time metadata enrichment (§ "Metadata enrichment at chunk time") | [`professional_rag_kit/ingestion/chunking/metadata.py`](../../professional_rag_kit/ingestion/chunking/) |

## The experiment — `reproduce.py`

```bash
make ch03            # launch the notebooks
python chapters-companion/ch03-chunking/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's central claim on the golden set: **semantic chunking vs. a tuned ~200-token
recursive splitter**, and **Contextual Retrieval on vs. off** — printing a *quality* delta (recall,
nDCG@10, top-20 retrieval failure rate) **and** a *cost* delta (index-time LLM \$/M tokens, extra
embedding passes). The point of the chapter in one table: the cheap, structure-respecting baseline
is strong and is the line every expensive method must clear on *your* data — semantic chunking
usually loses on structured docs, while Contextual Retrieval's halved failure rate (5.7% → 2.9%,
→1.9% with rerank) is the one index-time LLM win worth defaulting to.

## Lift it into your project

```python
from professional_rag_kit.ingestion import Chunker

chunks = Chunker.default().split(doc)                  # structure-aware recursive ~200 tok + metadata

# escalate only when measurement says it pays — one line, no rewrite:
chunks = Chunker.strategy("contextual").split(doc)     # Anthropic Contextual Retrieval (needs LLM)
```

## Ship-this verdict (from the book)

> Ship a *structure-aware recursive splitter at ~200 tokens with ~10–20% overlap*, operating on
> the structured markup your parser emits. This is the default for the overwhelming majority of
> RAG systems and the baseline every other method here must beat. Do not replace it until you
> have measured a specific, repeatable retrieval win from a more expensive method *on your own
> documents*. Tune size and overlap on your corpus; expect a plateau, not a knife-edge.

And on the one index-time LLM method worth defaulting to:

> This is the one index-time LLM method worth defaulting to when your corpus is past the
> ~200k-token RAG threshold and suffers from context-stripped chunks. Contextual Embeddings plus
> Contextual BM25 cut top-20 retrieval failures by *49%* (5.7% → 2.9%), and adding a reranker
> reaches *67%* (→ 1.9%), for about *\$1.02 per million tokens* with Claude 3 Haiku and prompt
> caching. Below ~200k tokens, do not build RAG at all — put the corpus in the prompt.

## Prerequisites

`make setup && make up` (installs `professional_rag_kit`, loads sample data, starts Qdrant). Chunking consumes the
structured text from [Chapter 2](../ch02-document-processing/); the chunks it emits are embedded in
[Chapter 4](../ch04-embeddings/).
