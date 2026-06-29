# Chapter 7 — Reranking

> 📖 Companion to **Book Chapter 7: Re-ranking**. Read the chapter, then run these top-to-bottom.
> Everything for this chapter is in this folder — walkthroughs, the experiment, and links to the
> production code. Nothing to hunt for.

## What you'll be able to do after this folder

Take a recall-optimized candidate list from first-stage retrieval and put the *right* evidence at
the top — with a cross-encoder, a listwise-LLM reranker, or a multi-stage cascade — and know
exactly what each costs you in latency before you ship it.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_cross_encoder.ipynb` · `01_cross_encoder.py` | Cross-encoder reranking — jina-reranker-v3 (§ "Cross-encoder rerankers") | [`professional_rag_kit/retrieval/rerank/cross_encoder.py`](../../professional_rag_kit/retrieval/rerank/) |
| 2 | `02_listwise_llm.ipynb` | Listwise LLM reranking — RankZephyr-style (§ "Listwise LLM rerankers") | [`professional_rag_kit/retrieval/rerank/listwise.py`](../../professional_rag_kit/retrieval/rerank/) |
| 3 | `03_colbert_tier.ipynb` | ColBERT late-interaction tier (§ "Late interaction as a reranker") | [`professional_rag_kit/retrieval/rerank/colbert.py`](../../professional_rag_kit/retrieval/rerank/) |
| 4 | `04_cascade.ipynb` · `04_cascade.py` | Multi-stage cascade + latency budgeting (§ "Cascade ranking") | [`professional_rag_kit/retrieval/rerank/cascade.py`](../../professional_rag_kit/retrieval/rerank/) |

## The experiment — `reproduce.py`

```bash
make ch07            # launch the notebooks
python chapters-companion/ch07-reranking/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's central claim on the golden set: **reranker on vs. off**, and **listwise
LLM vs. a distilled cross-encoder** — printing a *quality* delta (nDCG@10, answer correctness)
**and** a *cost* delta (added p95 latency, $/1k queries). The point of the chapter in one table:
the reranker is the highest-leverage quality lever *and* often 60–84% of retrieval-pipeline
latency. You decide with both numbers in front of you.

## Lift it into your project

```python
from professional_rag_kit.retrieval import Reranker

reranker = Reranker.default()                 # jina-reranker-v3, open, self-host
top5 = reranker.rerank(query, candidates)[:5]

# managed swap — one line, no rewrite:
reranker = Reranker.from_provider("cohere")   # needs COHERE_API_KEY
```

## Ship-this verdict (from the book)

> Add a cross-encoder reranker first — it's the highest return-on-effort change after a working
> first stage. Self-host **jina-reranker-v3** (open SOTA, BEIR 61.94); reach for a listwise LLM
> reranker only when you've measured the cross-encoder's ceiling binding, and budget for its 1–2s
> latency. Put it in a cascade so the expensive tier only ever sees a handful of candidates.

## Prerequisites

`make setup && make up` (installs `professional_rag_kit`, loads sample data, starts Qdrant). First-stage
retrieval is [Chapter 6](../ch06-retrieval/); what the generator does with the reranked context is
[Chapter 8](../ch08-context-construction/).
