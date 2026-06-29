# Chapter 8 — Context Construction & Assembly

> 📖 Companion to **Book Chapter 8: Context Construction and Assembly**. Read the chapter, then run
> these top-to-bottom. Everything for this chapter is in this folder — walkthroughs, the experiment,
> and links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Turn a ranked shortlist into the exact string of tokens the model reads — and recover the quality
naive concatenation throws away. You'll fold the list against *lost-in-the-middle* so the strongest
evidence lands on the two attention peaks, budget the window in tokens and rerank-then-truncate,
compress with LLMLingua only when a passage must stay but can't be afforded whole, de-duplicate with
MMR on the reranked shortlist, arbitrate contradictions as a separate stage, and decide per query
between tight retrieval and a long-context read (Self-Route).

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_lost_in_the_middle_fold.ipynb` · `01_lost_in_the_middle_fold.py` | The zipper fold — strongest evidence first and last (§ "Lost in the middle: position is a feature") | [`professional_rag_kit/retrieval/context/`](../../professional_rag_kit/retrieval/context/) |
| 2 | `02_rerank_then_truncate.ipynb` | Budget in tokens, not passages; truncate after ranking, reserve answer headroom (§ "Budgeting the window") | [`professional_rag_kit/retrieval/context/`](../../professional_rag_kit/retrieval/context/) |
| 3 | `03_prompt_compression.ipynb` · `03_prompt_compression.py` | LLMLingua / LongLLMLingua / LLMLingua-2 prompt compression (§ "Prompt compression") | [`professional_rag_kit/retrieval/context/`](../../professional_rag_kit/retrieval/context/) |
| 4 | `04_mmr_diversity.ipynb` | MMR on the reranked shortlist — kill redundancy, λ≈0.6–0.7 (§ "MMR and diversity") | [`professional_rag_kit/retrieval/context/`](../../professional_rag_kit/retrieval/context/) |
| 5 | `05_dedup_contradiction.ipynb` | Dedup + inter-context contradiction arbitration (decomposition · source-reliability · recency/authority) (§ "Deduplication and contradiction") | [`professional_rag_kit/retrieval/context/`](../../professional_rag_kit/retrieval/context/) |
| 6 | `06_self_route_long_context.ipynb` · `06_self_route_long_context.py` | Self-Route — retrieve-and-self-assess vs. stuff the window (§ "Long context versus tighter retrieval") | [`professional_rag_kit/retrieval/context/`](../../professional_rag_kit/retrieval/context/) |

## The experiment — `reproduce.py`

```bash
make ch08            # launch the notebooks
python chapters-companion/ch08-context-construction/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's central claims on the golden set: **lost-in-the-middle ordering** (hold the
passage set fixed, vary only assembly order — rank-order vs. the zipper fold vs. a random shuffle —
and read off the accuracy delta you reclaim for free) and **compression vs. retrieve-less** (does a
LLMLingua compressor beat "just send fewer, better passages" once you charge it for its *own* call).
It prints a *quality* delta (exact-match / LLM-judge faithfulness) **and** a *cost* delta (total
latency and tokens, compressor included) — because the chapter's recurring lesson is that the
expensive method does not always win, and the highest-ROI move here costs nothing.

## Lift it into your project

```python
from professional_rag_kit.retrieval import ContextBuilder

builder = ContextBuilder(token_budget=4000)        # tokens, not passages; headroom reserved
context = builder.build(query, reranked_passages)  # fold + truncate, lost-in-the-middle aware

# add the conditional stages only where measurement licenses them:
builder = ContextBuilder(token_budget=4000, mmr=0.6, compress="longllmlingua")
```

## Ship-this verdict (from the book)

> Build context assembly as an ordered pipeline — *order, budget, compress, diversify, arbitrate,
> route* — and spend complexity only where measurement licenses it. The free wins (the
> lost-in-the-middle fold, rerank-then-truncate, dedup) come first and clear most of the bar;
> compression, MMR, and contradiction arbitration are conditional additions you justify on your own
> data. Hold one rule above all: *a long context window does not replace careful assembly* — it
> re-exposes lost-in-the-middle at full scale and costs more. Route, compress, and order the context
> the model sees; that, not window size, is what turns a ranked list into a correct answer.

## Prerequisites

`make setup && make up` (installs `professional_rag_kit`, loads sample data, starts Qdrant). The ranked shortlist
this chapter assembles comes from first-stage [Chapter 6](../ch06-retrieval/) and the reranker in
[Chapter 7](../ch07-reranking/); the zipper fold and MMR both assume that reranked ordering, so run
those first. How the generator grounds and cites the assembled context is Chapter 13.
