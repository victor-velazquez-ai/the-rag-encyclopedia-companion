# Chapter 16 — Performance, Cost, Latency, and Observability

> 📖 Companion to **Book Chapter 16: Performance, Cost, Latency, and Observability**. Read the
> chapter, then run these top-to-bottom. Everything for this chapter is in this folder — walkthroughs,
> the experiment, and links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Make the system *survivable* — the work that does not show up in a quality metric. You'll ship a
semantic cache gated on a measured hit rate and a per-prompt correctness threshold (vCache-style),
decompose the staged latency budget and tune the *dominant* term, set the SLA on P99 (not the mean),
run vLLM continuous batching + PagedAttention for self-hosted throughput, make prompt caching and model
routing your two biggest cost levers, and instrument from day one with OTel/OpenInference spans, three-
surface drift detection, and a thumbs→annotation→eval-gate feedback loop. The maturity ladder
(L0 naive → L5 enterprise) names the destination.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_semantic_cache.ipynb` · `01_semantic_cache.py` | Semantic cache + the grey-zone problem; vCache learned per-prompt threshold (§ "Semantic caching") | [`professional_rag_kit/production/serving/`](../../professional_rag_kit/production/serving/) |
| 2 | `02_latency_budget.ipynb` | The staged budget — generation dominates, the reranker is the retrieval-half surprise (§ "The staged-pipeline latency budget") | [`professional_rag_kit/production/serving/`](../../professional_rag_kit/production/serving/) |
| 3 | `03_tail_latency_p99.ipynb` | P99 over the mean; TTFT, inter-token latency, streaming hides total latency (§ "Tail latency") | [`professional_rag_kit/production/serving/`](../../professional_rag_kit/production/serving/) |
| 4 | `04_continuous_batching.ipynb` | vLLM continuous batching + PagedAttention + chunked prefill (§ "Batching and streaming") | [`professional_rag_kit/production/serving/`](../../professional_rag_kit/production/serving/) |
| 5 | `05_prompt_cache_and_routing.ipynb` · `05_prompt_cache_and_routing.py` | Stable-prefix-first prompt caching + FrugalGPT/RouteLLM model routing (§ "Cutting cost") | [`professional_rag_kit/production/serving/`](../../professional_rag_kit/production/serving/) |
| 6 | `06_tracing.ipynb` · `06_tracing.py` | OTel GenAI + OpenInference span kinds (RETRIEVER→RERANKER→LLM) (§ "Observability: tracing standards") | [`professional_rag_kit/production/observability/`](../../professional_rag_kit/production/observability/) |
| 7 | `07_drift_detection.ipynb` | Three-surface drift; centroid-cosine / MMD / classifier, never univariate KS on embeddings (§ "Drift detection") | [`professional_rag_kit/production/observability/`](../../professional_rag_kit/production/observability/) |
| 8 | `08_feedback_loop.ipynb` | Thumbs → annotation queue → eval dataset; online LLM-judge; inline guardrail (§ "Feedback loops") | [`professional_rag_kit/production/observability/`](../../professional_rag_kit/production/observability/) |

## The experiment — `reproduce.py`

```bash
make ch16            # launch the notebooks
python chapters-companion/ch16-performance-observability/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's central claims on the golden set: a **semantic-cache correctness** test —
sweeping a static similarity threshold to expose the grey zone where correct and incorrect cache hits
overlap, then showing a vCache-style per-prompt error-bounded threshold lift the hit rate at the same
error — and a **model-routing cost** test that routes the golden set through a cheap/strong cascade and
reports quality retained vs. strong-model calls saved. It prints a *cost/latency* delta (hit rate,
$/1k queries, P99, TTFT) **and** a *quality* delta (faithfulness, correctness held at the routing
threshold), because the chapter's whole argument is that the levers that keep a system *alive* must not
cost the quality the earlier chapters bought.

## Lift it into your project

```python
from professional_rag_kit.production.serving import SemanticCache, Router
from professional_rag_kit.production.observability import Tracer

cache = SemanticCache.default(target_error=0.02, ttl_seconds=3600)  # error-bounded, not a static knob
router = Router.default()                          # cheap model first; escalate only when it must
tracer = Tracer.default()                          # OTel + OpenInference spans from day one
```

## Ship-this verdict (from the book)

> Treat performance, cost, and observability as a single operate-and-measure loop, not three
> afterthoughts. The five committed defaults that ship in 2026: a *measured-hit-rate, error-bounded
> semantic cache* with a TTL; a *tail-latency (P99) budget* tuned at the generation stage with
> streaming and a sub-200 ms TTFT; *vLLM continuous batching + PagedAttention* for self-hosted
> throughput; *prompt caching (stable prefix first) plus model routing* as the dominant cost levers;
> and *day-one tracing, three-surface drift detection, and a thumbs-to-eval-gate feedback loop*. The
> techniques that make a RAG system *good* are in the chapters before this one; the techniques that
> keep it *alive* are here. Ship both.

## Prerequisites

`make setup && make up` (installs `professional_rag_kit`, loads sample data, starts Qdrant). The reranker that is the
retrieval-half latency surprise is [Chapter 7](../ch07-reranking/); the eval *methodology* the online
judge and feedback gate run — RAGAS, the RAG Triad, judge calibration — is
[Chapter 14](../ch14-evaluation/); and the *security* guardrail against injection (vs. the operational
quality guardrail here) is [Chapter 15](../ch15-enterprise-hardening/). This is the production-operations
capstone — the last chapter before the book's capstone system.
