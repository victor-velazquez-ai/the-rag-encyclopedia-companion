# Chapter 6 — Retrieval & Query Understanding

> 📖 Companion to **Book Chapter 6: Retrieval Strategies and Query Understanding**. Read the
> chapter, then run these top-to-bottom. Everything for this chapter is in this folder —
> walkthroughs, the experiment, and links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Get the right candidates *into the set* — because nothing downstream can recover a document the
first stage never surfaced. You'll run BM25 + dense fused with RRF, stand up SPLADE-v3 as a learned
sparse leg, reshape the query itself with HyDE / step-back / multi-query / decomposition, and put a
cheap router in front so each query takes only the machinery it deserves — the keystone idea of the
chapter.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_hybrid_rrf.ipynb` · `01_hybrid_rrf.py` | BM25 + dense fused with Reciprocal Rank Fusion, k=60 (§ "Hybrid search and Reciprocal Rank Fusion") | [`professional_rag_kit/retrieval/hybrid/`](../../professional_rag_kit/retrieval/hybrid/) |
| 2 | `02_splade_sparse_leg.ipynb` | SPLADE-v3 learned sparse retrieval — term expansion on one inverted index (§ "Learned sparse retrieval: SPLADE-v3") | [`professional_rag_kit/retrieval/hybrid/`](../../professional_rag_kit/retrieval/hybrid/) |
| 3 | `03_hyde.ipynb` · `03_hyde.py` | HyDE — hypothetical document embeddings (§ "HyDE") | [`professional_rag_kit/retrieval/query/`](../../professional_rag_kit/retrieval/query/) |
| 4 | `04_step_back.ipynb` | Step-back prompting — abstract, then retrieve (§ "Step-back prompting") | [`professional_rag_kit/retrieval/query/`](../../professional_rag_kit/retrieval/query/) |
| 5 | `05_multi_query_decomposition.ipynb` | Multi-query paraphrase + multi-hop decomposition, fused with RRF (§ "Multi-query and decomposition") | [`professional_rag_kit/retrieval/query/`](../../professional_rag_kit/retrieval/query/) |
| 6 | `06_semantic_routing.ipynb` · `06_semantic_routing.py` | Semantic router — embed + k-NN intent routing (~5000 ms → ~100 ms vs an LLM router) (§ "Semantic routing") | [`professional_rag_kit/retrieval/routing/`](../../professional_rag_kit/retrieval/routing/) |
| 7 | `07_adaptive_rag_router.ipynb` | Adaptive-RAG complexity classifier — no-retrieval / single-step / multi-step (§ "Adaptive-RAG") | [`professional_rag_kit/retrieval/routing/`](../../professional_rag_kit/retrieval/routing/) |

## The experiment — `reproduce.py`

```bash
make ch06            # launch the notebooks
python chapters-companion/ch06-retrieval/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's central claims on the golden set: **hybrid+RRF vs. dense-only** (does
fusing the two legs beat either alone, and by how much on the identifier-bearing queries dense
blurs), and **HyDE on a strong vs. a weak embedder** (the lift that is +16 to +32 nDCG\@10 against
unsupervised Contriever and shrinks — or reverses — against a modern fine-tuned embedder). It prints
a *quality* delta (nDCG\@10, Recall\@k) **and** a *cost* delta (added LLM-call latency, $/1k
queries), because the chapter's whole argument is that the expensive path must earn its place.

## Lift it into your project

```python
from professional_rag_kit.retrieval import HybridRetriever

retriever = HybridRetriever.from_config("configs/default.yaml")   # BM25 + dense + RRF (k=60)
hits = retriever.search("ERR_CONN_4021 on checkout", top_k=50)    # exact ID + semantic, fused

# route the expensive query transforms on, don't turn them on globally:
retriever = HybridRetriever.from_config("configs/default.yaml", query_transform="auto")
```

## Ship-this verdict (from the book)

> Build retrieval as a routed hybrid, not a single pipeline. Default every query to BM25 + dense
> fused with RRF (k = 60, swept), because the legs fail on opposite inputs and ranks dodge the
> score-scale mismatch that breaks linear blending. Layer a cheap classifier (Adaptive-RAG-style) in
> front so simple queries skip retrieval, standard queries get the hybrid, and only true multi-hop
> queries pay for the iterative loops of Part III — and route model/effort the same way (RouteLLM,
> vLLM Semantic Router), where the prize is often *cheaper and more accurate* at once. Add query
> transformation (HyDE, step-back, decomposition) only on the routed slice that needs it, and only
> after measuring that it still beats your embedder.

## Prerequisites

`make setup && make up` (installs `professional_rag_kit`, loads sample data, starts Qdrant). This chapter
produces the candidate set that the reranker scores in [Chapter 7](../ch07-reranking/); how the
surviving passages are ordered, compressed, and assembled is [Chapter 8](../ch08-context-construction/).
The embedding model behind the dense leg is [Chapter 4](../ch04-embeddings/); the index that stores
it is [Chapter 5](../ch05-vector-stores/).
