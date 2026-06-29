# Chapter 9 — GraphRAG and Knowledge Graphs

> 📖 Companion to **Book Chapter 9: GraphRAG and Knowledge Graphs**. Read the chapter, then run
> these top-to-bottom. Everything for this chapter is in this folder — walkthroughs, the experiment,
> and links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Decide — on *your* query log, not a vibe — whether a graph earns its cost, then build the right one:
LightRAG when the corpus is living and updates matter, LazyGraphRAG when the index budget binds, and
full Microsoft GraphRAG only when bounded, slow-changing, global-sensemaking traffic justifies the
bill. You'll read each system's win rate against *its own* stated baseline and never stack them on
one axis.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_when_a_graph_earns_its_cost.ipynb` | The test for a graph-shaped problem — global vs. multi-hop vs. local (§ "Why similarity search cannot answer a global question") | [`professional_rag_kit/architectures/graph/`](../../professional_rag_kit/architectures/graph/) |
| 2 | `02_lightrag.ipynb` · `02_lightrag.py` | LightRAG — dual-level retrieval + incremental updates, the default (§ "LightRAG") | [`professional_rag_kit/architectures/graph/light_rag.py`](../../professional_rag_kit/architectures/graph/) |
| 3 | `03_lazy_graphrag.ipynb` · `03_lazy_graphrag.py` | LazyGraphRAG — co-occurrence graph, ~0.1% index cost (§ "LazyGraphRAG") | [`professional_rag_kit/architectures/graph/lazy_graph.py`](../../professional_rag_kit/architectures/graph/) |
| 4 | `04_microsoft_graphrag.ipynb` | Full GraphRAG — Leiden communities, reports, Global/Local/DRIFT modes (§ "Microsoft GraphRAG") | [`professional_rag_kit/architectures/graph/graph_rag.py`](../../professional_rag_kit/architectures/graph/) |
| 5 | `05_hipporag_ppr.ipynb` | HippoRAG 2 — Personalized PageRank over an LLM graph (§ "HippoRAG 2", frontier) | [`professional_rag_kit/architectures/graph/`](../../professional_rag_kit/architectures/graph/) |

## The experiment — `reproduce.py`

```bash
make ch09            # launch the notebooks
python chapters-companion/ch09-graphrag/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's central comparison from the master map: **GraphRAG vs. LightRAG vs. vector
RAG on local-fact traffic** — and, on a set of genuinely global questions, an answer-quality
head-to-head (comprehensiveness / diversity win rates, *never* nDCG). It prints, for each system, the
**index-time token bill and update cost** alongside the quality number — because the whole chapter's
discipline is that a graph's quality win must justify its index and rebuild cost, and the win rates
must each be read against *their own* baseline.

## Lift it into your project

```python
from professional_rag_kit.architectures.graph import GraphRAG

graph = GraphRAG.default()                 # LightRAG — graph + incremental updates
graph.index(corpus)
answer = graph.query("what are the recurring root causes across this quarter's outages?")

# escalate only when measurement says so — one line, no rewrite:
graph = GraphRAG.variant("graphrag")       # full Microsoft GraphRAG (bounded, global-first corpora)
```

## Ship-this verdict (from the book)

> Most teams should ship vector RAG and add a graph only when the query log proves global or
> multi-hop questions are a recurring, meaningful share of traffic — then start with LightRAG or
> LazyGraphRAG, not full GraphRAG. LightRAG is the pragmatic default for a living corpus (incremental
> updates, retrieval in fewer than 100 tokens, competitive *against GraphRAG*); LazyGraphRAG when
> index budget binds (~0.1% of GraphRAG's index cost, comparable global quality at >700× lower query
> cost). Spend the full Microsoft GraphRAG index bill only when global sensemaking is the whole point
> and the corpus is bounded enough to rebuild. Demand answer-quality evaluation (comprehensiveness /
> diversity win rates), never nDCG, for global questions — and always state *which* baseline each win
> rate is measured against, because in this chapter they are not the same.

## Prerequisites

`make setup && make up` (installs `professional_rag_kit`, loads sample data, starts Qdrant). The flat baseline a
graph must beat is [Chapter 6](../ch06-retrieval/) + [Chapter 7](../ch07-reranking/); the cheaper
hierarchical and adaptive machinery for overlapping problems is [Chapter 10](../ch10-adaptive-rag/);
agentic multi-hop and the entity resolution that deduplicates graph nodes is
[Chapter 11](../ch11-agentic-rag/).
