# Chapter 5 — Vector Stores and Indexing at Scale

> 📖 Companion to **Book Chapter 5: Vector Stores and Indexing at Scale**. Read the chapter, then
> run these top-to-bottom. Everything for this chapter is in this folder — walkthroughs, the
> experiment, and links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Treat the index as an engineering decision with a defensible answer, not a procurement choice made
on vibes. You'll pick the ANN family from your *binding constraint* (HNSW, IVF-PQ, ScaNN, DiskANN),
turn on in-index quantization where the real cost savings live, and — most importantly — make
filtered ANN a first-class concern so a multi-tenant system doesn't silently return the wrong
neighbors. Every failure mode in this chapter is one the system won't report on its own; you'll
learn to measure recall against an exhaustive baseline and see them.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus on **Qdrant**. The
"key" ones are also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_hnsw_tradeoff.ipynb` · `01_hnsw_tradeoff.py` | HNSW + the recall/latency/memory/dollar sweep (§ "The recall/latency/memory/dollar tradeoff") | [`ragkit/ingestion/indexing/ann.py`](../../ragkit/ingestion/indexing/) |
| 2 | `02_ann_families.ipynb` | IVF-PQ · ScaNN · DiskANN — choosing by constraint (§ "The recall/latency/memory/dollar tradeoff") | [`ragkit/ingestion/indexing/ann.py`](../../ragkit/ingestion/indexing/) |
| 3 | `03_filtered_ann.ipynb` · `03_filtered_ann.py` | Filterable HNSW vs. naive pre/post-filtering (§ "Filtered ANN: where naive systems silently break") | [`ragkit/ingestion/indexing/filtered.py`](../../ragkit/ingestion/indexing/) |
| 4 | `04_quantization.ipynb` · `04_quantization.py` | In-index int8 / binary / RaBitQ + rescore (§ "In-index quantization") | [`ragkit/ingestion/indexing/quantize.py`](../../ragkit/ingestion/indexing/) |
| 5 | `05_pgvector_vs_dedicated.ipynb` | pgvector(+scale) vs. a dedicated store (§ "pgvector versus a dedicated store") | [`ragkit/ingestion/indexing/store.py`](../../ragkit/ingestion/indexing/) |
| 6 | `06_freshness_compaction.ipynb` | Incremental insert, tombstones, compaction scheduling (§ "Incremental indexing and the freshness problem") | [`ragkit/ingestion/indexing/freshness.py`](../../ragkit/ingestion/indexing/) |

## The experiment — `reproduce.py`

```bash
make ch05            # launch the notebooks
python chapters/ch05-vector-stores/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's central claims on the golden set: **HNSW vs. quantized recall** (int8 /
binary / RaBitQ + rescore), and **the filtered-ANN recall collapse** — running naive post-filtering
against Qdrant's filterable HNSW under an adversarial, negatively-correlated filter. Prints a
*quality* delta (recall@k vs. an *exhaustive filtered* baseline) **and** a *cost* delta (QPS, p99
latency, bytes/vector, RAM). The point of the chapter in one table: naive post-filtering can fail to
reach 0.9 recall and *never says so*, and the master pattern — quantize for the first pass, rescore
the shortlist with full precision — is what makes 32× compression safe.

## Lift it into your project

```python
from ragkit.ingestion import VectorStore

store = VectorStore.default()                          # Qdrant: filterable HNSW + int8 + float rescore
store.upsert(chunks)                                   # incremental insert into the graph
hits = store.search(qvec, top_k=10,
                    filter={"tenant_id": 42},          # filter-aware: payload-aware HNSW, not post-filter
                    oversample=4)                      # oversample the 1-bit first pass, then rescore

# scale up when the constraint changes — one line, no rewrite:
store = VectorStore.connect(url, index="diskann", quantization="rabitq")
```

## Ship-this verdict (from the book)

> Choose the index from your binding constraint, not your habits: *HNSW* when the working set
> fits in RAM and latency is the priority, *DiskANN* when it does not, *IVF-PQ or RaBitQ* when
> memory is the wall. Apply *int8 quantization always* and *RaBitQ (1-bit, 32×)* when storage
> dominates — and always *quantize for the first pass, then rescore the shortlist with full
> precision.* Make *filtered ANN* a first-class index requirement, never an afterthought:
> naive post-filtering silently collapses recall on selective or security-bearing filters.
> Stay on *pgvector* while you are inside its scale-and-feature envelope and graduate to a
> dedicated store (the companion repo standardizes on *Qdrant*) when filtered-ANN guarantees,
> in-index quantization, or billion-scale sharding become the requirement — and validate every
> one of these choices by measuring recall against an exhaustive baseline under your own filters.

## Prerequisites

`make setup && make up` (installs `ragkit`, loads sample data, starts Qdrant). This store indexes
the vectors produced in [Chapter 4](../ch04-embeddings/); searching and ordering what it returns is
first-stage retrieval in [Chapter 6](../ch06-retrieval/).
