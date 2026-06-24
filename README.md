# The RAG Encyclopedia — Official Companion

The hands-on companion to **_The RAG Encyclopedia: Advanced Enterprise Retrieval-Augmented
Generation_** by Victor B.

The book teaches you *what* to build and *why*, and commits to a verdict on every technique.
This repository is where you **run it, measure it, and lift it into your own systems** — a
chapter-by-chapter learning lab and a clean, importable, production-grade library, tightly
cross-linked so you never hunt for anything.

> **🚧 Status — Phase 1: navigable scaffold.** Right now this repo is the *complete structure*
> with a detailed README in every folder describing exactly what will be built. Runnable
> notebooks and the `ragkit` implementation land in Phase 2. If a folder only has a `README.md`,
> that's expected for now — follow the [CHANGELOG](CHANGELOG.md).

---

## The one rule that organizes this repo

**The chapter is the front door.** Reading Chapter 7? Open [`chapters/ch07-reranking/`](chapters/ch07-reranking/)
and *everything* for that chapter is right there — the runnable walkthroughs, the run commands,
the reproduction experiment, and a direct link to the production code that implements it. There
is exactly **one learning axis** (the book's chapters) and **one reuse axis** (the `ragkit`
library), and they are cross-linked at every step. No four-folder scavenger hunt.

```
the-rag-encyclopedia-repo/
├── chapters/        ← THE LEARNING PATH — walk it in book order, hand by hand
├── ragkit/          ← THE REUSE SPINE — clean, importable, production-grade library
├── capstone/        ← the full enterprise RAG system, everything wired together
├── data/            ← small sample corpus + golden sets (runs with NO paid API key)
└── docs/            ← setup, how-to-use, provider-swap guide, dataset cards
```

---

## Two ways to use it

### 📖 "I'm reading the book"
Open the chapter folder, run the numbered notebooks top-to-bottom as you read. Each one narrates
a single technique and runs live against the sample data.

```bash
docker compose up -d          # Qdrant, locally, no keys
make setup                    # install ragkit + deps, load sample data
make ch07                     # launch Chapter 7's walkthroughs
```

### 🛠️ "I'm building my own system"
`import ragkit` and lift the production components straight into your project. Read
[`capstone/`](capstone/) for the assembled reference system, and swap any provider
(Qwen3→Voyage, jina→Cohere) by editing **one line** of config — no rewrite.

```python
from ragkit.retrieval import HybridRetriever, Reranker
from ragkit.ingestion import Embedder

retriever = HybridRetriever.from_config("configs/default.yaml")   # BM25 + dense + RRF
hits = retriever.search("your query", top_k=50)
ranked = Reranker.default().rerank("your query", hits)[:5]
```

---

## The master map

Find anything in one look. Each chapter maps to a folder, a run command, the production module
that implements it, and the head-to-head experiment it reproduces.

| Book Ch | Folder | Run | Production code (`ragkit/`) | Reproduces (quality **+** cost) |
|---|---|---|---|---|
| 2 · Document Processing | [`chapters/ch02-document-processing/`](chapters/ch02-document-processing/) | `make ch02` | `ingestion/parsing/` | VLM vs. classical OCR vs. hybrid parsing |
| 3 · Chunking | [`chapters/ch03-chunking/`](chapters/ch03-chunking/) | `make ch03` | `ingestion/chunking/` | semantic vs. tuned ~200-tok recursive; contextual retrieval |
| 4 · Embeddings | [`chapters/ch04-embeddings/`](chapters/ch04-embeddings/) | `make ch04` | `ingestion/embedding/` | self-host Qwen3 vs. managed; int8/binary + rescore vs. float |
| 5 · Vector Stores | [`chapters/ch05-vector-stores/`](chapters/ch05-vector-stores/) | `make ch05` | `ingestion/indexing/` | HNSW vs. quantized recall; filtered-ANN recall collapse |
| 6 · Retrieval & Routing | [`chapters/ch06-retrieval/`](chapters/ch06-retrieval/) | `make ch06` | `retrieval/{hybrid,query,routing}/` | hybrid+RRF vs. dense-only; HyDE on strong vs. weak embedder |
| 7 · Reranking | [`chapters/ch07-reranking/`](chapters/ch07-reranking/) | `make ch07` | `retrieval/rerank/` | reranker on/off; listwise-LLM vs. distilled cross-encoder |
| 8 · Context Construction | [`chapters/ch08-context-construction/`](chapters/ch08-context-construction/) | `make ch08` | `retrieval/context/` | lost-in-the-middle ordering; compression vs. retrieve-less |
| 9 · GraphRAG | [`chapters/ch09-graphrag/`](chapters/ch09-graphrag/) | `make ch09` | `architectures/graph/` | GraphRAG vs. LightRAG vs. vector on local-fact traffic |
| 10 · Hierarchical & Adaptive | [`chapters/ch10-adaptive-rag/`](chapters/ch10-adaptive-rag/) | `make ch10` | `architectures/{hierarchical,adaptive}/` | RAPTOR collapsed-tree; Adaptive/CRAG/Self-RAG/FLARE |
| 11 · Agentic & Multi-System | [`chapters/ch11-agentic-rag/`](chapters/ch11-agentic-rag/) | `make ch11` | `architectures/agentic/` | single-shot vs. agentic multi-hop; entity resolution |
| 12 · Multimodal | [`chapters/ch12-multimodal/`](chapters/ch12-multimodal/) | `make ch12` | `architectures/multimodal/` | ColPali visual retrieval vs. OCR-then-embed |
| 13 · Grounded Generation | [`chapters/ch13-grounded-generation/`](chapters/ch13-grounded-generation/) | `make ch13` | `production/generation/` | CAD / Chain-of-Note grounding; span-level citation |
| 14 · Evaluation | [`chapters/ch14-evaluation/`](chapters/ch14-evaluation/) | `make ch14` | `eval/` | RAGAS metrics; LLM-judge bias + calibration |
| 15 · Enterprise Hardening | [`chapters/ch15-enterprise-hardening/`](chapters/ch15-enterprise-hardening/) | `make ch15` | `production/security/` | retrieval-time ACL trimming; indirect-injection defense |
| 16 · Performance & Observability | [`chapters/ch16-performance-observability/`](chapters/ch16-performance-observability/) | `make ch16` | `production/{serving,observability}/` | semantic-cache correctness; routing cost cuts |

> Chapter 1 (Introduction) is conceptual — no code. Start your setup in [`docs/SETUP.md`](docs/SETUP.md).

---

## The reference stack (all-open, no paid key required)

Every default is the verdict-recommended choice from its chapter, self-hostable and permissively
licensed. Managed APIs appear only as **one-line config swaps** so you can benchmark them.

| Role | Default (open, self-host) | Managed swap |
|---|---|---|
| Vector store | **Qdrant** (Docker) | — |
| Embeddings | **Qwen3-Embedding-8B** (Apache-2.0) | Voyage · Gemini |
| Reranker | **jina-reranker-v3** (0.6B, open) | Cohere Rerank 4 |
| Sparse leg | **BM25** (+ optional SPLADE-v3) | — |
| Generator | self-host instruct model via **vLLM** | any OpenAI-compatible endpoint |

See [`docs/PROVIDER-SWAP.md`](docs/PROVIDER-SWAP.md) for how the one-line swap works.

---

## The evaluation harness — the book's spine, made runnable

`ragkit/eval/` plus each chapter's `reproduce.py` re-run the book's head-to-head comparisons on a
version-controlled golden set and print a **quality number and a cost number** — because the
book's argument is never quality alone. This is what makes every verdict in the book *falsifiable
on your own corpus*. Run the whole suite with `make reproduce`.

---

## Layout

- **[`chapters/`](chapters/)** — one folder per book chapter; each is self-contained (see any chapter's README).
- **[`ragkit/`](ragkit/)** — the importable library: `core · ingestion · retrieval · architectures · production · eval`.
- **[`capstone/`](capstone/)** — the complete assembled enterprise RAG reference system.
- **[`data/`](data/)** — sample corpora + golden sets that run out of the box.
- **[`docs/`](docs/)** — [SETUP](docs/SETUP.md) · [HOW-TO-USE](docs/HOW-TO-USE.md) · [PROVIDER-SWAP](docs/PROVIDER-SWAP.md) · dataset cards.

## License

Code: MIT (see [LICENSE](LICENSE)). Sample data: see [`data/README.md`](data/README.md) for per-dataset terms.
