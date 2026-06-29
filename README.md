# The RAG Encyclopedia — Official Companion

The hands-on companion to **_The RAG Encyclopedia: Advanced Enterprise Retrieval-Augmented
Generation_** by **Victor Miguel Velazquez Espitia**.

The book teaches you *what* to build and *why*, and commits to a verdict on every technique.
This repository is where you **run it, measure it, and lift it into your own systems** — a
chapter-by-chapter learning lab and a clean, importable, production-grade library, tightly
cross-linked so you never hunt for anything.

> **Status: live.** The `professional_rag_kit` library and the assembled capstone are implemented and
> verified — **121 unit tests pass offline (no API key needed)**. The chapter walkthroughs run against
> the small sample corpus; add your API keys (below) to run the full cloud examples.

---

## The one rule that organizes this repo

**The chapter is the front door.** Reading Chapter 7? Open [`chapters-companion/ch07-reranking/`](chapters-companion/ch07-reranking/)
and *everything* for that chapter is right there — the runnable walkthroughs, the run commands,
the reproduction experiment, and a direct link to the production code that implements it. There
is exactly **one learning axis** (the book's chapters) and **one reuse axis** (the `professional_rag_kit`
library), and they are cross-linked at every step. No four-folder scavenger hunt.

```
the-rag-encyclopedia-repo/
├── chapters-companion/        ← THE LEARNING PATH — walk it in book order, hand by hand
├── professional_rag_kit/          ← THE REUSE SPINE — clean, importable, production-grade library
├── capstone-project/        ← the full enterprise RAG system, everything wired together
├── data/            ← small sample corpus + golden sets (tiny; the only cost is your API usage)
└── docs/            ← setup, how-to-use, provider-swap guide, dataset cards
```

---

## Two ways to use it

### 📖 "I'm reading the book"
Open the chapter folder, run the numbered notebooks top-to-bottom as you read. Each one narrates
a single technique and runs live against the sample data.

```bash
cp .env.example .env          # add your ANTHROPIC_API_KEY + OPENAI_API_KEY
docker compose up -d          # Qdrant, locally
make setup                    # install professional_rag_kit + deps, load sample data
make ch07                     # launch Chapter 7's walkthroughs
```

### 🛠️ "I'm building my own system"
`import professional_rag_kit` and lift the production components straight into your project. Read
[`capstone-project/`](capstone-project/) for the assembled reference system, and swap any provider
(Qwen3→Voyage, jina→Cohere) by editing **one line** of config — no rewrite.

```python
from professional_rag_kit.retrieval import HybridRetriever, Reranker
from professional_rag_kit.ingestion import Embedder

retriever = HybridRetriever.from_config("configs/default.yaml")   # BM25 + dense + RRF
hits = retriever.search("your query", top_k=50)
ranked = Reranker.default().rerank("your query", hits)[:5]
```

---

## The master map

Find anything in one look. Each chapter maps to a folder, a run command, the production module
that implements it, and the head-to-head experiment it reproduces.

| Book Ch | Folder | Run | Production code (`professional_rag_kit/`) | Reproduces (quality **+** cost) |
|---|---|---|---|---|
| 2 · Document Processing | [`chapters-companion/ch02-document-processing/`](chapters-companion/ch02-document-processing/) | `make ch02` | `ingestion/parsing/` | VLM vs. classical OCR vs. hybrid parsing |
| 3 · Chunking | [`chapters-companion/ch03-chunking/`](chapters-companion/ch03-chunking/) | `make ch03` | `ingestion/chunking/` | semantic vs. tuned ~200-tok recursive; contextual retrieval |
| 4 · Embeddings | [`chapters-companion/ch04-embeddings/`](chapters-companion/ch04-embeddings/) | `make ch04` | `ingestion/embedding/` | self-host Qwen3 vs. managed; int8/binary + rescore vs. float |
| 5 · Vector Stores | [`chapters-companion/ch05-vector-stores/`](chapters-companion/ch05-vector-stores/) | `make ch05` | `ingestion/indexing/` | HNSW vs. quantized recall; filtered-ANN recall collapse |
| 6 · Retrieval & Routing | [`chapters-companion/ch06-retrieval/`](chapters-companion/ch06-retrieval/) | `make ch06` | `retrieval/{hybrid,query,routing}/` | hybrid+RRF vs. dense-only; HyDE on strong vs. weak embedder |
| 7 · Reranking | [`chapters-companion/ch07-reranking/`](chapters-companion/ch07-reranking/) | `make ch07` | `retrieval/rerank/` | reranker on/off; listwise-LLM vs. distilled cross-encoder |
| 8 · Context Construction | [`chapters-companion/ch08-context-construction/`](chapters-companion/ch08-context-construction/) | `make ch08` | `retrieval/context/` | lost-in-the-middle ordering; compression vs. retrieve-less |
| 9 · GraphRAG | [`chapters-companion/ch09-graphrag/`](chapters-companion/ch09-graphrag/) | `make ch09` | `architectures/graph/` | GraphRAG vs. LightRAG vs. vector on local-fact traffic |
| 10 · Hierarchical & Adaptive | [`chapters-companion/ch10-adaptive-rag/`](chapters-companion/ch10-adaptive-rag/) | `make ch10` | `architectures/{hierarchical,adaptive}/` | RAPTOR collapsed-tree; Adaptive/CRAG/Self-RAG/FLARE |
| 11 · Agentic & Multi-System | [`chapters-companion/ch11-agentic-rag/`](chapters-companion/ch11-agentic-rag/) | `make ch11` | `architectures/agentic/` | single-shot vs. agentic multi-hop; entity resolution |
| 12 · Multimodal | [`chapters-companion/ch12-multimodal/`](chapters-companion/ch12-multimodal/) | `make ch12` | `architectures/multimodal/` | ColPali visual retrieval vs. OCR-then-embed |
| 13 · Grounded Generation | [`chapters-companion/ch13-grounded-generation/`](chapters-companion/ch13-grounded-generation/) | `make ch13` | `production/generation/` | CAD / Chain-of-Note grounding; span-level citation |
| 14 · Evaluation | [`chapters-companion/ch14-evaluation/`](chapters-companion/ch14-evaluation/) | `make ch14` | `eval/` | RAGAS metrics; LLM-judge bias + calibration |
| 15 · Enterprise Hardening | [`chapters-companion/ch15-enterprise-hardening/`](chapters-companion/ch15-enterprise-hardening/) | `make ch15` | `production/security/` | retrieval-time ACL trimming; indirect-injection defense |
| 16 · Performance & Observability | [`chapters-companion/ch16-performance-observability/`](chapters-companion/ch16-performance-observability/) | `make ch16` | `production/{serving,observability}/` | semantic-cache correctness; routing cost cuts |

> Chapter 1 (Introduction) is conceptual — no code. Start your setup in [`docs/SETUP.md`](docs/SETUP.md).

---

## The reference stack (bring your own API key — no GPU)

The companion runs on cloud APIs so every example works on a laptop. Each component is a one-line
provider swap. *(The book's **production** verdict is self-hosting open models — Qwen3, jina — for
licensing/cost at scale; that's the documented `[selfhost]` swap, see [PROVIDER-SWAP](docs/PROVIDER-SWAP.md).)*

| Role | Default (BYO key) | Key | Swap |
|---|---|---|---|
| Generation + LLM rerank | **Claude** `claude-opus-4-8` | `ANTHROPIC_API_KEY` | OpenAI GPT |
| Embeddings | **OpenAI** `text-embedding-3-large` | `OPENAI_API_KEY` | Voyage |
| Sparse leg | **BM25** (+ optional SPLADE-v3) | none | — |
| Vector store | **Qdrant** (local Docker) | none | — |

Anthropic has no embeddings API, so the default uses an Anthropic key (generation) **and** an OpenAI
key (embeddings); set both providers to `openai` to run on a single key. The self-host stack
(Qwen3 + jina + vLLM) needs no keys but a GPU. See [`docs/PROVIDER-SWAP.md`](docs/PROVIDER-SWAP.md).

---

## The evaluation harness — the book's spine, made runnable

`professional_rag_kit/eval/` plus each chapter's `reproduce.py` re-run the book's head-to-head comparisons on a
version-controlled golden set and print a **quality number and a cost number** — because the
book's argument is never quality alone. This is what makes every verdict in the book *falsifiable
on your own corpus*. Run the whole suite with `make reproduce`.

---

## Layout

- **[`chapters-companion/`](chapters-companion/)** — one folder per book chapter; each is self-contained (see any chapter's README).
- **[`professional_rag_kit/`](professional_rag_kit/)** — the importable library: `core · ingestion · retrieval · architectures · production · eval`.
- **[`capstone-project/`](capstone-project/)** — the complete assembled enterprise RAG reference system.
- **[`data/`](data/)** — sample corpora + golden sets that run out of the box.
- **[`docs/`](docs/)** — [SETUP](docs/SETUP.md) · [HOW-TO-USE](docs/HOW-TO-USE.md) · [PROVIDER-SWAP](docs/PROVIDER-SWAP.md) · dataset cards.

## License

Code: MIT (see [LICENSE](LICENSE)). Sample data: see [`data/README.md`](data/README.md) for per-dataset terms.
