# Changelog — The RAG Encyclopedia companion

## Phase 1 — navigable scaffold (in progress)
- Repository structure established around the "chapter is the front door" principle.
- Root `README.md` with the master-map table (chapter → folder → run → `ragkit` module → reproduces).
- Infra: `pyproject.toml` (ragkit + extras), `Makefile` (`make chNN` / `make reproduce`),
  `docker-compose.yml` (Qdrant), `.env.example`, `.gitignore`.
- Docs: `SETUP`, `HOW-TO-USE` (the two journeys + folder anatomy), `PROVIDER-SWAP`.
- `ragkit/` package skeleton: `core` (chunk schema, config/providers) + Part modules; exemplar
  `retrieval/rerank` stubs.
- Exemplar chapter folder `ch07-reranking/` (the per-chapter README template).
- Remaining in Phase 1: the other 14 chapter READMEs + full `ragkit` submodule stubs + `capstone/`
  and `data/` specs.

## Phase 2 — implementation (in progress)
- **Stack pivot: bring-your-own-key cloud APIs** (no GPU). Default generation = Claude
  `claude-opus-4-8` (Anthropic SDK); embeddings = OpenAI `text-embedding-3-large` (Anthropic has no
  embeddings API); reranking = LLM-listwise. The book's open self-host stack (Qwen3/jina/vLLM) is the
  `[selfhost]` extra + a provider swap. Updated README, `.env.example`, SETUP, PROVIDER-SWAP, pyproject.
- **`ragkit.core` implemented + tested**: `Chunk` (pydantic, with `with_context` for Contextual
  Retrieval), `Config`/`load_config` (YAML + env-override provider selection), `ProviderRegistry`
  (the one-line swap mechanism).
- **`ragkit.production.generation` implemented**: `GroundedGenerator` (Ch 13) — context-faithful
  system prompt with span citation + abstention; Claude default, OpenAI swap; lazy SDK imports.
- **Pure spine components implemented + tested**: `retrieval/hybrid/fusion.rrf_fuse` (RRF, Ch 6) and
  `eval/metrics` (recall@k, MRR, nDCG@k, Ch 14).
- **Ingestion + retrieval stack implemented**:
  - `ingestion/chunking` — `Chunker` recursive ~200-tok structure-aware baseline (Ch 3), pure + tested.
  - `ingestion/embedding` — `Embedder` (OpenAI default, Voyage/Qwen3 swaps; query/doc asymmetry; MRL), lazy.
  - `ingestion/indexing` — `VectorStore` over Qdrant (filterable HNSW, int8+rescore, ACL filter), lazy.
  - `retrieval/hybrid` — pure `BM25` + `HybridRetriever` (BM25 + dense, RRF-fused).
  - `retrieval/rerank` — `Reranker` LLM-listwise default (reuses generation provider) + Cohere/jina swaps.
  - `retrieval/context` — `ContextBuilder` lost-in-the-middle zipper fold + MMR (Ch 8), pure + tested.
  - Package `__init__`s export the conveniences (`from ragkit.retrieval import HybridRetriever`, ...).
- **37 unit tests pass**; verified end-to-end on the pure path (chunk → BM25 → context fold) with no key.
- **First per-chapter walkthroughs (notebooks) — all offline, no API key:**
  - `tools/py2nb.py` — pure-Python percent-script → `.ipynb` converter (notebooks are generated from
    runnable scripts, so code + notebook never drift).
  - Ch 3 chunking, Ch 6 hybrid+RRF, Ch 8 lost-in-the-middle + MMR, Ch 14 retrieval metrics — each a
    paired `.py` (verified by execution) + `.ipynb`, narrating one technique against `ragkit`.
- **Eval harness + runnable reproduction (Ch 14), all offline:**
  - `ragkit.eval`: `GoldenSet` (jsonl loader), `Harness` (scores a retriever's nDCG@k/Recall@k/MRR over
    the golden set), `Scorecard`. Pure + tested.
  - Committed sample data: `data/corpus_small/docs.jsonl` (8 docs) + `data/golden/qa.jsonl` (8 queries);
    `data/prepare.py` verifies them (no download). 
  - `ragkit.eval.suite` (`make reproduce`): body-only BM25 vs BM25 body+title RRF fusion on the golden
    set — a real measured delta (+0.033 nDCG@5; fusion recovers a title-only-vocabulary query) with a
    cost column, no API key. `chapters/ch14-evaluation/reproduce.py` runs it.
  - **42 unit tests pass.**
- Remaining: the rest of the notebooks (incl. key-requiring Ch 4/7/13), parsing (Ch 2), query
  transforms/routing (Ch 6), architectures (Ch 9–12), security/serving/observability (Ch 15/16),
  keyed reproductions, and `capstone/`.
