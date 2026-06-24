# Capstone — the assembled enterprise RAG system

Every technique in the book, wired together into one coherent, production-shaped system. The
chapters teach the components in isolation; the capstone is where you see them **compose** — and
where a developer starts when building their own.

## Run it now

```bash
python -m capstone.app.run                  # answers demo questions over the sample corpus
python -m capstone.app.run "your question"  # ask your own
```

With **`ANTHROPIC_API_KEY`** (or `OPENAI_API_KEY`) set, answers are synthesized and cited by
Claude/GPT through the grounded generator. With **no key**, the pipeline falls back to an offline
extractive answerer so the whole flow still runs end-to-end — the retrieval → fold path is identical
either way. It abstains ("I don't have enough information…") when the corpus doesn't support an answer.

The runnable core ([`app/pipeline.py`](app/pipeline.py)) wires the verdict stack:
**hybrid retrieve (BM25 + dense, RRF) → rerank → lost-in-the-middle fold → grounded generate**, with
the dense leg (Embedder + Qdrant) and LLM reranker as opt-in components. The full enterprise diagram
below is the target; routing, hardening, serving, and observability layer on next.

## The full target stack
A complete RAG service over the sample enterprise corpus, exercising the book's verdict stack:

```
Ingestion   →  hybrid parse (Ch 2) → recursive+contextual chunking (Ch 3)
               → embeddings, int8+rescore (Ch 4) → Qdrant, filterable HNSW (Ch 5)
Retrieval   →  BM25+dense+RRF (Ch 6) → rerank (Ch 7)
               → lost-in-the-middle context assembly (Ch 8)          [✅ runnable today]
Routing     →  Adaptive-RAG complexity gate (Ch 10): no-retrieval / single-shot / agentic multi-hop (Ch 11)
Generation  →  grounded generation + span citation + abstention (Ch 13)   [✅ runnable today]
Hardening   →  retrieval-time ACL trimming + PII redaction + injection defense (Ch 15)
Serving     →  prompt-cache discipline + semantic cache (Ch 16)
Eval/Ops    →  golden-set CI gate (Ch 14) + OTel/OpenInference tracing + drift (Ch 16)
```

## How it maps back
Every stage links to the chapter that argues it and the `ragkit` module that implements it, so the
capstone doubles as a worked example of using the library. It is the answer to "okay, I read all 16
chapters — what does the whole thing look like assembled?"

## Layout (Phase 2)
```
capstone/
├── app/            the service (ingest → retrieve → route → generate → serve)
├── configs/        the verdict-stack config (open default) + a managed variant
├── eval/           the capstone's own golden set + CI gate
└── README.md
```
