# Capstone — the assembled enterprise RAG system

Every technique in the book, wired together into one coherent, production-shaped system. The
chapters teach the components in isolation; the capstone is where you see them **compose** — and
where a developer starts when building their own.

> 🚧 Phase 1: this is the spec. The runnable capstone lands in Phase 2.

## What it is
A complete RAG service over the sample enterprise corpus that exercises the book's verdict stack
end to end:

```
Ingestion   →  hybrid parse (Ch 2) → recursive+contextual chunking (Ch 3)
               → Qwen3 embeddings, int8+rescore (Ch 4) → Qdrant, filterable HNSW (Ch 5)
Retrieval   →  BM25+dense+RRF (Ch 6) → jina cross-encoder rerank (Ch 7)
               → lost-in-the-middle context assembly (Ch 8)
Routing     →  Adaptive-RAG complexity gate (Ch 10): no-retrieval / single-shot / agentic multi-hop (Ch 11)
Generation  →  grounded generation + span citation + abstention (Ch 13)
Hardening   →  retrieval-time ACL trimming + PII redaction + injection defense (Ch 15)
Serving     →  vLLM + prompt-cache discipline + semantic cache (Ch 16)
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
