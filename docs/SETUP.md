# Setup

The default stack is **bring-your-own-key cloud APIs** (no GPU). Four commands.

## Requirements
- Python 3.11+
- Docker (for Qdrant)
- An **Anthropic API key** (generation) and an **OpenAI API key** (embeddings). One OpenAI key
  suffices if you set both providers to `openai` — see [PROVIDER-SWAP](PROVIDER-SWAP.md).

## Quickstart
```bash
git clone <this-repo> && cd the-rag-encyclopedia-repo
cp .env.example .env        # add ANTHROPIC_API_KEY + OPENAI_API_KEY
docker compose up -d        # Qdrant at http://localhost:6333/dashboard
make setup                  # pip install -e ".[dev]" + load the sample data
make ch07                   # open a chapter's notebooks (any chNN)
```

## What `make setup` does
1. Installs `professional_rag_kit` (editable) plus the dev extras (Jupyter, ruff, pytest).
2. Runs `python -m data.prepare` to fetch/prepare the small sample corpus and golden sets.

The library code is testable with no key or network: `pip install -e ".[dev]" && pytest`.

## Optional extras
Install only what you need (keeps the base light):
```bash
pip install -e ".[managed]"         # Voyage embeddings / Cohere rerank swaps
pip install -e ".[selfhost]"        # the book's open production stack (Qwen3/jina/vLLM — needs a GPU)
pip install -e ".[architectures]"   # graph RAG (Ch 9)
pip install -e ".[multimodal]"      # ColPali (Ch 12)
pip install -e ".[all]"             # everything
```

## Swapping providers
Flip a provider with an env var or `configs/*.yaml` — no code change. `GEN_PROVIDER=openai`,
`EMBED_PROVIDER=voyage`, etc. To run the all-open self-host stack with no API keys (needs a GPU):
`pip install -e ".[selfhost]"` then `GEN_PROVIDER=vllm EMBED_PROVIDER=qwen3`. See
[PROVIDER-SWAP.md](PROVIDER-SWAP.md).

## Troubleshooting
- **Qdrant connection refused** → `docker compose ps`; ensure port 6333 is free.
- **`AuthenticationError`** → check `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in your `.env`.
- **One key only** → set `GEN_PROVIDER=openai EMBED_PROVIDER=openai` to run everything on OpenAI.
