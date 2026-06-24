# Setup

The default stack is **all-open and needs no API key.** Three commands.

## Requirements
- Python 3.11+
- Docker (for Qdrant)
- ~8 GB free RAM for the local models; a GPU is optional (the small open models run on CPU, slower)

## Quickstart
```bash
git clone <this-repo> && cd the-rag-encyclopedia-repo
docker compose up -d        # Qdrant at http://localhost:6333/dashboard
make setup                  # pip install -e ".[dev]" + load the sample data
make ch07                   # open a chapter's notebooks (any chNN)
```

## What `make setup` does
1. Installs `ragkit` (editable) plus the dev extras (Jupyter, ruff, pytest).
2. Runs `python -m data.prepare` to fetch/prepare the small sample corpus and golden sets.

## Optional extras
Install only what a Part needs (keeps the base light):
```bash
pip install -e ".[architectures]"   # graph RAG (Ch 9)
pip install -e ".[multimodal]"      # ColPali (Ch 12)
pip install -e ".[serving]"         # vLLM generator (Ch 16)
pip install -e ".[all]"             # everything
```

## Going managed (optional)
Copy `.env.example` to `.env` and set the keys for any managed provider you want to benchmark, then
select it per-component in `configs/*.yaml` or via the `*_PROVIDER` env vars. See
[PROVIDER-SWAP.md](PROVIDER-SWAP.md). You never need this to run the book's experiments.

## Troubleshooting
- **Qdrant connection refused** → `docker compose ps`; ensure port 6333 is free.
- **Model download slow on first run** → weights cache under `./models/`; subsequent runs are fast.
- **Out of memory** → use the smaller embedder (`Qwen3-Embedding-0.6B`) via `configs/small.yaml`.
