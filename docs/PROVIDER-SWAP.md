# Provider swap — one line, no rewrite

The companion runs on **bring-your-own-key cloud APIs** so every example works on a laptop with no
GPU. Each swappable component is built behind a factory that reads a `provider` name — the pipeline
code is identical, only the provider changes. This is how you reproduce the book's "managed vs.
self-host" comparisons and decide, on your own data, what each option buys you.

> **Book vs. companion.** The book's *production* verdict is to self-host open models (Qwen3
> embeddings, jina reranker) for licensing and cost at scale (Ch 4/7). The companion defaults to
> APIs for friction-free learning; self-hosting those open models is the documented swap below.

## The default stack (BYO key)

| Component | Default | Key | Swaps |
|---|---|---|---|
| Generation + LLM rerank | **Claude** `claude-opus-4-8` | `ANTHROPIC_API_KEY` | `openai` (GPT) |
| Embeddings | **OpenAI** `text-embedding-3-large` | `OPENAI_API_KEY` | `voyage` |
| Reranker | **LLM listwise** (reuses generation) | — | `cohere` |
| Vector store | **Qdrant** (local Docker) | none | — |

Anthropic has no embeddings endpoint, so the default needs both an Anthropic key (generation) and an
OpenAI key (embeddings). To run on **one** key, set both providers to `openai`.

## The pattern

```python
from ragkit.production.generation import GroundedGenerator

gen = GroundedGenerator.default()                 # Claude (claude-opus-4-8)
ans = gen.generate("What was Q3 revenue?", passages)
print(ans.text, ans.citations, ans.abstained)

# managed swap — one line, reads the key from .env:
gen = GroundedGenerator.from_provider("openai")   # GPT
```

Or declaratively, and never touch code:

```yaml
# configs/openai.yaml
generation: { provider: openai, model: gpt-4o }
embedding:  { provider: openai, model: text-embedding-3-large }
```

```bash
RAGKIT_CONFIG=configs/openai.yaml make reproduce
# or, no file at all:
GEN_PROVIDER=openai EMBED_PROVIDER=openai make reproduce
```

## Self-hosting the book's verdict picks

To run the all-open production stack the book recommends (no API keys, needs a GPU), install the
self-host extra and select the local providers — same pipeline, different config:

```bash
pip install -e ".[selfhost]"          # sentence-transformers, etc.
GEN_PROVIDER=vllm EMBED_PROVIDER=qwen3 RERANK_PROVIDER=jina make reproduce
```

The eval harness reports the API and self-host runs **on the same axes** (same golden set, same
metrics, plus the cost column) so the comparison is honest.
