# Provider swap — one line, no rewrite

The book's promise (Appendix B): every managed alternative is a **config swap, never a rewrite.**
The pipeline code is identical; only the `provider` changes. This is how you reproduce the
"managed vs. self-host" comparisons and decide on your own data whether convenience is worth the bill.

## The pattern

Every swappable component is built behind a small factory that reads a provider name:

```python
from ragkit.ingestion import Embedder
from ragkit.retrieval import Reranker

# default — all-open, self-host
embedder = Embedder.default()                 # Qwen3-Embedding-8B (Apache-2.0)
reranker = Reranker.default()                 # jina-reranker-v3 (open)

# managed — one line each (reads the key from .env)
embedder = Embedder.from_provider("voyage")   # or "gemini"
reranker = Reranker.from_provider("cohere")
```

Or set it declaratively and never touch code:

```yaml
# configs/managed.yaml
embedding: { provider: voyage,  model: voyage-3.5 }
rerank:    { provider: cohere,  model: rerank-v4.0 }
```

```bash
RAGKIT_CONFIG=configs/managed.yaml make reproduce
```

## Supported providers (Phase 2)

| Component | Default (open) | Managed swaps | Key |
|---|---|---|---|
| Embeddings | Qwen3-Embedding-8B | `voyage`, `gemini` | `VOYAGE_API_KEY` / `GEMINI_API_KEY` |
| Reranker | jina-reranker-v3 | `cohere` | `COHERE_API_KEY` |
| Generator | vLLM (local) | any OpenAI-compatible | `OPENAI_API_KEY` + `OPENAI_BASE_URL` |

The eval harness reports the open and managed runs **on the same axes** (same golden set, same
metrics, plus the cost column) so the comparison is honest.
