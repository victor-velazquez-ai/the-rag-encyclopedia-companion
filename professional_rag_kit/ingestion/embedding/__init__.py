"""professional_rag_kit.ingestion.embedding — text (and pages) into meaning-bearing vectors (Book Ch 4).

The single most consequential retrieval decision, and the one most often made by accident: you
inherit a model's dimensionality, license, and query/doc asymmetry for the life of the index.

Bring-your-own-key default: **OpenAI ``text-embedding-3-large``** (Anthropic has no embeddings API).
Voyage is a one-line swap; the book's self-host verdict (Qwen3-Embedding-8B, Apache-2.0) is a
``[selfhost]`` backend. The facade designs out the two classic failure modes from the chapter:
- a *query/doc asymmetry mismatch* — encode queries and documents the same way the index was built
  (OpenAI is symmetric; Voyage uses an ``input_type``). Get this wrong and recall drops silently.
- *MRL truncation without re-normalization* — pass ``dims`` to truncate (OpenAI/Voyage support it);
  the backends here request the API's native shortened vector rather than truncating by hand.

SDK imports are lazy; this module imports without ``openai``/``voyageai`` installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from professional_rag_kit.core.config import ProviderRegistry, load_config


@dataclass
class Embedder:
    """Facade over the embedding providers. ``default()`` → OpenAI text-embedding-3-large."""

    provider: str = "openai"
    model: str = "text-embedding-3-large"
    dims: int | None = None  # MRL truncation (re-normalized by the API)

    @classmethod
    def default(cls) -> "Embedder":
        return cls()

    @classmethod
    def from_provider(cls, provider: str, *, model: str = "", dims: int | None = None) -> "Embedder":
        model = model or {
            "openai": "text-embedding-3-large",
            "voyage": "voyage-3-large",
            "qwen3": "Qwen/Qwen3-Embedding-8B",
        }.get(provider, "")
        return cls(provider=provider, model=model, dims=dims)

    @classmethod
    def from_config(cls, path: str | None = None) -> "Embedder":
        c = load_config(path).embedding
        return cls.from_provider(c.provider, model=c.model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        backend = ProviderRegistry.get("embedding", self.provider)
        return backend(self.model, list(texts), "document", self.dims)

    def embed_query(self, text: str) -> list[float]:
        backend = ProviderRegistry.get("embedding", self.provider)
        return backend(self.model, [text], "query", self.dims)[0]


# --- Backends ----------------------------------------------------------------
@ProviderRegistry.register("embedding", "openai")
def _openai_embed(model, texts, input_type, dims):  # OpenAI is symmetric; input_type unused
    import openai

    client = openai.OpenAI()
    kwargs = {"model": model or "text-embedding-3-large", "input": texts}
    if dims:
        kwargs["dimensions"] = dims  # native MRL shortening
    resp = client.embeddings.create(**kwargs)
    return [d.embedding for d in resp.data]


@ProviderRegistry.register("embedding", "voyage")
def _voyage_embed(model, texts, input_type, dims):
    import voyageai

    vo = voyageai.Client()
    r = vo.embed(texts, model=model or "voyage-3-large", input_type=input_type, output_dimension=dims)
    return r.embeddings


@ProviderRegistry.register("embedding", "qwen3")
def _qwen3_embed(model, texts, input_type, dims):  # self-host (needs `pip install -e ".[selfhost]"`)
    from sentence_transformers import SentenceTransformer

    # query/doc asymmetry: Qwen3 uses an instruction prefix on the QUERY side only.
    prompt = "Instruct: Retrieve relevant passages\nQuery: " if input_type == "query" else ""
    enc = SentenceTransformer(model or "Qwen/Qwen3-Embedding-8B")
    vecs = enc.encode([prompt + t for t in texts], normalize_embeddings=True)
    return [v.tolist() for v in vecs]


__all__ = ["Embedder"]
