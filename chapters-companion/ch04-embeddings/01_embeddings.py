# %% [markdown]
# # Chapter 4 - Embeddings: turning text into meaning-bearing vectors
#
# The single most consequential retrieval decision, and the one most often made by accident: you
# inherit a model's dimensionality, license, and **query/doc asymmetry** for the life of the index.
# This walkthrough demos the `Embedder` facade live against **OpenAI text-embedding-3-large** (the
# bring-your-own-key default; Anthropic has no embeddings API).
#
# **This notebook needs `OPENAI_API_KEY`.** The first cell guards: with no key, the live cells below
# are documented but not run. Nothing here is destructive.
#
# Production code: [`professional_rag_kit/ingestion/embedding`](../../professional_rag_kit/ingestion/embedding/__init__.py).
# Book section: Ch 4, "Model selection", "Instruction-tuned and asymmetric encoders", "Matryoshka".

# %%
import os

HAVE_KEY = bool(os.environ.get("OPENAI_API_KEY"))
if not HAVE_KEY:
    print("OPENAI_API_KEY is not set.")
    print("Set OPENAI_API_KEY to run the live cells below (each makes a real embeddings API call).")
    print("The narration still explains exactly what each call does.")
else:
    print("OPENAI_API_KEY found - live cells will call the embeddings API.")

# %%
from professional_rag_kit.ingestion import Embedder


def cosine(a, b):
    """Plain cosine similarity (the API returns L2-normalized vectors, but be explicit)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


# `Embedder.default()` -> OpenAI text-embedding-3-large, 3072 dims.
embedder = Embedder.default()
print("provider:", embedder.provider, "| model:", embedder.model, "| dims:", embedder.dims or 3072)

# %% [markdown]
# ## embed_query vs. embed_documents - the asymmetry surface
# The facade exposes *two* methods on purpose. Queries and documents must be encoded the same way the
# index was built. OpenAI is symmetric (the two calls behave identically), but Voyage and the self-host
# Qwen3 default use an instruction prefix on the **query side only** - get that wrong and recall drops
# silently with no error. Always route queries through `embed_query` and corpus text through
# `embed_documents` so a provider swap can't introduce a prefix mismatch.

# %%
QUERY = "How do I keep merged table cells intact when parsing?"
DOCS = [
    "Serialize HTML tables to Markdown so the row/column grid survives into retrieval.",
    "The recursive splitter packs sentences to a target token count with an overlap.",
]

if HAVE_KEY:
    qvec = embedder.embed_query(QUERY)
    dvecs = embedder.embed_documents(DOCS)
    print("query vector dims:", len(qvec))
    print("doc[0] cosine to query:", round(cosine(qvec, dvecs[0]), 3))
    print("doc[1] cosine to query:", round(cosine(qvec, dvecs[1]), 3))
    print("-> the table doc should score higher; it answers the question.")
else:
    print("(skipped - needs OPENAI_API_KEY) would embed 1 query + 2 docs and rank them.")

# %% [markdown]
# ## A paraphrase pair scores high; an unrelated pair scores low
# The whole point of a learned embedding: surface form differs, meaning matches. A paraphrase pair
# should land far above an unrelated pair in cosine space. This is the sanity check you run before
# trusting a new model on your data.

# %%
A = "The recursive chunker respects document structure."
B = "A structure-aware splitter cuts on paragraph and sentence boundaries."  # paraphrase of A
C = "Quantization shrinks each vector to one byte per dimension."            # unrelated

if HAVE_KEY:
    va, vb, vc = embedder.embed_documents([A, B, C])
    print("paraphrase  cos(A,B):", round(cosine(va, vb), 3))
    print("unrelated   cos(A,C):", round(cosine(va, vc), 3))
    assert cosine(va, vb) > cosine(va, vc), "paraphrase should beat the unrelated pair"
else:
    print("(skipped - needs OPENAI_API_KEY) would show cos(A,B) > cos(A,C).")

# %% [markdown]
# ## Matryoshka (MRL): truncate dimensions without re-embedding
# text-embedding-3-large supports MRL - pass `dims=` and the API returns a natively shortened,
# **re-normalized** vector (not a hand-sliced one, which would break cosine). Fewer dims means a
# smaller, cheaper index at a small, *measurable* quality cost. The chapter's trap: never truncate by
# hand without re-normalizing. Here the backend requests the shortened vector for you.

# %%
small = Embedder.from_provider("openai", dims=512)
print("MRL embedder dims request:", small.dims)

if HAVE_KEY:
    full_vec = embedder.embed_query(QUERY)        # 3072
    small_vec = small.embed_query(QUERY)          # 512, API re-normalized
    print("full dims:", len(full_vec), "| MRL-512 dims:", len(small_vec))
    print("MRL-512 is ~6x smaller per vector - measure nDCG@10 to price the quality you traded.")
else:
    print("(skipped - needs OPENAI_API_KEY) would compare 3072-dim vs 512-dim (MRL) vectors.")

# %% [markdown]
# ## The self-host verdict is one line away
# The book's shippable self-host pick is **Qwen3-Embedding-8B (Apache-2.0)** - a provider swap, not a
# rewrite. It carries the query-side instruction prefix the OpenAI default doesn't need, which is
# exactly why the facade keeps query/doc encoding separate. (`[selfhost]` extra: needs
# `sentence-transformers`; not run here.)

# %%
selfhost = Embedder.from_provider("qwen3")
print("self-host swap:", selfhost.provider, "->", selfhost.model)
print("same embed_query / embed_documents API; the prefix asymmetry is handled inside the backend.")

# %%
print("\nembeddings ready. Ch 5 indexes these vectors in a filterable, quantized store.")
