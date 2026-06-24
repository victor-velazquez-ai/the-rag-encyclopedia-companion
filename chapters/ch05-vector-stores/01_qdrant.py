# %% [markdown]
# # Chapter 5 - Vector stores: where the vectors live and how you find them fast
#
# A bad index returns plausible neighbors that are merely *not the nearest ones* - and you never see
# it unless you measure recall against an exhaustive baseline. This walkthrough drives `ragkit`'s
# `VectorStore` (Qdrant) through its chapter defaults: **filterable HNSW + int8 quantization + a
# full-precision rescore**, an incremental `upsert`, and a `search` whose ACL filter rides *inside*
# the ANN traversal (the Ch 15 security tie-in) - never a naive post-filter that silently collapses
# recall.
#
# **This notebook needs a running Qdrant AND `OPENAI_API_KEY`** (to embed the chunks). The guard cell
# checks both; the live cells below it are documented but not run without them.
#
# Production code: [`ragkit/ingestion/indexing`](../../ragkit/ingestion/indexing/__init__.py).
# Book section: Ch 5, "In-index quantization" and "Filtered ANN: where naive systems silently break".

# %%
import os

HAVE_KEY = bool(os.environ.get("OPENAI_API_KEY"))
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

print("Prerequisites for the live cells:")
print("  1. Start Qdrant:  docker compose up -d   (or `make up`)")
print("  2. Set OPENAI_API_KEY to embed the chunks (text-embedding-3-large).")
print()
print("OPENAI_API_KEY set:", HAVE_KEY, "| QDRANT_URL:", QDRANT_URL)
READY = HAVE_KEY  # we still only run the live path when a key is present; Qdrant is checked at connect
if not READY:
    print("-> live cells are documented but will be skipped (no key).")

# %%
from ragkit.core.schema import Chunk
from ragkit.ingestion import Embedder, VectorStore

# A tiny multi-tenant corpus. `allowed_groups` is the retrieval-time ACL the Ch 15 security path
# depends on - it must ride inside the ANN filter, not be applied after the fact.
CHUNKS = [
    Chunk(id="c1", doc_id="handbook", text="Reimbursement requests are filed in the finance portal.",
          allowed_groups=["finance"]),
    Chunk(id="c2", doc_id="handbook", text="The on-call rotation is published every Monday.",
          allowed_groups=["engineering"]),
    Chunk(id="c3", doc_id="handbook", text="Expense approvals over $5k need a director sign-off.",
          allowed_groups=["finance"]),
]
QUERY = "how do I get an expense approved?"

# %% [markdown]
# ## Connect: filterable HNSW + int8 + rescore, in one call
# `VectorStore.connect` creates (or reuses) a Qdrant collection wired with the chapter's defaults:
# HNSW for the graph, **int8 scalar quantization** so the first pass is ~4x smaller in RAM, and the
# rescore step that recovers recall. `dim=3072` matches text-embedding-3-large.

# %%
if READY:
    store = VectorStore.connect(QDRANT_URL, collection="ch05_demo", dim=3072, quantization="int8")
    print("connected; collection ready (filterable HNSW + int8 quantization).")
else:
    print("(skipped - needs Qdrant + key) VectorStore.connect(...) would create the collection.")

# %% [markdown]
# ## Embed, then upsert - the index grows incrementally
# `upsert` requires each chunk to carry an `embedding`, so we run the Ch 4 embedder first, then insert.
# Upsert extends the existing HNSW graph rather than rebuilding it - that's what makes incremental
# freshness cheap (Ch 5, "the freshness problem"). The ACL `allowed_groups` is stored in the payload.

# %%
if READY:
    embedder = Embedder.default()
    vecs = embedder.embed_documents([c.text for c in CHUNKS])
    for c, v in zip(CHUNKS, vecs):
        c.embedding = v
    store.upsert(CHUNKS)
    print("upserted", len(CHUNKS), "embedded chunks into the HNSW graph.")
else:
    print("(skipped - needs Qdrant + key) would embed 3 chunks and upsert them.")

# %% [markdown]
# ## Search with an ACL filter that rides inside the ANN traversal
# `search(..., allowed_groups=["finance"])` builds a Qdrant `Filter` that is applied *during* the HNSW
# walk (filterable HNSW), not as a post-filter on an already-truncated result set. On a selective or
# security-bearing filter, a post-filter silently drops recall; the filterable path does not. The
# `finance` caller should see only the two finance chunks - never the engineering on-call chunk.

# %%
if READY:
    qvec = embedder.embed_query(QUERY)
    hits = store.search(qvec, top_k=5, allowed_groups=["finance"])
    for h in hits:
        print(round(h.score, 3), h.chunk.allowed_groups, "|", h.chunk.text[:48])
    leaked = [h for h in hits if "finance" not in h.chunk.allowed_groups]
    assert not leaked, "ACL filter leaked a non-finance chunk - this is a Ch 15 security failure"
    print("\nno cross-tenant leak: the engineering chunk never surfaced.")
else:
    print("(skipped - needs Qdrant + key) would search with allowed_groups=['finance'] and show no leak.")

# %% [markdown]
# ## The master pattern: quantize the first pass, rescore the shortlist
# `search` passes `rescore=True` with an `oversample` factor: Qdrant retrieves an *oversampled*
# candidate set using the cheap int8 vectors, then re-ranks that shortlist with full-precision
# vectors. This is what makes aggressive compression (up to 32x with binary/RaBitQ) safe - you pay
# float accuracy only on the few candidates that matter. Raise `oversample` to buy back recall.

# %%
if READY:
    tight = store.search(qvec, top_k=3, allowed_groups=["finance"], oversample=2)
    wide = store.search(qvec, top_k=3, allowed_groups=["finance"], oversample=8)
    print("oversample=2 top score:", round(tight[0].score, 3))
    print("oversample=8 top score:", round(wide[0].score, 3))
    print("-> wider oversampling rescues recall the quantized first pass can miss.")
else:
    print("(skipped - needs Qdrant + key) would compare oversample=2 vs oversample=8 recall.")

# %%
print("\nindex done. Ch 6 turns these hits into first-stage retrieval (hybrid + rerank).")
