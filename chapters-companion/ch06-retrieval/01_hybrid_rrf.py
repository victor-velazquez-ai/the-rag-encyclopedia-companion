# %% [markdown]
# # Chapter 6 — Hybrid retrieval and Reciprocal Rank Fusion
#
# Sparse (BM25) and dense retrieval fail on **opposite** inputs: dense blurs exact identifiers and
# rare entities; sparse misses paraphrase. Run both and fuse their *ranked lists* with RRF — ranks
# are scale-free, so RRF sidesteps the unbounded-BM25-vs-bounded-cosine mismatch that makes linear
# score-blending fragile. This runs offline (BM25 + a stand-in dense ranking); the real dense leg
# uses the Ch 4 embedder + Qdrant.
#
# Production code: [`professional_rag_kit/retrieval/hybrid`](../../professional_rag_kit/retrieval/hybrid/__init__.py).

# %%
from professional_rag_kit.retrieval.hybrid import BM25, rrf_fuse

CORPUS = [
    ("d1", "The connection failed with error code ERR_CONN_4021 during handshake."),
    ("d2", "An automobile is a wheeled motor vehicle used for transportation."),
    ("d3", "Cars remain the most common form of personal transport worldwide."),
    ("d4", "Refunds are issued within thirty days of the original purchase date."),
]

# %% [markdown]
# ## BM25 nails the exact identifier...
# A query containing the literal error code lands `d1` at the top — exact-match is BM25's strength,
# and exactly what a dense vector tends to blur.

# %%
bm = BM25(CORPUS)
print("BM25 'ERR_CONN_4021 handshake':", bm.rank("ERR_CONN_4021 handshake", top_k=4))

# %% [markdown]
# ## ...and misses the paraphrase
# "vehicle for getting around" shares no literal terms with d2/d3, so BM25 returns nothing — the
# vocabulary-mismatch blind spot. A dense retriever would catch it. (Here we stand in a dense
# ranking by hand; in the repo it comes from `Embedder` + `VectorStore`.)

# %%
print("BM25 'vehicle for getting around':", bm.rank("vehicle for getting around", top_k=4) or "(nothing)")
dense_ranking = ["d3", "d2", "d4", "d1"]  # what a semantic retriever would return for that query
print("dense (semantic) ranking:        ", dense_ranking)

# %% [markdown]
# ## RRF recovers what either leg alone would miss
# Fuse the two ranked lists. A doc ranked highly by *either* leg surfaces — the fused set covers a
# failure class neither method handles alone. `k=60` is the paper's empirical default; sweep it on
# your own data.

# %%
sparse_ranking = bm.rank("automobile transport", top_k=4)  # a query BM25 *can* serve
fused = rrf_fuse([sparse_ranking, dense_ranking], k=60)
for doc_id, score in fused:
    print(f"{doc_id}  fused={score:.5f}")

# %% [markdown]
# ## The same thing, end to end
# `HybridRetriever.from_corpus(...)` wires BM25 (and, when you pass an `Embedder` + `VectorStore`,
# the dense leg) and fuses for you.

# %%
from professional_rag_kit.retrieval.hybrid import HybridRetriever

hr = HybridRetriever.from_corpus(CORPUS)  # sparse-only here; add embedder+store for full hybrid
print("\nhybrid 'refunds thirty days':", hr.search("refunds thirty days", top_k=3))
print("\nNext: rerank the fused candidates (Ch 7), then assemble context (Ch 8).")
