# %% [markdown]
# # Chapter 8 — Context assembly: the zipper fold and MMR
#
# After retrieval and reranking you have a ranked shortlist. How you *order and trim* it into the
# prompt is a set of cheap, mostly model-free wins. Two of them, run offline here:
#
# - **Lost-in-the-middle fold** — models attend most to the start and end of a long context, least to
#   the middle. So place the strongest passages on those two peaks.
# - **MMR** — drop near-duplicates so the budget buys *new* evidence, not the same fact three times.
#
# Production code: [`ragkit/retrieval/context`](../../ragkit/retrieval/context/__init__.py).

# %%
from ragkit.retrieval.context import ContextBuilder, mmr, reorder_lost_in_middle

# A reranked shortlist, strongest first (rank 1 ... rank 5).
ranked = [
    "R1: the strongest, most on-topic passage",
    "R2: the second strongest passage",
    "R3: still relevant",
    "R4: weakly relevant",
    "R5: the weakest passage",
]

# %% [markdown]
# ## The zipper fold
# Rank 1 goes first, rank 2 goes **last**, rank 3 second, rank 4 second-to-last — zippering inward —
# so the two strongest passages sit on the attention peaks and the weakest lands in the dead middle.

# %%
folded = reorder_lost_in_middle(ranked)
for i, p in enumerate(folded):
    print(f"position {i}: {p}")

assert folded[0].startswith("R1")  # strongest on the front peak
assert folded[-1].startswith("R2")  # second strongest on the back peak
assert folded[len(folded) // 2].startswith("R5")  # weakest buried in the middle

# %% [markdown]
# ## MMR: relevance with diversity
# MMR greedily picks the doc maximizing `λ·sim(doc, query) − (1−λ)·max sim(doc, picked)`. With three
# docs — A and A' near-duplicates, B different — a diversity-leaning λ skips the redundant A'.

# %%
query_vec = [1.0, 0.0]
doc_vecs = [[1.0, 0.0], [0.98, 0.02], [0.7, 0.7]]  # A, A' (dup of A), B
labels = ["A", "A'(dup)", "B"]

picked_relevance = mmr(query_vec, doc_vecs, k=2, lambda_=0.7)  # relevance-dominant default
picked_diverse = mmr(query_vec, doc_vecs, k=2, lambda_=0.3)  # diversity-leaning
print("lambda=0.7 (default):  ", [labels[i] for i in picked_relevance])
print("lambda=0.3 (diverse):  ", [labels[i] for i in picked_diverse])

# %% [markdown]
# ## Putting it together
# `ContextBuilder` folds, then trims to a token budget. (MMR is opt-in — pass `mmr=λ` plus vectors.)

# %%
ctx = ContextBuilder(token_budget=40).build("the query", ranked)
print(ctx)
print("\nNext: hand this context to the grounded generator (Ch 13).")
