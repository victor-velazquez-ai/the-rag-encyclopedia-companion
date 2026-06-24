# %% [markdown]
# # Chapter 14 — Retrieval metrics, and why you need all three
#
# Evaluation is the spine of the book: *measurement licenses complexity.* The first move is
# separating **retrieval** quality from **generation** quality — they're orthogonal, and a metric
# that conflates them is worse than none. This walkthrough runs the three retrieval metrics offline.
#
# Production code: [`ragkit/eval/metrics`](../../ragkit/eval/metrics.py).

# %%
from ragkit.eval.metrics import mrr, ndcg_at_k, recall_at_k

# A query's ground truth: which docs are relevant, and (for nDCG) how relevant (graded gains).
relevant = {"d1", "d3"}
gains = {"d1": 3.0, "d3": 1.0}  # d1 is highly relevant, d3 mildly

# Two systems return the same docs in different orders.
system_a = ["d1", "x1", "d3", "x2"]  # both relevant docs near the top
system_b = ["x1", "x2", "d3", "d1"]  # both relevant docs buried

# %% [markdown]
# ## Recall@k — the floor (order-agnostic)
# Did the relevant docs show up at all within the cutoff? Order doesn't matter, so A and B tie.

# %%
print("recall@4   A:", recall_at_k(system_a, relevant, 4), " B:", recall_at_k(system_b, relevant, 4))
print("recall@2   A:", recall_at_k(system_a, relevant, 2), " B:", recall_at_k(system_b, relevant, 2))

# %% [markdown]
# ## MRR — rewards getting *one* good hit early
# Reciprocal rank of the first relevant doc. A's first hit is at rank 1; B's first is at rank 3.

# %%
print("MRR  A:", round(mrr(system_a, relevant), 3), " B:", round(mrr(system_b, relevant), 3))
assert mrr(system_a, relevant) > mrr(system_b, relevant)

# %% [markdown]
# ## nDCG@k — graded relevance + full ordering
# The only one that knows d1 (gain 3) is worth more than d3 (gain 1) **and** rewards ranking it
# higher. This is the metric that separates A from B when recall ties.

# %%
print("nDCG@4  A:", round(ndcg_at_k(system_a, gains, 4), 3), " B:", round(ndcg_at_k(system_b, gains, 4), 3))

# %% [markdown]
# ## The lesson
# Same recall, very different nDCG/MRR. Report **retrieval** metrics (these) separately from
# **generation** metrics (faithfulness, answer correctness) so every regression is attributed to the
# right stage before you touch a component. The reproduction suite scores both halves on the golden set.

# %%
print("\nstage-wise attribution is the habit; end-to-end accuracy alone hides which half broke.")
