# %% [markdown]
# # Chapter 10 - Adaptive RAG: route by complexity, retrieve at the right altitude
#
# Flat RAG retrieves the same way every time: always, once, blindly. This chapter's policies are
# guards over different decision points. Two run fully OFFLINE here:
#
# - **Adaptive-RAG routing** - the cost optimizer. Send a query down the cheapest path that can
#   answer it: no retrieval / single-step / multi-step. A PURE heuristic, no model.
# - **Parent-child (small-to-big)** - index small child chunks for precision, return the larger
#   parent for context. PURE - the structural map-up needs no model.
#
# Production code: [`professional_rag_kit/architectures/adaptive`](../../professional_rag_kit/architectures/adaptive/__init__.py)
# and [`professional_rag_kit/architectures/hierarchical`](../../professional_rag_kit/architectures/hierarchical/__init__.py).
# Book section: Ch 10 (complexity routing; small-to-big hierarchy).

# %%
from professional_rag_kit.architectures.adaptive import AdaptiveRAG
from professional_rag_kit.architectures.hierarchical import ParentChildIndex

# %% [markdown]
# ## The complexity router (PURE - the cost optimizer)
# `route(query)` is a dependency-free heuristic the chapter explicitly allows as the cheap default.
# A plain factoid the model knows ("what is the capital of France") needs no retrieval; a normal
# lookup gets one retrieve-then-generate pass; a compositional / multi-hop question takes the heavy
# multi-step path. Order matters: multi-hop cues win over no-retrieval cues.

# %%
adaptive = AdaptiveRAG()  # retriever/generator optional - we only exercise the pure router
EXAMPLES = [
    "What is the capital of France?",
    "Summarize our Q3 refund policy.",
    "Compare GraphRAG and LightRAG and explain how their index costs relate.",
]
for q in EXAMPLES:
    print(f"{adaptive.route(q):>13}  <-  {q}")

# %% [markdown]
# ## Why routing is the cost optimizer
# Retrieval has a price - latency, tokens, and (for multi-step) several round trips. Routing spends
# that budget only where it buys accuracy. The distribution below shows the gate at work: the easy
# factoid skips retrieval entirely, the lookup pays once, only the compositional query pays the
# multi-step cost.

# %%
from collections import Counter

dist = Counter(adaptive.route(q) for q in EXAMPLES)
for route_name in ("no_retrieval", "single", "multi"):
    print(f"  {route_name:>13}: {dist.get(route_name, 0)}")

# %% [markdown]
# ## Parent-child: index small, return big (PURE)
# Small chunks retrieve precisely; large chunks read well. Parent-child exploits that asymmetry:
# embed and search the small *child* chunks, but on a hit return the larger *parent* for context.
# We register two parents, each with two children.

# %%
index = ParentChildIndex()
index.add(
    "doc1",
    "Full section on refunds: eligibility, the thirty-day window, and how to request one.",
    [("doc1.c1", "Refunds are eligible within thirty days."),
     ("doc1.c2", "Request a refund from the orders page.")],
)
index.add(
    "doc2",
    "Full section on shipping: carriers, delivery windows, and tracking.",
    [("doc2.c1", "Standard shipping takes three to five days."),
     ("doc2.c2", "Track a package with the order number.")],
)
print("child doc1.c1 belongs to parent:", index.parent_of("doc1.c1"))

# %% [markdown]
# ## Auto-merging: children collapse up to one parent
# Say the first stage retrieves three child chunks - two of them from the same parent. `retrieve`
# maps each child up to its parent and de-duplicates, so the two siblings collapse into ONE coherent
# parent passage instead of scattered fragments. The generator sees clean context, not overlap.

# %%
retrieved_children = ["doc1.c1", "doc1.c2", "doc2.c1"]  # two hits share doc1
parents = index.retrieve(retrieved_children)
print(f"retrieved {len(retrieved_children)} children -> {len(parents)} parent passages:")
for p in parents:
    print(f"  - {p}")

# %% [markdown]
# ## The reflective policies ship as patterns over your frontier model
# CRAG (grade-then-correct), Self-RAG (support critique), and FLARE (retrieve on low-confidence
# spans) are control graphs over the model you already run - they need a key. The call shape:
#
# ```python
# # export ANTHROPIC_API_KEY=...
# rag = AdaptiveRAG(retriever=my_retriever, generator=my_generator)
# rag.crag(query)        # retrieve -> grade -> correct (web-fallback hook) -> generate
# rag.self_rag(query)    # draft -> critique support -> re-draft if unsupported
# rag.flare(query)       # generate sentence-by-sentence, retrieve on low-confidence spans
# ```
#
# The policy is the durable asset; the model is yours. Next: evaluate whether each policy earns its
# delta (Ch 16).
print("Offline: route() and ParentChildIndex run with no key.")
print("CRAG / Self-RAG / FLARE are patterns over your frontier model (need a key).")
