# %% [markdown]
# # Chapter 9 - GraphRAG: when a graph earns its cost
#
# A graph is an expensive structure to build and maintain, so the only honest reason to build one is
# that your traffic contains questions flat retrieval provably cannot answer: *global sensemaking*
# (a synthesis no single chunk contains) or *multi-hop* (the answer is a path through relationships
# that similarity search cannot assemble). This walkthrough runs OFFLINE - we build a small graph
# from hand-written triples and walk a neighborhood. The pure pieces (`from_triples`, `local_query`)
# need no model.
#
# Production code: [`ragkit/architectures/graph`](../../ragkit/architectures/graph/__init__.py).
# Book section: Ch 9 (extract triples -> local / global query).

# %%
from ragkit.architectures.graph import GraphIndex

# %% [markdown]
# ## Build a knowledge graph from triples (PURE - no model)
# In production, an LLM extracts `(subject, relation, object)` triples from your documents. Here we
# supply them directly so the build is deterministic and offline. `from_triples` deduplicates edges,
# resolves entity names case-insensitively, and keeps an undirected adjacency view for traversal.

# %%
TRIPLES = [
    ("Ada Lovelace", "collaborated with", "Charles Babbage"),
    ("Charles Babbage", "designed", "Analytical Engine"),
    ("Ada Lovelace", "wrote notes on", "Analytical Engine"),
    ("Analytical Engine", "is a", "mechanical computer"),
    ("Alan Turing", "cited", "Ada Lovelace"),
    ("Alan Turing", "designed", "Universal Turing Machine"),
]
graph = GraphIndex.from_triples(TRIPLES)
print("entities:", graph.entities())
print("edge count:", len(graph.triples))

# %% [markdown]
# ## Local query: walk an entity's neighborhood
# A *local* query is the retrieval primitive a graph exists to give: seed on an entity and return its
# neighborhood - the connected facts a generator needs to answer "what is connected to X". With
# `hops=1` we get Ada Lovelace's immediate relationships.

# %%
local = graph.local_query("Ada Lovelace", hops=1)
print("resolved entity:", local["entity"])
print("neighbors:", local["neighbors"])
print("local context triples:")
for s, r, o in local["triples"]:
    print(f"  {s} | {r} | {o}")

# %% [markdown]
# ## Multi-hop: where flat retrieval breaks
# "How is Alan Turing connected to the Analytical Engine?" has no single chunk that states it - the
# answer is a *path* (Turing -> cited -> Ada Lovelace -> wrote notes on -> Analytical Engine).
# Expanding the neighborhood to `hops=2` from Turing pulls in the bridge entity and surfaces that
# path. This is the multi-hop case a graph answers and similarity search cannot.

# %%
two_hop = graph.local_query("Alan Turing", hops=2)
print("Turing 2-hop neighbors:", two_hop["neighbors"])
print("subgraph triples (the path is in here):")
for s, r, o in two_hop["triples"]:
    print(f"  {s} | {r} | {o}")
assert "Analytical Engine" in two_hop["neighbors"]  # the bridge was reached

# %% [markdown]
# ## Index-time extraction and global queries need a key
# `build(docs)` runs the LLM once per document to extract triples, and `global_query(question)` does
# the community/summary-style synthesis answer over the whole graph. Both go through the generation
# backend, so they need an API key. The call shape (guarded so this cell is a no-op offline):

# %%
import os

if os.environ.get("ANTHROPIC_API_KEY"):
    docs = ["Ada Lovelace collaborated with Charles Babbage on the Analytical Engine."]
    built = GraphIndex.build(docs)  # one LLM extraction call per doc
    print("extracted triples:", built.triples)
    print(built.global_query("Who worked on early computing and how are they connected?"))
else:
    print("(skipped) set ANTHROPIC_API_KEY to run LLM extraction + global_query.")
    print("Offline, from_triples + local_query give you the graph and its neighborhoods.")

# %% [markdown]
# ## When a graph earns its keep
# Build a graph only when a real fraction of your traffic needs global sensemaking or multi-hop
# answers. If your traffic is dominated by local-fact lookups, a flat hybrid index (Ch 6) is cheaper
# and just as good. Microsoft GraphRAG, LightRAG, and LazyGraphRAG are points on a
# cost-versus-structure curve over this same skeleton.
print("\nNext: route easy queries away from the graph (Ch 10 adaptive RAG).")
