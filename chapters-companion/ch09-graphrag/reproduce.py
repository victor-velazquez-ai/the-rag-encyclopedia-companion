"""Chapter 9 - does a graph earn its cost? (offline local-query demo).

The chapter's thesis: a graph earns its keep only on global-sensemaking or multi-hop traffic that
flat retrieval cannot serve. This script builds a small knowledge graph from triples and runs a
local query plus a multi-hop neighborhood walk - the retrieval primitive a graph exists to give.

Fully OFFLINE: `from_triples` and `local_query` are pure (no model, no SDK). Index-time extraction
(`build`) and `global_query` need an API key; their call shape is shown in 01_graphrag.py.
"""

from __future__ import annotations

from professional_rag_kit.architectures.graph import GraphIndex

TRIPLES = [
    ("Ada Lovelace", "collaborated with", "Charles Babbage"),
    ("Charles Babbage", "designed", "Analytical Engine"),
    ("Ada Lovelace", "wrote notes on", "Analytical Engine"),
    ("Analytical Engine", "is a", "mechanical computer"),
    ("Alan Turing", "cited", "Ada Lovelace"),
    ("Alan Turing", "designed", "Universal Turing Machine"),
]


def main() -> None:
    graph = GraphIndex.from_triples(TRIPLES)
    print("entities:", graph.entities())
    print("edges:", len(graph.triples))

    print("\nLOCAL query (hops=1) - Ada Lovelace's neighborhood:")
    local = graph.local_query("Ada Lovelace", hops=1)
    print("  neighbors:", local["neighbors"])
    for s, r, o in local["triples"]:
        print(f"    {s} | {r} | {o}")

    print("\nMULTI-HOP (hops=2) - Alan Turing reaches the Analytical Engine via Ada Lovelace:")
    two = graph.local_query("Alan Turing", hops=2)
    print("  neighbors:", two["neighbors"])
    reached = "Analytical Engine" in two["neighbors"]
    print("  bridge entity 'Analytical Engine' reached:", reached)

    print("\nThis path (Turing -> Ada Lovelace -> Analytical Engine) is what flat retrieval cannot")
    print("assemble - the multi-hop case where a graph earns its cost.")


if __name__ == "__main__":
    main()
