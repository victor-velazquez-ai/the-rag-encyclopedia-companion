"""ragkit.architectures.graph — knowledge-graph RAG (Book Ch 9).

A graph is an expensive structure to build and maintain, so the only honest reason to build one is
that your traffic contains global-sensemaking or multi-hop questions flat retrieval provably cannot
answer. The systems below are points on a cost-versus-structure curve, not interchangeable rivals;
the facade defaults to the pragmatic choice and lets you escalate when measurement justifies it.

    light_rag.py    LightRAG (arXiv:2410.05779) — graph + dual-level retrieval, incremental updates;
                    the default once you have confirmed a graph-shaped problem on a living corpus.
    lazy_graph.py   LazyGraphRAG — co-occurrence graph at ~0.1% of GraphRAG's index cost; defers LLM
                    work to query time. Reach for it when index budget binds or the case is unproven.
    graph_rag.py    Microsoft GraphRAG (arXiv:2404.16130) — full entity/relation extraction,
                    hierarchical-Leiden communities, per-community reports; Global/Local/DRIFT query
                    modes. The heaviest option; reserve for bounded, slow-changing, global-first corpora.

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class GraphRAG:
#     """Facade over the graph-RAG variants. `GraphRAG.default()` → LightRAG (graph + incremental)."""
#     @classmethod
#     def default(cls) -> "GraphRAG": ...
#     @classmethod
#     def variant(cls, name: str) -> "GraphRAG":   # "lightrag" | "lazy" | "graphrag"
#         ...
#     def index(self, corpus) -> None:
#         """Build (or incrementally merge into) the knowledge graph for this corpus."""
#         ...
#     def query(self, question: str, mode: str = "auto") -> list:
#         """Answer over the graph; `mode` selects Global/Local/DRIFT for full GraphRAG."""
#         ...

__all__ = ["GraphRAG"]  # populated in Phase 2
