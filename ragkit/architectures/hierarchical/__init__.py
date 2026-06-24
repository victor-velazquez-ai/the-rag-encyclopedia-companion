"""ragkit.architectures.hierarchical — retrieval at the right altitude (Book Ch 10).

Flat retrieval returns equal-sized chunks; hierarchy lets retrieval return a unit sized to the
question. The cheap default does this with chunk-size nesting and no LLM summarization; the
specialist builds an LLM summary tree for long-document synthesis and pays a batch build bill.

    parent_child.py   parent-child / small-to-big / auto-merging — index small for precision,
                      return big for context. No summarization, no summary drift, trivial
                      incremental updates. The default hierarchy in every RAG system.
    raptor.py         RAPTOR (arXiv:2401.18059) — recursive UMAP + soft-GMM(BIC) clustering and
                      LLM summarization into a tree, queried in collapsed-tree mode (flatten every
                      layer into one flat top-k; the paper's mode, beats tree-traversal). The
                      specialist for long-document synthesis; pays an index-time LLM bill + staleness.

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class HierarchicalRetriever:
#     """Facade over hierarchy modes. `HierarchicalRetriever.default()` → parent-child / small-to-big."""
#     @classmethod
#     def default(cls) -> "HierarchicalRetriever": ...
#     @classmethod
#     def raptor(cls) -> "HierarchicalRetriever":   # collapsed-tree retrieval (the recommended mode)
#         ...
#     def index(self, documents) -> None:
#         """Build the hierarchy (chunk-size nesting, or the RAPTOR summary tree)."""
#         ...
#     def search(self, query: str, top_k: int = 10) -> list:
#         """Retrieve at the altitude that best answers the query (parent return / collapsed pool)."""
#         ...

__all__ = ["HierarchicalRetriever"]  # populated in Phase 2
