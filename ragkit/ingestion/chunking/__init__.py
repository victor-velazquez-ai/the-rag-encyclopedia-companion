"""ragkit.ingestion.chunking — the atomic unit of everything downstream (Book Ch 3).

The component that gets the least respect and causes the most quiet damage: a boundary in the
wrong place severs a fact from the context that makes it answerable, and no reranker repairs it.
The chapter's finding reframes the field — a *tuned ~200-token recursive splitter* is a genuinely
strong baseline that every expensive method must beat *on your data* (semantic chunking usually
loses on structured docs). So the default is cheap, and the LLM-budget methods are opt-in.

    recursive.py     structure-aware recursive split, ~200 tok, 10-20% overlap — the default/baseline
    semantic.py      embedding-breakpoint semantic chunking — niche: unstructured/stitched corpora only
    late.py          Late Chunking — embed whole doc first, then segment + mean-pool (needs LC encoder)
    contextual.py    Anthropic Contextual Retrieval — LLM-prepended context + BM25; the headline win
    propositional.py Dense-X atomic propositions — niche: out-of-domain factoid lookup
    hierarchical.py  small-to-big / sentence-window / auto-merging — search small, return large
    metadata.py      chunk-time metadata (doc_id, section_path, page, dates, tenant/ACL) — non-negotiable

One ``Chunker`` facade over all of them, so the default is one call and escalation is one argument.
The spend rule is the chapter's spine: Contextual Retrieval (Embeddings + BM25 cut top-20 failures
49%, →67% with rerank, ~$1.02/M tok) is the one index-time LLM method worth defaulting to past the
~200k-token RAG threshold; below that line, don't build RAG — put the corpus in the prompt.

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class Chunker:
#     """Facade over the chunking strategies. `Chunker.default()` → structure-aware recursive ~200 tok."""
#     @classmethod
#     def default(cls) -> "Chunker":
#         """Structure-aware recursive splitter, ~200 tokens, ~15% overlap, metadata attached."""
#         ...
#     @classmethod
#     def strategy(cls, name: str, **opts) -> "Chunker":
#         # "recursive" | "semantic" | "late" | "contextual" | "propositional" | "hierarchical"
#         ...
#     def split(self, doc) -> list:
#         """Split a parsed document into Chunk records, each carrying chunk-time metadata."""
#         ...

__all__ = ["Chunker"]  # populated in Phase 2
