"""ragkit.ingestion — Part I: turning documents into searchable vectors (Book Ch 2–5).

The four stages every RAG corpus passes through before a query ever runs. Each is a committed
default from its chapter, with the expensive alternatives behind one interface so you escalate
only when measurement says it pays.

    parsing/    hybrid VLM/OCR → faithful structured text; tables as HTML; chart surrogates  (Ch 2)
    chunking/   structure-aware recursive-200 default; semantic/late/contextual/parent-child (Ch 3)
    embedding/  Qwen3 default; instruction prefixes; MRL truncation; int8/binary + rescore    (Ch 4)
    indexing/   Qdrant; filterable HNSW; in-index quantization; tombstone/compaction          (Ch 5)

The chapter ordering is the data flow: parse → chunk → embed → index. A failure upstream is
unrecoverable downstream (a number the parser drops, a fact a bad boundary severs, a prefix
mismatch that silently sinks recall), which is why each stage commits to a defensible default.

Phase-1 scaffold. Phase 2 exports the top-level conveniences: Chunker, Embedder, VectorStore.
"""

# Parsing is exposed through a `parse()` entry point rather than a single class (Ch 2).
from ragkit.ingestion.chunking import Chunker
from ragkit.ingestion.embedding import Embedder
from ragkit.ingestion.indexing import VectorStore

__all__ = ["Chunker", "Embedder", "VectorStore"]
