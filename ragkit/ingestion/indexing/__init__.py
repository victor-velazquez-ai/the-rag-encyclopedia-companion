"""ragkit.ingestion.indexing — where the vectors live and how you find them fast (Book Ch 5).

The least glamorous component and the one whose failures are most silent: a bad index returns
plausible neighbors that are merely *not the nearest ones*, and you never see it unless you measure
recall against an exhaustive baseline. The chapter standardizes on *Qdrant* precisely because it
exposes the mechanisms that matter — filterable HNSW, in-index quantization, tunable search — rather
than hiding them behind one "search" call.

    store.py        Qdrant setup + collection config; HNSW (m 16-32, ef_construct 128-256) default
    ann.py          the four ANN families: HNSW (in-RAM default) · IVF-PQ · ScaNN · DiskANN (out-of-RAM)
    filtered.py     filterable HNSW — payload-aware edges + brute-force-below-threshold (never naive post-filter)
    quantize.py     in-index quantization: int8/SQ8 by default; RaBitQ (error-bounded 1-bit, 32×) at scale
    freshness.py    incremental insert (cheap) vs. delete (tombstone + compaction); compaction scheduling

One ``VectorStore`` facade over Qdrant. The chapter's load-bearing rules: choose the index by your
*binding constraint* (fits in RAM → HNSW; out of RAM, one node → DiskANN; memory-bound at scale →
RaBitQ); treat *filtered ANN* as a first-class index requirement — naive post-filtering silently
collapses recall on selective or security-bearing filters; and obey the master pattern under all of
it — *quantize for the first pass, rescore the shortlist with full precision* (always oversample for
1-bit). Design for the delete, not just the insert: tombstones are cheap now, costly later.

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class VectorStore:
#     """Facade over Qdrant. `VectorStore.default()` → filterable HNSW + int8/SQ8 + float rescore."""
#     @classmethod
#     def default(cls) -> "VectorStore":
#         """Qdrant collection: HNSW (m=16, ef_construct=128), int8 quantization, rescore on."""
#         ...
#     @classmethod
#     def connect(cls, url: str, *, index: str = "hnsw", quantization: str = "int8") -> "VectorStore":
#         # index: "hnsw" | "diskann" ; quantization: "none" | "int8" | "binary" | "rabitq"
#         ...
#     def upsert(self, chunks: list) -> None: ...                     # incremental insert into the graph
#     def search(self, vector, top_k: int = 10, *, filter=None, oversample: int = 4) -> list:
#         """Filter-aware ANN: payload-aware HNSW, oversampled first pass, full-precision rescore."""
#         ...

__all__ = ["VectorStore"]  # populated in Phase 2
