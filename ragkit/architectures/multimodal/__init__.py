"""ragkit.architectures.multimodal — retrieve over pixels, not a lossy text projection (Book Ch 12).

One decision applied at every granularity: do you retrieve over a lossy text projection of a visual
artifact, or over the artifact itself? On visually-structured documents the answer is decisively the
artifact — but visual fidelity is bought with a multi-vector index, so it is paid for only when the
meaning is in the layout, not only in the words.

    colpali.py    ColPali / ColQwen2.5 (arXiv:2407.01449) — render page → VLM per-patch vectors →
                  ColBERT-style MaxSim late interaction; no OCR, no layout step, no chunker. Prefer
                  the ColQwen2.5 backbone; apply *token pooling* from day one (−66.7% vectors,
                  −2.2% quality) to blunt the ~30× storage cost. Requires a multimodal generator.
    unified.py    CLIP / SigLIP-style single-vector joint space — one pooled vector, cheap any-to-any
                  dot-product match. For whole-image semantic match, not region-precise document retrieval.
    objects.py    native multimodal embedding of individual charts / tables / figures (beats
                  image-to-text summaries by +13% mAP@5 on number-dense objects); keep a reference
                  to the original image regardless.

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class VisualRetriever:
#     """Visual document retrieval. `VisualRetriever.default()` → ColQwen2.5 + token pooling."""
#     @classmethod
#     def default(cls) -> "VisualRetriever": ...
#     @classmethod
#     def backbone(cls, name: str) -> "VisualRetriever":   # "colqwen2.5" | "colpali" | "colsmol"
#         ...
#     def index(self, pages, token_pooling: bool = True) -> None:
#         """Render each page, emit per-patch vectors, pool, and store for late-interaction search."""
#         ...
#     def search(self, query: str, top_k: int = 5) -> list:
#         """MaxSim the query tokens against page patches; return top page images."""
#         ...

__all__ = ["VisualRetriever"]  # populated in Phase 2
