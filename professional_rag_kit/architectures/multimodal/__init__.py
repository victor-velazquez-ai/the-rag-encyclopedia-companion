"""professional_rag_kit.architectures.multimodal — retrieve over pixels, not a lossy text projection (Book Ch 12).

One decision applied at every granularity: do you retrieve over a lossy text projection of a visual
artifact, or over the artifact itself? On visually-structured documents the answer is decisively the
artifact — but visual fidelity is bought with a multi-vector index, so it is paid for only when the
meaning is in the layout, not only in the words.

This module surfaces ``VisualRetriever`` — ColPali/ColQwen-style *visual document retrieval*:

    The thesis (Ch 12): render the page to an image and **embed the page image directly** — no OCR,
    no layout detection, no chunker. A vision-language model emits *one vector per patch*; queries
    encode to per-token vectors; scoring is ColBERT-style **MaxSim late interaction** (each query
    token matched to its best page patch). That deletes the entire brittle parse-and-chunk front end
    of Chapters 2–3, and on ViDoRe beats the strongest OCR-then-embed text pipeline (81.3 vs 67.0
    nDCG@5) while indexing ~18× faster.

    The price is **storage**: per-page multi-vectors run ~30× a single text embedding, and the index
    needs a *late-interaction* search path (the multi-vector / MaxSim storage tradeoff is Chapter 4's
    material). Apply **token pooling** from day one (−66.7% vectors for −2.2% quality) and require a
    multimodal generator downstream, or you have only deferred extraction.

The embedding heavy-lifting is `colpali-engine` (the optional ``[multimodal]`` extra). It is imported
lazily with a clear install message if absent, so this module imports with no heavy deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Backbone → the colpali-engine model/processor pair to load (Ch 12 model zoo; prefer ColQwen2.5).
_BACKBONES: dict[str, str] = {
    "colqwen2.5": "vidore/colqwen2.5-v0.2",
    "colqwen2": "vidore/colqwen2-v1.0",
    "colpali": "vidore/colpali-v1.3",
    "colsmol": "vidore/colSmol-256M",  # edge / low-memory footprint
}

_INSTALL_HINT = (
    "VisualRetriever needs the optional multimodal stack. Install it with:\n"
    '    pip install -e ".[multimodal]"   (colpali-engine + pillow)\n'
    "and run on a GPU-backed host — page encoding is a VLM forward pass."
)


@dataclass
class VisualRetriever:
    """ColPali/ColQwen-style visual document retrieval (Ch 12): embed the page image, skip OCR.

    A thin wrapper over ``colpali-engine``. Pages are rendered to images and encoded to per-patch
    multi-vectors; queries encode to per-token vectors; scoring is MaxSim late interaction. ``index``
    stores page multi-vectors (with **token pooling** by default to blunt the ~30× storage cost);
    ``search`` MaxSims the query tokens against stored page patches and returns the top page images.

    Everything that touches the model is **lazy** — construction is free and importing this module
    pulls in no heavy deps. The colpali-engine import is deferred to :meth:`_load`, with a clear
    install message if the ``[multimodal]`` extra is absent.
    """

    backbone: str = "colqwen2.5"
    token_pooling: bool = True  # Ch 12: apply from day one (−66.7% vectors, −2.2% quality)
    pool_factor: int = 3
    device: str = "cuda"
    _model: Any = field(default=None, repr=False, compare=False)
    _processor: Any = field(default=None, repr=False, compare=False)
    # stored per-page multi-vectors: (page_ref, multi_vector_tensor)
    _pages: list[tuple[Any, Any]] = field(default_factory=list, repr=False, compare=False)

    @classmethod
    def default(cls) -> "VisualRetriever":
        """ColQwen2.5 + token pooling — the chapter's committed default."""
        return cls(backbone="colqwen2.5", token_pooling=True)

    @classmethod
    def backbone_(cls, name: str) -> "VisualRetriever":
        """Pick a backbone: "colqwen2.5" | "colqwen2" | "colpali" | "colsmol" (Ch 12 model zoo)."""
        if name not in _BACKBONES:
            raise ValueError(f"Unknown backbone '{name}'. Choices: {sorted(_BACKBONES)}.")
        return cls(backbone=name)

    # -- lazy model loading ----------------------------------------------------
    def _load(self) -> None:
        """Lazily import colpali-engine and load the backbone model + processor (Ch 12)."""
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401  # colpali-engine's runtime dep
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(_INSTALL_HINT) from exc

        model_id = _BACKBONES[self.backbone]
        self._model = ColQwen2_5.from_pretrained(model_id, device_map=self.device).eval()
        self._processor = ColQwen2_5_Processor.from_pretrained(model_id)

    # -- the visual-retrieval surface (lazy) -----------------------------------
    def embed_page(self, image: Any) -> Any:
        """Encode one page image to its per-patch multi-vector (Ch 12: the page *is* the unit).

        ``image`` is a PIL image of a rendered page. Returns the page's multi-vector tensor (token
        pooling applied when enabled). No OCR, no layout step, no chunker anywhere in this path.
        """
        self._load()
        import torch

        batch = self._processor.process_images([image]).to(self._model.device)
        with torch.no_grad():
            vectors = self._model(**batch)
        page_vec = vectors[0]
        if self.token_pooling:
            page_vec = self._pool(page_vec)
        return page_vec

    def index(self, pages: list[Any]) -> "VisualRetriever":
        """Render-and-embed each page, storing its multi-vector for late-interaction search."""
        for page in pages:
            self._pages.append((page, self.embed_page(page)))
        return self

    def search(self, query: str, top_k: int = 5) -> list[tuple[Any, float]]:
        """MaxSim the query's per-token vectors against indexed page patches; return top pages.

        Returns ``[(page_ref, score), ...]`` best-first — page images you hand directly to a
        *multimodal* generator (Ch 12: text-only generation has not escaped extraction, only deferred
        it). The MaxSim scorer is colpali-engine's, the same late-interaction op dissected in Ch 4.
        """
        self._load()
        import torch

        batch = self._processor.process_queries([query]).to(self._model.device)
        with torch.no_grad():
            q_vec = self._model(**batch)
        page_vecs = [v for _, v in self._pages]
        if not page_vecs:
            return []
        scores = self._processor.score_multi_vector(q_vec, page_vecs)[0]
        order = torch.argsort(scores, descending=True)[:top_k]
        return [(self._pages[int(i)][0], float(scores[int(i)])) for i in order]

    def _pool(self, page_vec: Any) -> Any:
        """Token pooling: cluster/merge similar patch vectors before indexing (Ch 12 storage hedge)."""
        try:
            from colpali_engine.compression.token_pooling import HierarchicalTokenPooler  # type: ignore
        except ImportError:  # pragma: no cover - colpali present but pooling util moved
            return page_vec
        pooler = HierarchicalTokenPooler(pool_factor=self.pool_factor)
        return pooler.pool_embeddings(page_vec)


__all__ = ["VisualRetriever"]
