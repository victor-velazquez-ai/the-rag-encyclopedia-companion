"""professional_rag_kit.ingestion.indexing — where the vectors live and how you find them fast (Book Ch 5).

A bad index returns plausible neighbors that are merely *not the nearest ones*, and you never see it
unless you measure recall against an exhaustive baseline. The companion standardizes on **Qdrant**
(local Docker) because it exposes the mechanisms the chapter cares about — filterable HNSW, in-index
quantization, tunable search — rather than hiding them behind one "search" call.

This ``VectorStore`` wraps Qdrant with the chapter's defaults: HNSW + int8 scalar quantization with a
full-precision rescore (``quantize for the first pass, rescore the shortlist``), and *filterable* HNSW
so an access-control / metadata filter rides inside the ANN traversal — never a naive post-filter,
which silently collapses recall on selective filters (the Ch 15 security path depends on this).

SDK import is lazy; this module imports without ``qdrant-client`` installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from professional_rag_kit.core.schema import Chunk


@dataclass
class Hit:
    id: str
    score: float
    chunk: Chunk


@dataclass
class VectorStore:
    """Facade over a Qdrant collection. ``default()`` → filterable HNSW + int8 + rescore."""

    collection: str = "professional_rag_kit"
    url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333"))
    dim: int = 3072  # text-embedding-3-large
    quantization: str = "int8"
    _client: Any = None

    @classmethod
    def default(cls, **kw) -> "VectorStore":
        return cls(**kw)

    @classmethod
    def connect(cls, url: str = "", *, collection: str = "professional_rag_kit", dim: int = 3072,
                quantization: str = "int8") -> "VectorStore":
        vs = cls(collection=collection, url=url or os.environ.get("QDRANT_URL", "http://localhost:6333"),
                 dim=dim, quantization=quantization)
        vs.ensure_collection()
        return vs

    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient  # lazy

            self._client = QdrantClient(url=self.url)
        return self._client

    def ensure_collection(self) -> None:
        from qdrant_client import models as qm  # lazy

        c = self.client()
        if c.collection_exists(self.collection):
            return
        quant = None
        if self.quantization == "int8":
            quant = qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, always_ram=True)
            )
        elif self.quantization == "binary":
            quant = qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(always_ram=True))
        c.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
            quantization_config=quant,
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert chunks (each must carry an ``embedding``). Incremental — extends the HNSW graph."""
        from qdrant_client import models as qm  # lazy

        points = [
            qm.PointStruct(
                id=i,
                vector=ch.embedding,
                payload={"chunk_id": ch.id, "text": ch.text, "doc_id": ch.doc_id,
                         "allowed_groups": ch.allowed_groups, **ch.tags},
            )
            for i, ch in enumerate(chunks)
            if ch.embedding is not None
        ]
        self.client().upsert(collection_name=self.collection, points=points)

    def search(self, vector, top_k: int = 10, *, allowed_groups: list[str] | None = None,
               oversample: int = 4) -> list[Hit]:
        """Filter-aware ANN: oversampled first pass + full-precision rescore (Ch 5).

        If ``allowed_groups`` is given, the ACL filter rides *inside* the HNSW traversal (Ch 15).
        """
        from qdrant_client import models as qm  # lazy

        qfilter = None
        if allowed_groups:
            qfilter = qm.Filter(
                must=[qm.FieldCondition(key="allowed_groups", match=qm.MatchAny(any=allowed_groups))]
            )
        params = qm.SearchParams(
            quantization=qm.QuantizationSearchParams(rescore=True, oversampling=oversample)
        )
        res = self.client().query_points(
            collection_name=self.collection, query=vector, limit=top_k,
            query_filter=qfilter, search_params=params, with_payload=True,
        ).points
        return [
            Hit(
                id=p.payload.get("chunk_id", str(p.id)),
                score=p.score,
                chunk=Chunk(id=p.payload.get("chunk_id", str(p.id)), text=p.payload.get("text", ""),
                            doc_id=p.payload.get("doc_id", ""),
                            allowed_groups=p.payload.get("allowed_groups", [])),
            )
            for p in res
        ]


__all__ = ["VectorStore", "Hit"]
