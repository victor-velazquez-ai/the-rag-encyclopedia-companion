"""The chunk schema — the contract that flows through the entire pipeline (Book Ch 3, §metadata).

A chunk is never just text. The metadata stamped here is what later chapters depend on:
``doc_id`` for parent-child retrieval (Ch 3/10), ``section_path``/``page`` for citation (Ch 13),
``allowed_groups`` for retrieval-time access control (Ch 15), ``date`` for freshness/recency.

Phase-1 scaffold: the intended shape is shown below as a spec; the Pydantic model is
implemented in Phase 2. This file stays importable.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class Chunk(BaseModel):
#     id: str
#     text: str
#     embedding: list[float] | None = None
#     doc_id: str
#     section_path: list[str] = []        # ["3 Methods", "3.1 Setup"] → citation, parent lookup
#     page: int | None = None
#     date: str | None = None             # ISO; recency/authority precedence (Ch 8)
#     allowed_groups: list[str] = []      # retrieval-time ACL filter (Ch 15)
#     tags: dict[str, str] = {}

__all__ = ["Chunk"]  # populated in Phase 2
