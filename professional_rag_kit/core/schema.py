"""The chunk schema — the contract that flows through the entire pipeline (Book Ch 3, §metadata).

A chunk is never just text. The metadata stamped here is what later chapters depend on:
``doc_id`` for parent-child retrieval (Ch 3/10), ``section_path``/``page`` for citation (Ch 13),
``allowed_groups`` for retrieval-time access control (Ch 15), ``date`` for recency precedence (Ch 8).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One retrievable unit. Produced by chunking (Ch 3), embedded (Ch 4), indexed (Ch 5)."""

    id: str
    text: str
    embedding: list[float] | None = None

    # provenance — used by parent-child retrieval (Ch 3/10) and citation (Ch 13)
    doc_id: str
    section_path: list[str] = Field(default_factory=list)  # ["3 Methods", "3.1 Setup"]
    page: int | None = None

    # governance / precedence
    date: str | None = None                                 # ISO; recency/authority (Ch 8)
    allowed_groups: list[str] = Field(default_factory=list)  # retrieval-time ACL filter (Ch 15)
    tags: dict[str, str] = Field(default_factory=dict)

    def with_context(self, prefix: str) -> "Chunk":
        """Return a copy whose text is prefixed with situating context (Contextual Retrieval, Ch 3).

        The original ``text`` is preserved as ``tags['original_text']`` so generation can cite the
        raw passage while retrieval matches against the context-enriched version.
        """
        return self.model_copy(
            update={
                "text": f"{prefix}\n\n{self.text}",
                "tags": {**self.tags, "original_text": self.text},
            }
        )


__all__ = ["Chunk"]
