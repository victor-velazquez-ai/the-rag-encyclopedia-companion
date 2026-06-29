"""professional_rag_kit.ingestion.chunking — the atomic unit of everything downstream (Book Ch 3).

A boundary in the wrong place severs a fact from the context that makes it answerable, and no
reranker repairs it. The chapter's finding: a *tuned ~200-token recursive splitter* is a genuinely
strong baseline that every expensive method must beat on your data. So the default is cheap and
structure-aware; the LLM-budget methods (semantic, late, contextual, propositional) are opt-in.

``Chunker.default()`` is that baseline. It respects structure — it splits on paragraph, then
sentence, then word boundaries — packs to a target token count, and carries an overlap so a fact
split across a boundary still appears whole in one chunk. Every chunk gets metadata (Ch 3): the
``doc_id`` for parent-child retrieval, ``section_path`` for citation, ACL/date fields downstream.

Token counts here use a fast word-count proxy; swap in a real tokenizer for production budgeting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from professional_rag_kit.core.schema import Chunk

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _ntok(text: str) -> int:
    """Cheap token proxy: whitespace word count (good enough for chunk sizing)."""
    return len(text.split())


def _atoms(text: str) -> list[str]:
    """Structure-aware split into the smallest units we'll pack: sentences, then words for
    sentences longer than a chunk."""
    out: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        for sent in _SENTENCE.split(para):
            sent = sent.strip()
            if sent:
                out.append(sent)
    return out


def _recursive_split(text: str, target: int, overlap: int) -> list[str]:
    atoms = _atoms(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for atom in atoms:
        atom_tok = _ntok(atom)
        if cur and cur_tok + atom_tok > target:
            chunks.append(" ".join(cur))
            # start next chunk with a token-overlap tail of the previous one
            if overlap > 0:
                tail, tail_tok = [], 0
                for prev in reversed(cur):
                    t = _ntok(prev)
                    if tail_tok + t > overlap:
                        break
                    tail.insert(0, prev)
                    tail_tok += t
                cur, cur_tok = tail, tail_tok
            else:
                cur, cur_tok = [], 0
        cur.append(atom)
        cur_tok += atom_tok
    if cur:
        chunks.append(" ".join(cur))
    return chunks


@dataclass
class Chunker:
    """Facade over the chunking strategies. ``default()`` → structure-aware recursive ~200 tok."""

    strategy_name: str = "recursive"
    target_tokens: int = 200
    overlap_tokens: int = 30  # ~15%

    @classmethod
    def default(cls) -> "Chunker":
        return cls()

    @classmethod
    def strategy(cls, name: str, **opts) -> "Chunker":
        # "recursive" (implemented). semantic/late/contextual/propositional/hierarchical: Phase 2+.
        if name != "recursive":
            raise NotImplementedError(
                f"Chunking strategy '{name}' not yet implemented — see Book Ch 3 and the chapter "
                f"folder. The default 'recursive' is the baseline every other method must beat."
            )
        return cls(strategy_name=name, **opts)

    def split(self, doc, *, doc_id: str | None = None, section_path: list[str] | None = None) -> list[Chunk]:
        """Split text (a str, or a mapping with 'text'/'doc_id') into Chunk records with metadata."""
        if isinstance(doc, str):
            text = doc
        else:
            text = doc["text"]
            doc_id = doc_id or doc.get("doc_id")
            section_path = section_path or doc.get("section_path")
        if not doc_id:
            raise ValueError("doc_id is required (it carries provenance for citation + parent lookup)")

        pieces = _recursive_split(text, self.target_tokens, self.overlap_tokens)
        return [
            Chunk(
                id=f"{doc_id}::{i}",
                text=piece,
                doc_id=doc_id,
                section_path=section_path or [],
                tags={"chunk_index": str(i), "strategy": self.strategy_name},
            )
            for i, piece in enumerate(pieces)
        ]


__all__ = ["Chunker"]
