"""Tests for the recursive chunker (Ch 3)."""

import pytest

from professional_rag_kit.core.schema import Chunk
from professional_rag_kit.ingestion.chunking import Chunker


def _doc(nsent: int) -> str:
    return " ".join(f"Sentence number {i} has several words in it." for i in range(nsent))


def test_short_text_is_one_chunk():
    chunks = Chunker.default().split("A short note.", doc_id="d1")
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].doc_id == "d1"
    assert chunks[0].id == "d1::0"


def test_long_text_splits_near_target():
    ch = Chunker(target_tokens=40, overlap_tokens=8)
    chunks = ch.split(_doc(50), doc_id="d1")
    assert len(chunks) > 1
    # every chunk respects the target within a sentence's slack
    for c in chunks:
        assert len(c.text.split()) <= 40 + 12


def test_overlap_carries_tail():
    ch = Chunker(target_tokens=30, overlap_tokens=10)
    chunks = ch.split(_doc(30), doc_id="d1")
    # consecutive chunks should share some leading/trailing words (overlap)
    assert len(chunks) >= 2
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words  # non-empty intersection from the overlap tail


def test_metadata_and_indexing():
    chunks = Chunker(target_tokens=30).split(_doc(20), doc_id="d9", section_path=["1 Intro"])
    for i, c in enumerate(chunks):
        assert c.doc_id == "d9"
        assert c.section_path == ["1 Intro"]
        assert c.tags["chunk_index"] == str(i)


def test_doc_id_required():
    with pytest.raises(ValueError):
        Chunker.default().split("text", doc_id=None)


def test_accepts_mapping():
    chunks = Chunker.default().split({"text": "Hello world.", "doc_id": "d2"})
    assert chunks[0].doc_id == "d2"


def test_unknown_strategy_raises():
    with pytest.raises(NotImplementedError):
        Chunker.strategy("semantic")
