"""Tests for RRF fusion (Ch 6)."""

import math

import pytest

from professional_rag_kit.retrieval.hybrid.fusion import rrf_fuse


def test_doc_in_both_lists_outranks_doc_in_one():
    # D appears in both lists; A and X only in one each. D should win.
    bm25 = ["D", "A", "B"]
    dense = ["X", "Y", "D"]
    fused = rrf_fuse([bm25, dense])
    ids = [doc for doc, _ in fused]
    assert ids[0] == "D"


def test_exact_rrf_score_with_k60():
    # D at rank 1 in list1 and rank 3 in list2 → 1/61 + 1/63.
    fused = dict(rrf_fuse([["D"], ["X", "Y", "D"]], k=60))
    assert fused["D"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["X"] == pytest.approx(1 / 61)


def test_higher_rank_beats_lower_within_one_list():
    fused = rrf_fuse([["A", "B", "C"]])
    assert [d for d, _ in fused] == ["A", "B", "C"]


def test_ties_broken_deterministically_by_id():
    # A and B both at rank 1 of their own single-item lists → equal score, sorted by id.
    fused = rrf_fuse([["B"], ["A"]])
    assert [d for d, _ in fused] == ["A", "B"]


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        rrf_fuse([["A"]], k=0)
