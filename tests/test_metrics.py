"""Tests for retrieval metrics (Ch 14)."""

import math

import pytest

from ragkit.eval.metrics import dcg_at_k, mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_is_order_agnostic():
    relevant = {"A", "B"}
    assert recall_at_k(["A", "X", "B"], relevant, k=3) == 1.0
    assert recall_at_k(["B", "X", "A"], relevant, k=3) == 1.0  # order doesn't matter
    assert recall_at_k(["A", "X", "Y"], relevant, k=3) == 0.5


def test_recall_at_k_respects_cutoff():
    assert recall_at_k(["X", "Y", "A"], {"A"}, k=2) == 0.0
    assert recall_at_k(["X", "Y", "A"], {"A"}, k=3) == 1.0


def test_recall_empty_relevant_is_zero():
    assert recall_at_k(["A"], set(), k=1) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert mrr(["X", "A", "B"], {"A", "B"}) == pytest.approx(1 / 2)
    assert mrr(["A"], {"A"}) == 1.0
    assert mrr(["X", "Y"], {"A"}) == 0.0


def test_dcg_matches_formula():
    # gains: A=3 at rank 1, B=2 at rank 2 → 3/log2(2) + 2/log2(3)
    gains = {"A": 3.0, "B": 2.0}
    expected = 3.0 / math.log2(2) + 2.0 / math.log2(3)
    assert dcg_at_k(["A", "B"], gains, k=2) == pytest.approx(expected)


def test_ndcg_perfect_order_is_one():
    gains = {"A": 3.0, "B": 2.0, "C": 1.0}
    assert ndcg_at_k(["A", "B", "C"], gains, k=3) == pytest.approx(1.0)


def test_ndcg_worse_order_is_less_than_one():
    gains = {"A": 3.0, "B": 2.0, "C": 1.0}
    assert ndcg_at_k(["C", "B", "A"], gains, k=3) < 1.0


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k(["A", "B"], {}, k=2) == 0.0
