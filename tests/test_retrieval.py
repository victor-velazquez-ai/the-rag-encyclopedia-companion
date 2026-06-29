"""Tests for BM25, hybrid (sparse-only), context assembly, and LLM-listwise reranking."""

import pytest

from professional_rag_kit.core.config import ProviderRegistry
from professional_rag_kit.retrieval.context import ContextBuilder, mmr, reorder_lost_in_middle
from professional_rag_kit.retrieval.hybrid import BM25, HybridRetriever
from professional_rag_kit.retrieval.rerank import Reranker

CORPUS = [
    ("d1", "The error code ERR_CONN_4021 means the connection was refused."),
    ("d2", "Automobiles and cars share the same meaning in everyday speech."),
    ("d3", "Our return policy allows refunds within thirty days of purchase."),
]


def test_bm25_finds_exact_identifier():
    bm = BM25(CORPUS)
    hits = bm.rank("ERR_CONN_4021 refused", top_k=3)
    assert hits[0] == "d1"  # exact-match strength


def test_bm25_empty_for_no_overlap():
    bm = BM25(CORPUS)
    assert bm.search("xylophone", top_k=3) == []


def test_hybrid_sparse_only_search():
    hr = HybridRetriever.from_corpus(CORPUS)  # no dense leg
    fused = hr.search("refund policy", top_k=3)
    assert fused[0][0] == "d3"


# --- context assembly --------------------------------------------------------
def test_zipper_fold_places_strongest_on_peaks():
    folded = reorder_lost_in_middle(["r1", "r2", "r3", "r4", "r5"])
    assert folded[0] == "r1"  # strongest first
    assert folded[-1] == "r2"  # second strongest last
    assert folded[1] == "r3"  # then zippering inward
    assert "r5" in folded[1:-1]  # weakest in the middle


def test_mmr_relevance_dominant_keeps_near_duplicate():
    # q ~ A; A and A' are near-duplicates, B is different but less relevant.
    q = [1.0, 0.0]
    docs = [[1.0, 0.0], [0.98, 0.02], [0.7, 0.7]]  # A, A'(dup), B
    # λ=0.7 (relevance-dominant, the production default): the more-relevant A' is kept.
    assert mmr(q, docs, k=2, lambda_=0.7) == [0, 1]


def test_mmr_diversity_leaning_breaks_redundancy():
    q = [1.0, 0.0]
    docs = [[1.0, 0.0], [0.98, 0.02], [0.7, 0.7]]  # A, A'(dup), B
    chosen = mmr(q, docs, k=2, lambda_=0.3)  # diversity-leaning
    assert chosen[0] == 0  # most relevant first
    assert chosen[1] == 2  # now diversity beats the near-duplicate A'


def test_context_builder_budget_and_fold():
    cb = ContextBuilder(token_budget=10)
    out = cb.build("q", ["one two three", "four five six", "seven eight nine", "ten eleven twelve"])
    # budget 10 words keeps ~3 passages; output is folded, not empty
    assert out
    assert out.startswith("one two three")  # rank 1 on the front peak


# --- LLM listwise reranking (fake backend, no API key) -----------------------
def test_llm_listwise_reorders_by_model_output():
    ProviderRegistry.register("generation", "_fake_rank")(
        lambda model, system, prompt, max_tokens: "3, 1, 2"
    )
    rr = Reranker(provider="llm", gen_provider="_fake_rank", gen_model="x")
    out = rr.rerank("q", ["A", "B", "C"])
    assert out == ["C", "A", "B"]


def test_llm_listwise_backfills_dropped_indices():
    ProviderRegistry.register("generation", "_fake_partial")(
        lambda model, system, prompt, max_tokens: "2"  # model named only one
    )
    rr = Reranker(provider="llm", gen_provider="_fake_partial", gen_model="x")
    out = rr.rerank("q", ["A", "B", "C"])
    assert out[0] == "B" and set(out) == {"A", "B", "C"}  # rest appended stably


def test_reranker_empty_candidates():
    assert Reranker.default().rerank("q", []) == []
