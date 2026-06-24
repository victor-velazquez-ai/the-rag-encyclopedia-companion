"""Tests for the eval harness (Ch 14): GoldenSet, Harness scoring, and the offline suite."""

from ragkit.eval import GoldenSet, Harness


def _golden():
    return GoldenSet.from_items(
        [
            {"query": "q1", "relevant": ["a"]},
            {"query": "q2", "relevant": ["b"], "gains": {"b": 3.0}},
        ]
    )


def test_goldenset_iter_and_gain_map():
    g = _golden()
    assert len(g) == 2
    items = list(g)
    assert items[0].gain_map() == {"a": 1.0}  # defaults to 1.0 when no gains given
    assert items[1].gain_map() == {"b": 3.0}


def test_perfect_retriever_scores_one():
    g = _golden()
    answers = {"q1": ["a", "x"], "q2": ["b", "y"]}
    card = Harness.default().score_retrieval("perfect", lambda q: answers[q], g, k=5)
    assert card.ndcg == 1.0 and card.recall == 1.0 and card.mrr == 1.0
    assert card.n == 2


def test_worse_retriever_scores_lower():
    g = _golden()
    good = Harness.default().score_retrieval("good", lambda q: {"q1": ["a"], "q2": ["b"]}[q], g)
    bad = Harness.default().score_retrieval("bad", lambda q: {"q1": ["x", "a"], "q2": ["y", "b"]}[q], g)
    assert bad.mrr < good.mrr  # relevant doc buried at rank 2


def test_compare_sorts_by_ndcg():
    g = _golden()
    retrievers = {
        "top": lambda q: {"q1": ["a"], "q2": ["b"]}[q],
        "bottom": lambda q: {"q1": ["x", "a"], "q2": ["y", "b"]}[q],
    }
    cards = Harness.default().compare(retrievers, g, k=5)
    assert cards[0].name == "top"  # best nDCG first


def test_offline_suite_runs_and_fusion_does_not_regress():
    # the repo's sample data + suite must run with no key and produce sane numbers
    from ragkit.eval.suite import _build_retrievers, _load_corpus, _DATA

    corpus = _load_corpus()
    golden = GoldenSet.from_jsonl(_DATA / "golden" / "qa.jsonl")
    cards = Harness.default().compare(_build_retrievers(corpus), golden, k=5)
    by_name = {c.name: c for c in cards}
    assert by_name["BM25 (body only)"].recall > 0.5  # the corpus is BM25-friendly
    # fusion adds the title leg; on this corpus it should not hurt nDCG
    assert by_name["BM25 body + title (RRF)"].ndcg >= by_name["BM25 (body only)"].ndcg - 1e-9
