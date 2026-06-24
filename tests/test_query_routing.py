"""Tests for the pure logic in query routing + parsing (Book Ch 2, Ch 6).

Covers the dependency-free pieces: ``ComplexityClassifier`` (Adaptive-RAG complexity cues),
``route_by_vectors`` (cosine k-NN semantic routing), and ``html_table_to_markdown`` (Ch 2 table
serialization). The LLM/embedding-backed methods are import-verified only (no key in CI)."""

import pytest

from ragkit.ingestion.parsing import html_table_to_markdown, parse
from ragkit.retrieval.routing import ComplexityClassifier, route_by_vectors


# --- ComplexityClassifier (PURE) ---------------------------------------------
def test_complexity_no_retrieval_for_chitchat():
    cc = ComplexityClassifier()
    assert cc.classify("hello there") == "no_retrieval"
    assert cc.classify("thanks!") == "no_retrieval"


def test_complexity_single_for_plain_factoid():
    cc = ComplexityClassifier()
    assert cc.classify("what is the refund window for an online order") == "single"


def test_complexity_multi_on_compare_marker():
    cc = ComplexityClassifier()
    assert cc.classify("compare the revenue growth of the two divisions") == "multi"


def test_complexity_multi_on_and_then_marker():
    cc = ComplexityClassifier()
    assert cc.classify("find the CEO and then their birthplace") == "multi"


def test_complexity_multi_on_multiple_entities():
    cc = ComplexityClassifier()
    # two distinct capitalized entities -> multi-hop by the entity heuristic
    assert cc.classify("How did Tesla influence SpaceX strategy") == "multi"


def test_complexity_biases_toward_retrieving():
    cc = ComplexityClassifier()
    # a question that merely greets is still a question -> not no_retrieval (bias to retrieve)
    assert cc.classify("hello, what is our return policy?") != "no_retrieval"


# --- route_by_vectors (PURE cosine k-NN) -------------------------------------
def test_route_by_vectors_nearest_wins():
    routes = [[1.0, 0.0], [0.0, 1.0]]
    labels = ["billing", "code_search"]
    # query points along the first axis -> nearest is "billing"
    assert route_by_vectors([0.9, 0.1], routes, labels) == "billing"
    # query points along the second axis -> nearest is "code_search"
    assert route_by_vectors([0.1, 0.9], routes, labels) == "code_search"


def test_route_by_vectors_is_cosine_scale_invariant():
    routes = [[1.0, 0.0], [0.0, 1.0]]
    labels = ["a", "b"]
    # magnitude does not matter, only direction (cosine)
    assert route_by_vectors([50.0, 1.0], routes, labels) == "a"


def test_route_by_vectors_knn_majority():
    # three "a" utterances cluster near the query, one "b" is closest-but-outnumbered
    routes = [[1.0, 0.05], [0.98, 0.0], [0.95, 0.1], [0.0, 1.0]]
    labels = ["a", "a", "a", "b"]
    assert route_by_vectors([1.0, 0.0], routes, labels, k=3) == "a"


def test_route_by_vectors_validates_inputs():
    with pytest.raises(ValueError):
        route_by_vectors([1.0], [], [])
    with pytest.raises(ValueError):
        route_by_vectors([1.0, 0.0], [[1.0, 0.0]], ["a", "b"])  # mismatched lengths


# --- html_table_to_markdown (PURE, Ch 2) -------------------------------------
def test_html_table_to_markdown_basic_grid():
    html = (
        "<table>"
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "<tr><td>1.2</td><td>1.4</td></tr>"
        "</table>"
    )
    md = html_table_to_markdown(html)
    lines = md.splitlines()
    assert lines[0] == "| Q1 | Q2 |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1.2 | 1.4 |"


def test_html_table_colspan_is_flattened():
    # Ch 2: Markdown cannot express a merged cell; colspan is flattened (repeated) to stay rectangular
    html = (
        "<table>"
        '<tr><th colspan="2">FY2025</th></tr>'
        "<tr><th>Q1</th><th>Q2</th></tr>"
        "<tr><td>1.2</td><td>1.4</td></tr>"
        "</table>"
    )
    md = html_table_to_markdown(html)
    lines = md.splitlines()
    assert lines[0] == "| FY2025 | FY2025 |"  # colspan=2 repeated across both columns
    assert lines[2] == "| Q1 | Q2 |"


def test_html_table_escapes_pipes_and_entities():
    html = "<table><tr><td>a|b</td><td>x &amp; y</td></tr></table>"
    md = html_table_to_markdown(html)
    assert r"a\|b" in md
    assert "x & y" in md


def test_html_table_empty_returns_empty():
    assert html_table_to_markdown("<p>no table here</p>") == ""


# --- parse() dispatch (pure paths) -------------------------------------------
def test_parse_plaintext_passthrough():
    assert parse("just some text", content_type="text/plain") == "just some text"


def test_parse_html_inlines_table_as_markdown():
    html = "<html><body><p>Intro</p><table><tr><th>A</th></tr><tr><td>1</td></tr></table></body></html>"
    out = parse(html, content_type="text/html")
    assert "Intro" in out
    assert "| A |" in out
    assert "| 1 |" in out
