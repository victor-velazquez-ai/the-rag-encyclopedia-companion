"""Test the assembled capstone pipeline end-to-end, offline (no API key)."""

from capstone.app.pipeline import ExtractiveAnswerer, RAGPipeline

CORPUS = [
    ("d1", "Customers may request a refund within thirty days of purchase."),
    ("d2", "The error code ERR_CONN_4021 indicates the connection was refused."),
    ("d3", "Standard delivery takes five to seven business days."),
]


def _pipe():
    return RAGPipeline.from_corpus(CORPUS, ExtractiveAnswerer(), top_k=3)


def test_pipeline_answers_with_citation():
    ans = _pipe().answer("refund thirty days")
    assert "refund" in ans.text.lower()
    assert ans.citations and not ans.abstained  # cited a passage


def test_pipeline_abstains_off_corpus():
    ans = _pipe().answer("airspeed velocity of an unladen swallow")
    assert ans.abstained is True
    assert ans.citations == []


def test_pipeline_routes_to_right_doc():
    ans = _pipe().answer("ERR_CONN_4021 connection refused")
    assert "ERR_CONN_4021" in ans.text


def test_extractive_answerer_overlap_ranking():
    a = ExtractiveAnswerer()
    out = a.generate("business days delivery", ["a refund policy", "standard delivery business days"])
    assert "delivery" in out.text and out.citations == [2]  # picked the higher-overlap passage
