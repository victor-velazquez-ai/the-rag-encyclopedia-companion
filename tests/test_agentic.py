"""Tests for Ch 11 agentic RAG: EntityResolver (the differentiator), the plan-act-reflect loop,
and the MCP source boundary. The EntityResolver tests are the thorough ones — deterministic-first,
fuzzy-second, the review band, normalization, blocking, and misses."""

import pytest

from ragkit.architectures.agentic import (
    AgenticRAG,
    EntityResolver,
    MCPSource,
    normalize,
)


# --- normalization -----------------------------------------------------------
def test_normalize_strips_punct_case_and_legal_suffix():
    assert normalize("Acme, Inc.") == "acme"
    assert normalize("ACME Corporation") == "acme"
    assert normalize("I.B.M.") == "i b m"
    assert normalize("  Acme   Corp  ") == "acme"


def test_normalize_keeps_distinguishing_tokens():
    # only *trailing* legal suffixes are dropped; "co" inside a name survives
    assert normalize("Coca Cola Co") == "coca cola"
    assert normalize("Northrop Grumman") == "northrop grumman"


# --- deterministic / exact matching (runs first) -----------------------------
def test_deterministic_exact_alias_hit():
    er = EntityResolver()
    er.add_canonical("ENT_ACME", "Acme Corp", "ACME Corporation", "Acme Inc.")
    res = er.resolve_detail("acme corporation")
    assert res.method == "deterministic"
    assert res.canonical_id == "ENT_ACME"
    assert res.score == 1.0


def test_deterministic_matches_across_surface_forms_via_normalization():
    # the four "Acmes" of Ch 11 collapse deterministically once normalized
    er = EntityResolver().add_canonical("ENT_ACME", "Acme Corp", "ACME Corporation", "Acme Inc.")
    for surface in ["Acme Corp", "ACME Corporation", "Acme, Inc.", "  acme   inc  "]:
        assert er.resolve(surface) == "ENT_ACME"


def test_canonical_id_is_itself_an_alias():
    er = EntityResolver().add_canonical("ENT_IBM", "International Business Machines")
    assert er.resolve("ENT_IBM") == "ENT_IBM"


# --- fuzzy matching (runs second, only on the deterministic remainder) -------
def test_fuzzy_ibm_vs_dotted_acronym():
    er = EntityResolver().add_canonical("ENT_IBM", "IBM")
    # "I.B.M." normalizes to "i b m" (not exact vs "ibm") → resolved by the fuzzy stage
    res = er.resolve_detail("I.B.M.")
    assert res.canonical_id == "ENT_IBM"
    assert res.method == "fuzzy"
    assert res.score >= 0.85


def test_fuzzy_acme_inc_typo():
    er = EntityResolver().add_canonical("ENT_ACME", "Acme")
    res = er.resolve_detail("Acmee")  # one-char typo, not a legal suffix
    assert res.method == "fuzzy"
    assert res.canonical_id == "ENT_ACME"


def test_acme_inc_punct_variants_are_deterministic_not_fuzzy():
    # "Acme, Inc." vs "Acme Inc" both normalize to "acme" → exact, no fuzzy needed
    er = EntityResolver().add_canonical("ENT_ACME", "Acme Inc")
    res = er.resolve_detail("Acme, Inc.")
    assert res.method == "deterministic"
    assert res.canonical_id == "ENT_ACME"


# --- the review band (the three-way decision) --------------------------------
def test_review_band_returns_uncertain_match():
    er = EntityResolver(threshold=0.85, review_low=0.6)
    er.add_canonical("ENT_ACME", "Acme Corporation")
    # a partial overlap that lands between review_low and threshold
    res = er.resolve_detail("Acma Corp")
    assert res.review is True
    assert res.method == "review"
    assert er.review_low <= res.score < er.threshold
    # resolve() (the decision API) returns None for an uncertain match
    assert er.resolve("Acma Corp") is None


def test_above_threshold_is_a_decision_not_review():
    er = EntityResolver(threshold=0.85, review_low=0.6)
    er.add_canonical("ENT_GLOBEX", "Globex Industries")
    res = er.resolve_detail("Globex Industires")  # tiny typo, very high ratio
    assert res.review is False
    assert res.method == "fuzzy"
    assert er.resolve("Globex Industires") == "ENT_GLOBEX"


# --- misses ------------------------------------------------------------------
def test_miss_below_review_low():
    er = EntityResolver()
    er.add_canonical("ENT_ACME", "Acme Corporation")
    res = er.resolve_detail("Globex Industries")
    assert res.canonical_id is None
    assert res.method == "none"
    assert er.resolve("Globex Industries") is None


def test_empty_and_punct_only_mentions_miss():
    er = EntityResolver().add_canonical("ENT_ACME", "Acme")
    assert er.resolve("") is None
    assert er.resolve("   ") is None
    assert er.resolve_detail("!!!").method == "none"


def test_distinct_entities_stay_distinct():
    er = EntityResolver().add_canonical("ENT_ACME", "Acme").add_canonical("ENT_GLOBEX", "Globex")
    assert er.resolve("Acme") == "ENT_ACME"
    assert er.resolve("Globex") == "ENT_GLOBEX"


# --- blocking ----------------------------------------------------------------
def test_blocking_restricts_fuzzy_to_shared_first_token():
    er = EntityResolver(blocking=True, threshold=0.7, review_low=0.5)
    er.add_canonical("ENT_ACME", "Acme Holdings")
    er.add_canonical("ENT_APEX", "Apex Holdings")
    # "Acme Holdngs" shares first token "acme" → blocks to ENT_ACME, never compared to Apex
    res = er.resolve_detail("Acme Holdngs")
    assert res.canonical_id == "ENT_ACME"


def test_blocking_can_miss_when_first_token_differs():
    er = EntityResolver(blocking=True, threshold=0.5, review_low=0.5)
    er.add_canonical("ENT_ACME", "Acme Holdings")
    # different first token → blocked out of the only candidate (the documented recall tradeoff)
    assert er.resolve("Zcme Holdings") is None


def test_threshold_validation():
    with pytest.raises(ValueError):
        EntityResolver(threshold=0.7, review_low=0.8)  # review_low must be <= threshold


# --- AgenticRAG: the plan-act-reflect loop (fake retriever + generator) ------
def _fake_retriever(query):
    return [f"passage about {query}"]


def test_agentic_simple_query_skips_the_loop():
    # gate routes a local fact lookup to the single-shot path — never enters the loop
    gen = lambda system, prompt: "synthesized answer"
    ag = AgenticRAG(retriever=_fake_retriever, generator=gen)
    out = ag.run("define mitochondria")
    assert out["looped"] is False
    assert out["terminated"] == "single_shot"
    assert len(out["hops"]) == 1


def test_agentic_loop_terminates_on_enough():
    calls = {"n": 0}

    def gen(system, prompt):
        if "ENOUGH" in system:  # the reflect step
            calls["n"] += 1
            return "ENOUGH" if calls["n"] >= 2 else "MORE"
        return "next subquery"

    ag = AgenticRAG(retriever=_fake_retriever, generator=gen, max_hops=5)
    # a compositional query (has " and ") enters the loop
    out = ag.run("which subsidiary and its competitor share a state")
    assert out["looped"] is True
    assert out["terminated"] == "enough"
    assert 1 <= len(out["hops"]) <= 5


def test_agentic_loop_respects_turn_budget():
    # reflect always says MORE → loop must stop at the hard cap (non-termination guard)
    gen = lambda system, prompt: "MORE" if "ENOUGH" in system else "subquery"
    ag = AgenticRAG(retriever=_fake_retriever, generator=gen, max_hops=3)
    out = ag.run("compare both companies and their largest competitors and subsidiaries")
    assert out["terminated"] == "turn_budget"
    assert len(out["hops"]) == 3


def test_self_ask_decomposes_into_subquestions():
    gen = lambda system, prompt: (
        "Where is Acme headquartered?\nWho is Acme's largest competitor?"
        if "Self-Ask" in system or "follow-up" in system.lower()
        else "answer"
    )
    ag = AgenticRAG(retriever=_fake_retriever, generator=gen)
    out = ag.self_ask("which state is Acme's competitor in")
    assert out["method"] == "self_ask"
    assert len(out["hops"]) == 2


def test_ircot_is_the_loop_under_its_paper_name():
    gen = lambda system, prompt: "ENOUGH" if "ENOUGH" in system else "subquery"
    ag = AgenticRAG(retriever=_fake_retriever, generator=gen)
    out = ag.ircot("which of the two firms and its rival are larger")
    assert out["method"] == "ircot"


# --- MCPSource: the uniform tool boundary ------------------------------------
def test_mcp_source_wraps_an_in_process_retriever():
    src = MCPSource(name="vector_db", search_fn=lambda q: ["a", "b", "c", "d"])
    assert src.search("anything", top_k=2) == ["a", "b"]


def test_mcp_source_requires_a_backend():
    with pytest.raises(ValueError):
        MCPSource(name="empty").search("q")
