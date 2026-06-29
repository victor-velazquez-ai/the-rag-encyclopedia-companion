"""Tests for the pure cores of Part III architectures (Book Ch 9–10): graph, hierarchy, routing.

Only the dependency-free, deterministic logic is exercised here — no LLM or embedding calls. The
LLM-backed paths (GraphIndex.build/global_query, RaptorTree.build/query, AdaptiveRAG.crag/self_rag/
flare) are lazy patterns over your frontier model and are verified by import + signature, not run.
"""

from professional_rag_kit.architectures.adaptive import AdaptiveRAG
from professional_rag_kit.architectures.graph import GraphIndex, parse_triples
from professional_rag_kit.architectures.hierarchical import ParentChildIndex


# --- GraphIndex: from_triples + local_query neighborhood (Ch 9, pure) --------
TRIPLES = [
    ("Acme", "buys from", "Beta Corp"),
    ("Beta Corp", "is owned by", "Cyrus Holdings"),
    ("Cyrus Holdings", "is", "Sanctioned Entity"),
    ("Delta Inc", "buys from", "Echo Ltd"),  # disconnected component
]


def test_graph_from_triples_builds_edges():
    g = GraphIndex.from_triples(TRIPLES)
    assert len(g.triples) == 4
    assert "Acme" in g.entities()
    assert "Beta Corp" in g.entities()


def test_graph_from_triples_dedupes_and_strips():
    g = GraphIndex.from_triples(
        [("Acme", "buys from", "Beta Corp"), ("  Acme ", "buys from", "Beta Corp")]
    )
    assert len(g.triples) == 1  # exact-duplicate (whitespace-normalized) edge dropped


def test_local_query_returns_one_hop_neighborhood():
    g = GraphIndex.from_triples(TRIPLES)
    nbhd = g.local_query("Beta Corp")  # default 1 hop
    assert nbhd["entity"] == "Beta Corp"
    # one hop from Beta Corp reaches its direct graph neighbors, both directions
    assert "Acme" in nbhd["neighbors"]
    assert "Cyrus Holdings" in nbhd["neighbors"]
    assert "Delta Inc" not in nbhd["neighbors"]  # other component not reachable


def test_local_query_multi_hop_traverses_chain():
    g = GraphIndex.from_triples(TRIPLES)
    nbhd = g.local_query("Acme", hops=3)  # multi-hop: chase the supply/ownership chain
    assert "Sanctioned Entity" in nbhd["neighbors"]


def test_local_query_unknown_entity_is_empty():
    g = GraphIndex.from_triples(TRIPLES)
    nbhd = g.local_query("Nonexistent")
    assert nbhd["neighbors"] == [] and nbhd["triples"] == []


def test_parse_triples_tolerant():
    text = (
        "subject | relation | object\n"  # header dropped
        "- Acme | buys from | Beta Corp\n"
        "some prose with no pipes\n"
        "Beta Corp | is owned by | Cyrus Holdings\n"
    )
    out = parse_triples(text)
    assert ("Acme", "buys from", "Beta Corp") in out
    assert ("Beta Corp", "is owned by", "Cyrus Holdings") in out
    assert len(out) == 2


# --- ParentChildIndex: child -> parent (Ch 10, pure) -------------------------
def test_parent_child_retrieve_maps_children_to_parents():
    idx = ParentChildIndex()
    idx.add("p1", "PARENT ONE full text", [("c1", "child a"), ("c2", "child b")])
    idx.add("p2", "PARENT TWO full text", [("c3", "child c")])
    assert idx.parent_of("c1") == "p1"
    assert idx.retrieve(["c2"]) == ["PARENT ONE full text"]


def test_parent_child_dedupes_shared_parent():
    idx = ParentChildIndex()
    idx.add("p1", "PARENT ONE", [("c1", "a"), ("c2", "b")])
    idx.add("p2", "PARENT TWO", [("c3", "c")])
    # c1 and c2 share p1 (auto-merge to one passage); c3 -> p2; unknown id skipped
    out = idx.retrieve(["c1", "c2", "c3", "missing"])
    assert out == ["PARENT ONE", "PARENT TWO"]


# --- AdaptiveRAG.route: complexity heuristic on clear cases (Ch 10, pure) ----
def test_route_no_retrieval_for_simple_parametric():
    ar = AdaptiveRAG()
    assert ar.route("What is the capital of France?") == "no_retrieval"
    assert ar.route("Define photosynthesis") == "no_retrieval"


def test_route_multi_for_compositional():
    ar = AdaptiveRAG()
    assert ar.route("Compare LightRAG and GraphRAG on index cost") == "multi"
    assert ar.route("How does the sanction affect suppliers connected to the entity?") == "multi"


def test_route_single_for_plain_factoid():
    ar = AdaptiveRAG()
    assert ar.route("Our return policy window in days") == "single"


def test_route_multi_beats_no_retrieval_when_both_cue():
    ar = AdaptiveRAG()
    # "what is" (no-retrieval cue) + "difference between" (multi cue): compositional wins
    assert ar.route("What is the difference between BM25 and dense retrieval?") == "multi"


def test_router_override_takes_precedence():
    ar = AdaptiveRAG(router=lambda q: "single")
    assert ar.route("What is the capital of France?") == "single"
