"""ragkit.architectures.graph — knowledge-graph RAG (Book Ch 9).

A graph is an expensive structure to build and maintain, so the only honest reason to build one is
that your traffic contains global-sensemaking or multi-hop questions flat retrieval provably cannot
answer (Ch 9). The systems the chapter surveys — Microsoft GraphRAG, LightRAG, LazyGraphRAG — are
points on a cost-versus-structure curve; what they share is the same skeleton this module makes
concrete: extract ``(subject, relation, object)`` triples into a graph, then answer either *local*
(an entity's neighborhood, à la GraphRAG Local / LightRAG low-level) or *global* (a community/summary
synthesis, à la GraphRAG Global) questions over it.

``GraphIndex`` is that skeleton at LightRAG's pragmatic altitude — a graph + dual-level query, no
hierarchical community-report bill:

- ``from_triples`` / ``local_query`` are *pure* (no model, no SDK): build an in-memory graph and
  return an entity's neighborhood by edge traversal — the retrieval primitive a graph exists to give.
- ``build`` uses the generation backend (lazy, like ``production.generation``) to extract triples from
  text with a structured-extraction prompt, then delegates to ``from_triples``.
- ``global_query`` does the community/summary-style synthesis answer over the whole graph (lazy LLM).

LLM calls go through ``ragkit.core.config.ProviderRegistry`` — no vendor SDK is imported at module
top level, so this file imports fine with nothing installed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ragkit.core.config import ProviderRegistry, load_config

# Ch 9: the graph is only as good as the LLM's entity/relation extraction ("extraction quality is the
# ceiling"). Pin it down to clean, typed triples — one per line, pipe-delimited — so parsing is robust.
EXTRACTION_SYSTEM = (
    "You are a knowledge-graph extractor. From the text, extract factual relationships as triples.\n"
    "Output ONE triple per line in the exact form:  subject | relation | object\n"
    "Rules:\n"
    "1. Use specific, canonical entity names (not pronouns).\n"
    "2. Keep the relation a short verb phrase (e.g. 'founded', 'is owned by', 'located in').\n"
    "3. Extract only relationships the text actually states; do not infer from outside knowledge.\n"
    "4. Output ONLY the triple lines, nothing else."
)

# Ch 9 (GraphRAG Global): answer global/sensemaking questions by synthesizing over the graph's
# structure rather than over raw chunks — here, a flat rendering of the relationships stands in for
# the community reports a full pipeline would pre-compute.
GLOBAL_SYSTEM = (
    "You answer global, whole-corpus questions by synthesizing a knowledge graph's relationships.\n"
    "You are given a list of (subject, relation, object) facts. Identify the themes and connections "
    "across them and answer the question. Ground every claim in the supplied facts; if they do not "
    "support an answer, say so plainly."
)


@dataclass
class GraphIndex:
    """In-memory knowledge-graph index with dual-level (local / global) query (Book Ch 9).

    Stores directed, labeled edges as ``(subject, relation, object)`` triples plus an undirected
    adjacency view for neighborhood traversal. ``from_triples`` and ``local_query`` are pure;
    ``build`` and ``global_query`` use the lazy generation backend.
    """

    triples: list[tuple[str, str, str]] = field(default_factory=list)
    # entity -> list of (relation, neighbor, direction) where direction is "out" (entity is subject)
    # or "in" (entity is object). Undirected for traversal; direction is kept for faithful rendering.
    _adj: dict[str, list[tuple[str, str, str]]] = field(default_factory=lambda: defaultdict(list))

    # --- pure construction & query --------------------------------------------
    @classmethod
    def from_triples(cls, triples) -> "GraphIndex":
        """Build the graph from ``(subject, relation, object)`` triples — PURE (no model, no SDK).

        Whitespace is stripped and exact-duplicate edges are dropped; entity matching is
        case-insensitive on the surface form (the stored name is the first spelling seen).
        """
        idx = cls()
        seen: set[tuple[str, str, str]] = set()
        canon: dict[str, str] = {}  # lower(name) -> first-seen spelling

        def _name(raw: str) -> str:
            s = " ".join(str(raw).split())
            return canon.setdefault(s.lower(), s)

        for triple in triples:
            s, r, o = triple
            s, o = _name(s), _name(o)
            r = " ".join(str(r).split())
            if not s or not o or not r:
                continue
            key = (s.lower(), r.lower(), o.lower())
            if key in seen:
                continue
            seen.add(key)
            idx.triples.append((s, r, o))
            idx._adj[s].append((r, o, "out"))
            idx._adj[o].append((r, s, "in"))
        return idx

    def entities(self) -> list[str]:
        """All distinct entity names in the graph (pure)."""
        return sorted(self._adj.keys())

    def local_query(self, entity: str, hops: int = 1) -> dict:
        """Return ``entity``'s neighborhood — the graph's retrieval primitive (Ch 9 Local), PURE.

        ``hops`` (default 1) controls how far traversal spreads from the seed. Returns
        ``{"entity": <resolved name>, "neighbors": [...], "triples": [(s, r, o), ...]}`` where
        ``triples`` is every edge touching the visited subgraph — the local context a graph-RAG
        Local query feeds the generator. An unknown entity yields an empty neighborhood.
        """
        # case-insensitive resolve to the stored spelling
        resolved = None
        target = " ".join(str(entity).split()).lower()
        for name in self._adj:
            if name.lower() == target:
                resolved = name
                break
        if resolved is None:
            return {"entity": entity, "neighbors": [], "triples": []}

        visited: set[str] = {resolved}
        frontier = [resolved]
        for _ in range(max(0, hops)):
            nxt: list[str] = []
            for node in frontier:
                for _rel, nbr, _dir in self._adj.get(node, ()):
                    if nbr not in visited:
                        visited.add(nbr)
                        nxt.append(nbr)
            frontier = nxt

        neighbors = sorted(visited - {resolved})
        sub_triples = [t for t in self.triples if t[0] in visited and t[2] in visited]
        return {"entity": resolved, "neighbors": neighbors, "triples": sub_triples}

    # --- LLM-backed construction & query (lazy) -------------------------------
    @classmethod
    def build(
        cls,
        docs,
        *,
        provider: str = "anthropic",
        model: str = "",
        max_tokens: int = 2048,
    ) -> "GraphIndex":
        """Extract triples from ``docs`` with the LLM, then ``from_triples`` (Ch 9 index-time).

        ``docs`` is an iterable of text strings (one LLM extraction call per doc, mirroring
        GraphRAG's per-chunk extraction). The generation backend is resolved lazily through the
        provider registry — no SDK import happens unless a real provider is actually called.
        """
        backend = ProviderRegistry.get("generation", provider)
        model = model or load_config().generation.model
        triples: list[tuple[str, str, str]] = []
        for doc in docs:
            text = backend(model, EXTRACTION_SYSTEM, str(doc), max_tokens)
            triples.extend(parse_triples(text))
        return cls.from_triples(triples)

    def global_query(
        self,
        question: str,
        *,
        provider: str = "anthropic",
        model: str = "",
        max_tokens: int = 1024,
    ) -> str:
        """Community/summary-style synthesis answer over the whole graph (Ch 9 Global), lazy LLM.

        Renders the graph's relationships as context and asks the model to synthesize across them —
        the pragmatic stand-in for GraphRAG's pre-computed community reports.
        """
        backend = ProviderRegistry.get("generation", provider)
        model = model or load_config().generation.model
        facts = "\n".join(f"- {s} | {r} | {o}" for s, r, o in self.triples)
        prompt = f"Facts:\n{facts}\n\nQuestion: {question}"
        return backend(model, GLOBAL_SYSTEM, prompt, max_tokens).strip()


def parse_triples(text: str) -> list[tuple[str, str, str]]:
    """Parse ``subject | relation | object`` lines from an LLM extraction response — PURE.

    Tolerant of stray prose, blank lines, and markdown bullets; only well-formed 3-field
    pipe-delimited lines become triples.
    """
    out: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*").strip()
        if line.count("|") != 2:
            continue
        s, r, o = (part.strip() for part in line.split("|"))
        # drop a header row like "subject | relation | object"
        if {s.lower(), r.lower(), o.lower()} == {"subject", "relation", "object"}:
            continue
        if s and r and o:
            out.append((s, r, o))
    return out


__all__ = ["GraphIndex", "parse_triples", "EXTRACTION_SYSTEM", "GLOBAL_SYSTEM"]
