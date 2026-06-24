"""ragkit.architectures.agentic — multi-hop orchestration across hops and sources (Book Ch 11).

Three layers most RAG content conflates, kept deliberately separate here. *Loop* (agentic
multi-hop) only for the compositional minority, gated by the complexity classifier and capped by a
turn budget. *Wrap* (MCP) only when sources or consumers exceed one. *Resolve* (canonical-ID entity
resolution) *always* once the corpus crosses source systems — it is the precision substrate under
multi-source and graph RAG, not an optional nicety.

    loop.py        plan-act-reflect (planner-executor / ReAct) loop with a hard turn budget (3–5
                   hops) and a termination check; bounds error-accumulation and non-termination.
    multi_hop.py   Self-Ask (arXiv:2210.03350, explicit auditable sub-questions, the default) and
                   IRCoT (arXiv:2212.10509, free-form chain-of-thought steers retrieval) — interleave
                   retrieval with reasoning; normalize entities per sub-question before retrieving.
    mcp.py         Model Context Protocol source wrappers — one client-host-server boundary (Tools /
                   Resources / Prompts over JSON-RPC) so the agent discovers sources at runtime.
    entity.py      EntityResolver — the canonical-ID layer. Deterministic/exact match first, then
                   Fellegi-Sunter probabilistic (Splink, unsupervised EM, the default; dedupe for
                   active learning). Blocking → matching → clustering → canonical IDs; exposed as a
                   `resolve_entity` tool that normalizes queries to canonical IDs *before* retrieval.

Phase-1 scaffold: the facade surfaces are sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class AgenticRAG:
#     """Plan-act-reflect loop, gated by the Adaptive-RAG complexity classifier and turn-capped."""
#     @classmethod
#     def default(cls, max_hops: int = 5) -> "AgenticRAG": ...
#     def run(self, query: str) -> dict:
#         """Decompose → retrieve → reflect (enough? budget?) → loop or synthesize."""
#         ...
#
# class EntityResolver:
#     """Canonical-ID crosswalk. Deterministic-then-Fellegi-Sunter; Splink default, dedupe optional."""
#     @classmethod
#     def from_crosswalk(cls, path: str) -> "EntityResolver": ...
#     def resolve(self, mention: str) -> str:
#         """Map a surface form to its canonical ID (exact lookup first, fuzzy fallback)."""
#         ...

__all__ = ["AgenticRAG", "EntityResolver"]  # populated in Phase 2
