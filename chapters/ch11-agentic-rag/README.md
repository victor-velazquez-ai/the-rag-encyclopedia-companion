# Chapter 11 — Agentic and Multi-System RAG

> 📖 Companion to **Book Chapter 11: Agentic and Multi-System RAG**. Read the chapter, then run
> these top-to-bottom. Everything for this chapter is in this folder — walkthroughs, the experiment,
> and links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Keep three layers most teams conflate cleanly separate: *loop* (agentic multi-hop) only for the
compositional minority, gated by a complexity classifier and capped at 3–5 hops; *wrap* (MCP) only
when sources or consumers exceed one; *resolve* (canonical-ID entity resolution) *always* once the
corpus crosses source systems. You'll build a turn-budgeted plan-act-reflect loop with Self-Ask /
IRCoT inside it, wrap sources behind one MCP boundary, and stand up the entity-resolution crosswalk
(Splink) that makes "multi-source retrieval" mean anything at all.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_plan_act_reflect_loop.ipynb` · `01_plan_act_reflect_loop.py` | Plan-act-reflect loop, turn-capped, gated by the complexity classifier (§ "Agentic RAG") | [`ragkit/architectures/agentic/loop.py`](../../ragkit/architectures/agentic/) |
| 2 | `02_self_ask_ircot.ipynb` · `02_self_ask_ircot.py` | Multi-hop — Self-Ask (default) and IRCoT, interleave retrieval with reasoning (§ "Multi-hop retrieval") | [`ragkit/architectures/agentic/multi_hop.py`](../../ragkit/architectures/agentic/) |
| 3 | `03_mcp_tool_boundary.ipynb` | MCP — wrap each source as a server (Tools / Resources / Prompts), runtime discovery (§ "The MCP tool boundary") | [`ragkit/architectures/agentic/mcp.py`](../../ragkit/architectures/agentic/) |
| 4 | `04_entity_resolution_splink.ipynb` · `04_entity_resolution_splink.py` | Entity resolution — deterministic-then-Fellegi-Sunter, Splink, canonical IDs (§ "Entity resolution and the canonical layer") | [`ragkit/architectures/agentic/entity.py`](../../ragkit/architectures/agentic/) |
| 5 | `05_resolve_entity_tool.ipynb` | Normalize queries to canonical IDs *before* retrieval (the `resolve_entity` MCP tool) (§ "The ER pipeline and the resolve_entity tool") | [`ragkit/architectures/agentic/entity.py`](../../ragkit/architectures/agentic/) |

## The experiment — `reproduce.py`

```bash
make ch11            # launch the notebooks
python chapters/ch11-agentic-rag/reproduce.py   # or just the headline experiment
```

Reproduces this chapter's master-map comparisons: **single-shot vs. agentic multi-hop** on
compositional questions, and the **entity-resolution** completeness lift. The multi-hop run reports
not just final-answer correctness but *per-hop intermediate accuracy* (the chain's real ceiling is
the product of its hops), hop-count distribution, and non-termination rate — and licenses the loop
on *correctness-lift-per-dollar over single-shot on the routed subset*, never aggregate accuracy. The
ER run measures cross-source retrieval **completeness** with the canonical layer on vs. off, tracking
the false-merge rate explicitly because over-merging's cost is asymmetric.

## Lift it into your project

```python
from ragkit.architectures.agentic import AgenticRAG, EntityResolver

# resolve first — normalize entity mentions to canonical IDs BEFORE retrieving:
resolver = EntityResolver.from_crosswalk("configs/crosswalk.parquet")
canonical = resolver.resolve("ACME Corporation")        # -> the one canonical ID for all four "Acme"s

# then loop only on the compositional minority (turn-capped, classifier-gated):
agent = AgenticRAG.default(max_hops=5)
answer = agent.run("which 2024-acquired subsidiary is HQ'd in the same state as the largest competitor?")
```

## Ship-this verdict (from the book)

> Keep the three layers distinct and reach for each only on its trigger. *Loop* (agentic multi-hop)
> only for the compositional minority, gated by the complexity classifier and capped by a turn
> budget. *Wrap* (MCP) only when sources or consumers exceed one. *Resolve* (canonical-ID entity
> resolution) *always* when the corpus crosses source systems — this is the differentiator most RAG
> systems skip and the one that decides whether multi-source retrieval is coherent or fragmented. The
> senior posture is not "make it agentic"; it is "detect the small fraction that needs orchestration,
> and build the canonical substrate that the orchestration sits on."

## Prerequisites

`make setup && make up` (installs `ragkit`, loads sample data, starts Qdrant). The complexity
classifier that gates the loop is [Chapter 10](../ch10-adaptive-rag/); the graph whose nodes this
chapter's entity resolution deduplicates is [Chapter 9](../ch09-graphrag/); the access-control and
prompt-injection surface that the MCP tool boundary dramatically widens is
[Chapter 15](../ch15-enterprise-hardening/).
