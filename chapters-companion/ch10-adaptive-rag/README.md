# Chapter 10 — Hierarchical, Adaptive, and Self-Correcting RAG

> 📖 Companion to **Book Chapter 10: Hierarchical, Adaptive, and Self-Correcting RAG**. Read the
> chapter, then run these top-to-bottom. Everything for this chapter is in this folder —
> walkthroughs, the experiment, and links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Shape retrieval to the document and steer it with the query. You'll build the cheap parent-child
hierarchy that belongs in every system, escalate to a RAPTOR collapsed-tree only when long-document
synthesis earns it, and wire the four adaptive policies — Adaptive-RAG, CRAG, Self-RAG, FLARE — as
composable guards over your own frontier model, adding exactly the one that owns your dominant
failure.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_parent_child.ipynb` · `01_parent_child.py` | Parent-child / small-to-big / auto-merging — the default hierarchy (§ "Parent-child, small-to-big, and auto-merging") | [`professional_rag_kit/architectures/hierarchical/parent_child.py`](../../professional_rag_kit/architectures/hierarchical/) |
| 2 | `02_raptor_collapsed_tree.ipynb` · `02_raptor_collapsed_tree.py` | RAPTOR — UMAP+GMM summary tree, collapsed-tree retrieval (§ "RAPTOR") | [`professional_rag_kit/architectures/hierarchical/raptor.py`](../../professional_rag_kit/architectures/hierarchical/) |
| 3 | `03_adaptive_rag_routing.ipynb` · `03_adaptive_rag_routing.py` | Adaptive-RAG — route by complexity to no/single/multi-step (§ "Adaptive-RAG") | [`professional_rag_kit/architectures/adaptive/adaptive_rag.py`](../../professional_rag_kit/architectures/adaptive/) |
| 4 | `04_crag.ipynb` | CRAG — grade retrieval, then correct (refine / web fallback / both) (§ "CRAG") | [`professional_rag_kit/architectures/adaptive/crag.py`](../../professional_rag_kit/architectures/adaptive/) |
| 5 | `05_self_rag.ipynb` | Self-RAG — segment-level support critique as a pattern (§ "Self-RAG") | [`professional_rag_kit/architectures/adaptive/self_rag.py`](../../professional_rag_kit/architectures/adaptive/) |
| 6 | `06_flare.ipynb` | FLARE — retrieve on low-confidence spans mid-generation (§ "FLARE") | [`professional_rag_kit/architectures/adaptive/flare.py`](../../professional_rag_kit/architectures/adaptive/) |
| 7 | `07_composing_the_guards.ipynb` | Composing all four as guards on one pipeline (§ "The synthesis") | [`professional_rag_kit/architectures/adaptive/`](../../professional_rag_kit/architectures/adaptive/) |

## The experiment — `reproduce.py`

```bash
make ch10            # launch the notebooks
python chapters-companion/ch10-adaptive-rag/reproduce.py   # or just the headline experiment
```

Reproduces this chapter's master-map comparisons: the **RAPTOR collapsed-tree** lift on
long-document synthesis questions (vs. a flat baseline, collapsed-tree vs. tree-traversal), and the
**Adaptive / CRAG / Self-RAG / FLARE** policies measured the way the book demands — each on the
*delta it produces*. Adaptive-RAG is plotted as accuracy-per-step (near-full accuracy at ~half the
steps); CRAG as a lift on top of a base system (the Self-CRAG deltas); Self-RAG as a grounding delta;
FLARE as a θ-sweep of quality vs. retrievals-per-answer. Quality **and** cost, every time.

## Lift it into your project

```python
from professional_rag_kit.architectures.hierarchical import HierarchicalRetriever
from professional_rag_kit.architectures.adaptive import AdaptiveRouter

retriever = HierarchicalRetriever.default()        # parent-child / small-to-big (the default)
retriever = HierarchicalRetriever.raptor()         # escalate for long-document synthesis

# add one guard at a time — diagnose the dominant failure, then attach the policy that owns it:
router = AdaptiveRouter.default().with_policy("crag")   # complexity routing + grade-and-correct
answer = router.run("your query")
```

## Ship-this verdict (from the book)

> Build hierarchy and adaptive control as *separable, measured additions*, not a monolith. Default to
> parent-child hierarchy; escalate to RAPTOR-collapsed only for long-document synthesis. For runtime
> control, diagnose your dominant failure and add exactly one policy — Adaptive-RAG for cost, CRAG for
> weak retrieval, Self-RAG for grounding, FLARE for long-form needs — each as a *pattern over your
> frontier model*, each verified by the delta it produces on your own traffic. The committed posture:
> retrieval shaped to the document and steered by the query, with every layer of cleverness forced to
> prove it beats the flat baseline it replaces.

## Prerequisites

`make setup && make up` (installs `professional_rag_kit`, loads sample data, starts Qdrant). The flat retrieval
these policies steer is [Chapter 6](../ch06-retrieval/); the graph-based answer to overlapping global
and multi-hop problems is [Chapter 9](../ch09-graphrag/); agentic orchestration that takes the
steering further — multi-hop planning across systems, behind the same complexity gate — is
[Chapter 11](../ch11-agentic-rag/).
