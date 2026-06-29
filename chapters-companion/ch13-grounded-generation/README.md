# Chapter 13 — Grounded Generation and Citation

> 📖 Companion to **Book Chapter 13: Grounded Generation and Citation**. Read the chapter, then run
> these top-to-bottom. Everything for this chapter is in this folder — walkthroughs, the experiment,
> and links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Turn retrieved context into an answer that is *faithful* to that context, *cites* where each claim
came from span by span, and *refuses* when the evidence does not support a confident answer. You'll
run the four-layer defense in order — context-faithful prompting (free, default), Chain-of-Note for
noisy retrieval, Context-Aware Decoding to override a stale parametric prior, span-level citation
scored ALCE-style, first-class abstention, and source-reliability arbitration for conflicting
evidence — and you'll measure *faithfulness and correctness separately*, because the technique that
raises one can quietly damage the other.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_context_faithful_prompting.ipynb` · `01_context_faithful_prompting.py` | Opinion-based framing + counterfactual demonstrations — the free default (§ "Context-Faithful Prompting") | [`professional_rag_kit/production/generation/`](../../professional_rag_kit/production/generation/) |
| 2 | `02_chain_of_note.ipynb` | Per-document reading notes before answering — the noisy-retrieval filter, +7.9 EM (§ "Chain-of-Note") | [`professional_rag_kit/production/generation/`](../../professional_rag_kit/production/generation/) |
| 3 | `03_context_aware_decoding.ipynb` · `03_context_aware_decoding.py` | CAD — training-free contrastive decoding, +14.3% factuality (§ "Context-Aware Decoding (CAD)") | [`professional_rag_kit/production/generation/`](../../professional_rag_kit/production/generation/) |
| 4 | `04_span_citation.ipynb` · `04_span_citation.py` | Span-level citation scored ALCE-style (recall + precision) via an NLI judge (§ "Citation and attribution") | [`professional_rag_kit/production/generation/`](../../professional_rag_kit/production/generation/) |
| 5 | `05_abstention.ipynb` | Sufficiency gate → first-class "the sources do not support an answer" output (§ "Abstention as a first-class design choice") | [`professional_rag_kit/production/generation/`](../../professional_rag_kit/production/generation/) |
| 6 | `06_conflict_arbitration.ipynb` | Source-reliability weighting + surface-the-disagreement for conflicting evidence (§ "Conflicting and insufficient evidence") | [`professional_rag_kit/production/generation/`](../../professional_rag_kit/production/generation/) |

## The experiment — `reproduce.py`

```bash
make ch13            # launch the notebooks
python chapters-companion/ch13-grounded-generation/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's central claims on the golden set: **grounding on vs. off** (context-faithful
prompting + Chain-of-Note + CAD vs. a naive "answer using the context" prompt) and the chapter's most
important probe — a **counterfactual context test** that feeds deliberately wrong context and confirms
CAD now follows it, quantifying your *retrieval-quality dependency* made visible. It prints a *quality*
delta (faithfulness, answer correctness, ALCE citation recall/precision) **and** a *cost* delta (added
tokens/latency for the note step and CAD's second forward pass), because the chapter's whole point is
that *faithfulness is not correctness* — every technique that makes the model trust the context more
makes it trust bad context more too, and you decide with both numbers in front of you.

## Lift it into your project

```python
from professional_rag_kit.production.generation import GroundedGenerator

gen = GroundedGenerator.default()                 # context-faithful prompting + span citation + abstention
answer = gen.generate(query, passages)            # GroundedAnswer: text, citations, abstained, conflict
if answer.abstained:
    ...                                           # the sources did not support a confident answer
```

## Ship-this verdict (from the book)

> Build grounded generation as a four-layer stack, not a single setting: **(1)** ground with
> Context-Faithful Prompting by default, Chain-of-Note for noise, CAD for stale-prior override on
> self-hosted models; **(2)** make it checkable with span-level citation, measured by ALCE recall/
> precision and trained with AGREE, judged by NLI read as a low-recall lower bound; **(3)** catch
> residual hallucination with an inline fine-tuned detector plus offline FActScore, and ship
> *abstention* as a first-class output evaluated under a hallucination-penalized scheme; **(4)** handle
> conflict with source-reliability weighting, disclosure, and abstention. Above all, hold
> *faithfulness and correctness as separate goals* and measure them separately — the technique that
> raises one can lower the other, and CAD's +14.3% factuality and its amplification of noisy context
> are the same mechanism. The generator must fail honestly; that, more than any single method, is what
> makes a RAG system trustworthy enough to ship.

## Prerequisites

`make setup && make up` (installs `professional_rag_kit`, loads sample data, starts Qdrant). The context that feeds
this generator comes from [Chapter 8](../ch08-context-construction/); the *adaptive-retrieval* side of
grounding (Self-RAG, CRAG, FLARE) is [Chapter 10](../ch10-adaptive-rag/); how you *measure*
faithfulness and citation at scale (RAGAS, the harness, CI gates) is [Chapter 14](../ch14-evaluation/);
and injection — the retrieved document that carries an *instruction* rather than a fact — is
[Chapter 15](../ch15-enterprise-hardening/).
