# Chapter 14 — Evaluation

> 📖 Companion to **Book Chapter 14: Evaluation**. Read the chapter, then run these top-to-bottom.
> Everything for this chapter is in this folder — walkthroughs, the experiment, and links to the
> production code. Nothing to hunt for.

## What you'll be able to do after this folder

Build the instrument the rest of the book depends on — the one that *licenses complexity*, because no
technique earns its place until a number on a held-out set moves and stays moved. You'll separate
retrieval failure from generation failure (they are orthogonal), score retrieval with nDCG@k / Recall@k
/ MRR and generation with faithfulness / relevance / correctness *kept apart*, run RAGAS reference-free,
calibrate an LLM judge around its position / verbosity / self-enhancement biases, manufacture a golden
set when you don't have one, and wire the whole thing into a CI gate that blocks regressions.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_orthogonality.ipynb` · `01_orthogonality.py` | Stage-wise attribution — score retrieval and generation separately (§ "The orthogonality principle") | [`professional_rag_kit/eval/`](../../professional_rag_kit/eval/) |
| 2 | `02_retrieval_metrics.ipynb` | Recall@k, MRR, nDCG@k — at your *real* context depth (§ "Retrieval metrics: Recall@k, MRR, nDCG@k") | [`professional_rag_kit/eval/`](../../professional_rag_kit/eval/) |
| 3 | `03_generation_metrics.ipynb` | Faithfulness, answer relevance, answer correctness — three separate axes (§ "Generation metrics") | [`professional_rag_kit/eval/`](../../professional_rag_kit/eval/) |
| 4 | `04_ragas.ipynb` · `04_ragas.py` | RAGAS reference-free scoring — the production default (§ "RAGAS: the reference-free production default") | [`professional_rag_kit/eval/`](../../professional_rag_kit/eval/) |
| 5 | `05_llm_judge_calibration.ipynb` · `05_llm_judge_calibration.py` | Position-swap gating, cross-family judge, PPI confidence intervals (§ "LLM-as-judge: biases and calibration") | [`professional_rag_kit/eval/`](../../professional_rag_kit/eval/) |
| 6 | `06_synthetic_eval_set.ipynb` | RAGAS knowledge-graph evolutions + ARES — manufacture an eval set (§ "Synthetic eval-set generation") | [`professional_rag_kit/eval/`](../../professional_rag_kit/eval/) |
| 7 | `07_eval_driven_dev.ipynb` | Golden set, CI eval gate, offline-then-online loop (§ "Eval-driven development: the loop") | [`professional_rag_kit/eval/`](../../professional_rag_kit/eval/) |

## The experiment — `reproduce.py`

```bash
make ch14            # launch the notebooks
python chapters-companion/ch14-evaluation/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's spine claims on the golden set: the **orthogonality demonstration** — scoring
the same answers end-to-end vs. with separate retrieval and generation numbers, showing how a single
accuracy figure hides *which half* is broken (the FRAMES 0.40→0.66 no-retrieval-to-retrieval delta in
miniature, isolating how much "correctness" was parametric) — and an **LLM-judge calibration** pass that
runs position-swap consistency gating and prints how often raw verdicts flip when nothing changes but
order. It prints a *quality* number (nDCG@k, faithfulness) **and** a *cost* number (judge calls per
metric per query), because even the harness must price itself; this is the chapter that makes every
other verdict in the book falsifiable on your own corpus.

## Lift it into your project

```python
from professional_rag_kit.eval import Harness, GoldenSet

golden = GoldenSet.load("default")                # version-controlled queries + ground truth
card = Harness.default().run(my_pipeline, golden) # retrieval + generation scored SEPARATELY, plus cost
# gate a merge on it: block if card.ndcg_at_10 or card.faithfulness regresses past threshold
```

## Ship-this verdict (from the book)

> Build evaluation as the first-class instrument of your RAG program, not an afterthought: separate
> retrieval from generation failure, measure each on metrics that match the job (nDCG@k, Recall@k;
> faithfulness, relevance, correctness), default to RAGAS for automated scoring, calibrate every
> LLM judge before you trust it, synthesize and spot-check an eval set when you lack one, and wire
> it all into a golden set with a CI gate that blocks regressions. This is the discipline that lets
> you deploy frontier methods because they *work* — proven on your own data — rather than because
> they impress. Measurement licenses complexity; this chapter is how you do the measuring.

## Prerequisites

`make setup && make up` (installs `professional_rag_kit`, loads sample data, starts Qdrant). This chapter is the
spine: it scores the grounded generation of [Chapter 13](../ch13-grounded-generation/) and every
retrieval verdict back through Parts II–III; the *production* instrumentation that feeds the loop —
tracing, drift detection, the thumbs→annotation→eval-gate pipeline — is
[Chapter 16](../ch16-performance-observability/). Run `make reproduce` to re-run the whole book's
head-to-heads through this harness.
