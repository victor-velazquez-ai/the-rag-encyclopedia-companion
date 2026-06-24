# Chapter 2 — Document Processing and Parsing

> 📖 Companion to **Book Chapter 2: Document Processing and Parsing**. Read the chapter, then run
> these top-to-bottom. Everything for this chapter is in this folder — walkthroughs, the experiment,
> and links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Turn a PDF, scan, or office document into *faithful, structured text* — and defend the choice
between classical OCR, a pure VLM, and a detector-first hybrid in a design review. You'll parse with
the right architecture for your corpus, serialize tables so merged cells survive, and handle the
chapter's most dangerous failure mode — a VLM confidently reading the wrong number off a chart —
before it poisons your index.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_hybrid_parsing.ipynb` · `01_hybrid_parsing.py` | Detector-first hybrid parse — MinerU 2.5 / PaddleOCR-VL (§ "The three architectures") | [`ragkit/ingestion/parsing/hybrid.py`](../../ragkit/ingestion/parsing/) |
| 2 | `02_pure_vlm.ipynb` | Pure end-to-end VLM parse — dots.ocr (§ "The shipping models, compared") | [`ragkit/ingestion/parsing/vlm.py`](../../ragkit/ingestion/parsing/) |
| 3 | `03_classical_ocr.ipynb` | Zero-hallucination classical OCR (§ "The three architectures") | [`ragkit/ingestion/parsing/classical.py`](../../ragkit/ingestion/parsing/) |
| 4 | `04_reading_order.ipynb` | Layout-aware reading order on multi-column pages (§ "Layout-aware parsing and reading order") | [`ragkit/ingestion/parsing/hybrid.py`](../../ragkit/ingestion/parsing/) |
| 5 | `05_tables_to_html.ipynb` · `05_tables_to_html.py` | Table extraction + HTML serialization, TEDS (§ "Table extraction and serialization") | [`ragkit/ingestion/parsing/tables.py`](../../ragkit/ingestion/parsing/) |
| 6 | `06_charts_and_figures.ipynb` | Chart/figure surrogates — the untrusted-number warning (§ "Figures, charts, and formulas") | [`ragkit/ingestion/parsing/figures.py`](../../ragkit/ingestion/parsing/) |

## The experiment — `reproduce.py`

```bash
make ch02            # launch the notebooks
python chapters/ch02-document-processing/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's headline comparison on the sample documents: **VLM vs. classical OCR vs.
hybrid parsing** — printing a *quality* delta (OmniDocBench-style overall + table TEDS, in-cell
accuracy on the hardest tables) **and** a *cost* delta (pages/sec, GPU-seconds or \$/1k pages, and
emitted-token count). The point of the chapter in one table: hybrid wins the broad composite, pure
VLM buys operational simplicity, classical OCR buys zero hallucination — and you pick by your
corpus's *hardest* pages and your cost structure, not by a leaderboard rank.

## Lift it into your project

```python
from ragkit.ingestion.parsing import parse

doc = parse("report.pdf")                       # detector-first hybrid (MinerU 2.5), tables as HTML
doc = parse("report.pdf", backend="classical")  # zero-hallucination — legal/regulatory/finance
doc = parse("report.pdf", backend="vlm")         # dots.ocr — one model, no cascade
```

## Ship-this verdict (from the book)

> Choose your parser by your corpus's *hardest* pages and your *cost structure*, not by a single
> leaderboard rank. Hybrid (MinerU 2.5 / PaddleOCR-VL) is the production default; pure VLM
> (dots.ocr) for simplicity; DeepSeek-OCR for token-bound scale; classical OCR where
> zero-hallucination is a legal requirement. Always serialize tables as HTML, always measure
> in-cell accuracy on finance/clinical data, and never let an unverified VLM chart number become a
> fact. Get this stage right and everything downstream has a chance; get it wrong and no reranker,
> no embedding, and no clever prompt can recover what the parser silently lost.

## Prerequisites

`make setup && make up` (installs `ragkit`, loads sample data, starts Qdrant). This is the first
stage of the pipeline — the faithful, structured text it emits is what
[Chapter 3](../ch03-chunking/) splits into chunks.
