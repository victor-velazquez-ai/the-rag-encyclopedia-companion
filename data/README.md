# data — sample corpora + golden sets

Small, openly-licensed data so **every walkthrough and experiment runs out of the box, with no
paid API key and no huge download.** `make data` (or `python -m data.prepare`) fetches and prepares
everything below.

> 🚧 Phase 1: this is the spec + dataset cards. `prepare.py` and the prepared files land in Phase 2.

## What's here (Phase 2)

| Dataset | What it is | Used by |
|---|---|---|
| `corpus_small/` | ~1–5k openly-licensed documents (docs + a few visually-rich PDFs) | most chapters |
| `corpus_multisystem/` | the same entities under different names across mock CRM/ERP/finance exports | Ch 11 (entity resolution) |
| `golden/qa.jsonl` | queries with honest relevance judgments + reference answers | the eval harness (Ch 14) |
| `golden/multihop.jsonl` | multi-hop questions whose evidence spans 2–4 documents | Ch 10/11 |
| `pdfs/` | a handful of charts/tables/scanned pages | Ch 2, Ch 12 |

## Why small and open
The book's claim is that you should reproduce its comparisons **on your own corpus**. The sample
data exists only to make the machinery run end-to-end on a laptop; swap in your own documents and
golden set the moment you want a real answer. See [`docs/HOW-TO-USE.md`](../docs/HOW-TO-USE.md).

## Licensing
Each subfolder ships a `CARD.md` with the source and license of its data. Nothing here requires
acceptance of restrictive terms; anything with attribution requirements is noted in its card.
