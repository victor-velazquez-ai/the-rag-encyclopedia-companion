# Chapter 4 — Embeddings

> 📖 Companion to **Book Chapter 4: Embeddings**. Read the chapter, then run these top-to-bottom.
> Everything for this chapter is in this folder — walkthroughs, the experiment, and links to the
> production code. Nothing to hunt for.

## What you'll be able to do after this folder

Make the single most consequential retrieval decision deliberately instead of by accident. You'll
build the embedding stage as five separable, measured choices — model, query/document asymmetry,
dimensionality (Matryoshka), quantization, and domain fine-tuning — and know the exact cost of each,
including the silent failure modes (a prefix mismatch, a truncation without re-normalization) that
quietly sink recall everywhere.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_model_selection.ipynb` · `01_model_selection.py` | Self-host Qwen3 vs. managed; the MTEB/licensing trap (§ "Model selection") | [`ragkit/ingestion/embedding/encoders.py`](../../ragkit/ingestion/embedding/) |
| 2 | `02_instruction_prefixes.ipynb` | Instruction-tuned asymmetric encoding; prefix mismatch (§ "Instruction-tuned and asymmetric encoders") | [`ragkit/ingestion/embedding/instruct.py`](../../ragkit/ingestion/embedding/) |
| 3 | `03_matryoshka.ipynb` · `03_matryoshka.py` | MRL truncation + mandatory re-normalization (§ "Matryoshka Representation Learning") | [`ragkit/ingestion/embedding/matryoshka.py`](../../ragkit/ingestion/embedding/) |
| 4 | `04_quantization_rescore.ipynb` · `04_quantization_rescore.py` | int8 / binary quantization + float rescore (§ "Quantization: 4× to 32× smaller") | [`ragkit/ingestion/embedding/quantize.py`](../../ragkit/ingestion/embedding/) |
| 5 | `05_late_interaction.ipynb` | Multi-vector late interaction — ColBERTv2 / ColPali (§ "Multi-vector late interaction") | [`ragkit/ingestion/embedding/encoders.py`](../../ragkit/ingestion/embedding/) |
| 6 | `06_finetune_hard_negatives.ipynb` | Domain fine-tune, positive-aware hard-negative mining (§ "Domain fine-tuning") | [`ragkit/ingestion/embedding/finetune.py`](../../ragkit/ingestion/embedding/) |

## The experiment — `reproduce.py`

```bash
make ch04            # launch the notebooks
python chapters/ch04-embeddings/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's central claim on the golden set: **self-host Qwen3 vs. a managed embedder**,
and **int8/binary + rescore vs. full float** — printing a *quality* delta (nDCG@10, recall@k) **and**
a *cost* delta (\$/1M tokens or GPU-seconds, bytes/vector, search latency). The point of the chapter
in one table: the highest MTEB number is rarely the right model (read the license before the score —
NV-Embed-v2's 72.31 is unshippable), and quantization-plus-rescore stacked with MRL truncation
(~128× smaller) is the largest lever on vector-store cost at near-float quality.

## Lift it into your project

```python
from ragkit.ingestion import Embedder

embedder = Embedder.default()                          # Qwen3-Embedding-8B, Apache-2.0, int8+rescore
qvec = embedder.embed_query("your question")           # applies the query instruction prefix
dvecs = embedder.embed_documents(chunks)               # applies the doc prefix — never mix the two

# managed swap, MRL-truncated — one line, no re-embedding ecosystem change:
embedder = Embedder.from_provider("voyage", dims=512)  # needs VOYAGE_API_KEY
```

## Ship-this verdict (from the book)

> Build the embedding stage as five separable, measured decisions — model, asymmetry, dimension,
> quantization, fine-tuning — not one model pick. The defaults that ship in 2026: Qwen3-8B
> (self-host) or Gemini/Voyage (managed); instruction-tuned asymmetric encoding with shared
> prefixes; MRL truncation re-normalized; int8-then-float rescore stacked with MRL for ~128×
> compression; and positive-aware fine-tuning only when the domain earns it. Verify each on
> nDCG@10 against your own held-out queries — the leaderboard screens, your data decides.

## Prerequisites

`make setup && make up` (installs `ragkit`, loads sample data, starts Qdrant). Embeddings turn the
chunks from [Chapter 3](../ch03-chunking/) into vectors; the store that indexes and searches them at
scale is [Chapter 5](../ch05-vector-stores/).
