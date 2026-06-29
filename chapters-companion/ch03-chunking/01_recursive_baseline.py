# %% [markdown]
# # Chapter 3 — The recursive chunking baseline
#
# The book's finding: a *tuned ~200-token, structure-aware recursive splitter* is a strong baseline
# that the expensive methods (semantic, late, contextual) must beat **on your data**. This walkthrough
# runs that baseline from `professional_rag_kit` — no API key, no GPU.
#
# Production code: [`professional_rag_kit/ingestion/chunking`](../../professional_rag_kit/ingestion/chunking/__init__.py).

# %%
from professional_rag_kit.ingestion import Chunker

DOC = (
    "Retrieval-augmented generation grounds a language model in retrieved text. "
    "The first design decision is how to cut documents into chunks. "
    "A chunk that is too large dilutes the embedding; one that is too small severs a fact "
    "from the context that makes it answerable. "
    "The recursive splitter respects structure: it splits on paragraph, then sentence, then word "
    "boundaries, packing to a target size with an overlap so a fact split across a boundary still "
    "appears whole in one chunk. "
    "Semantic chunking, which splits at embedding-distance boundaries, often loses to this baseline "
    "on real, well-structured documents — so we start here and make the expensive method prove itself."
)

# %% [markdown]
# ## Split with the default (~200 tokens, ~15% overlap)
# `Chunker.default()` is the verdict baseline. Here we use a small target so the short demo doc
# splits into several chunks you can inspect.

# %%
chunker = Chunker(target_tokens=30, overlap_tokens=6)
chunks = chunker.split(DOC, doc_id="rag-intro")

print(f"{len(chunks)} chunks\n")
for c in chunks:
    print(f"[{c.id}]  ({len(c.text.split())} tok)  {c.text[:70]}...")

# %% [markdown]
# ## Two things every chunk carries
# 1. **Metadata** — `doc_id` (parent lookup + citation), a chunk index, the strategy. Non-negotiable
#    per Ch 3; downstream chapters depend on it (citation in Ch 13, access control in Ch 15).
# 2. **Overlap** — consecutive chunks share a tail, so a fact at a boundary is never orphaned.

# %%
first, second = chunks[0], chunks[1]
print("metadata:", second.doc_id, second.section_path, second.tags)

shared = set(first.text.split()) & set(second.text.split())
print("\noverlap words shared by chunk 0 and 1:", sorted(shared))
assert shared, "expected a non-empty overlap tail"

# %% [markdown]
# ## The escalation rule
# The default is cheap and strong. Escalating to an LLM-budget method is a measured decision — the
# chapter's headline win is **Anthropic Contextual Retrieval** (prepend per-chunk context before
# indexing), which cut top-20 retrieval failures by up to 67%. Reach for it past the ~200k-token
# threshold; below that line, don't build RAG at all — put the corpus in the prompt.
#
# Run the reproduction to see semantic-vs-recursive on the golden set: `python reproduce.py`.

# %%
print("\nbaseline ready. Next: embeddings (Ch 4) turn these chunks into vectors.")
