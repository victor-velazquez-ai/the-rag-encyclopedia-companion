# %% [markdown]
# # Chapter 7 - Reranking: the cheap-recall -> precise-rerank cascade
#
# A first stage casts a wide, cheap net (BM25); an expensive model re-scores only the survivors.
# This is the production shape of retrieval: recall cheaply, then spend on precision over a
# shortlist. This walkthrough runs OFFLINE - we register a tiny deterministic generation backend so
# the listwise LLM reranker runs with NO API key. With a real key it is Claude/GPT via
# `Reranker.default()`.
#
# Production code: [`professional_rag_kit/retrieval/rerank`](../../professional_rag_kit/retrieval/rerank/__init__.py).
# Book section: Ch 7 (reranking tiers; the cascade).

# %%
from professional_rag_kit.core.config import ProviderRegistry
from professional_rag_kit.retrieval.hybrid import BM25
from professional_rag_kit.retrieval.rerank import Reranker

# %% [markdown]
# ## Stage 1 - cheap recall with BM25
# BM25 is the brutally strong lexical baseline. It is fast and runs over the whole corpus, so it is
# the right tool to cast a wide net and pull a generous candidate pool. But it ranks by lexical
# overlap, so a doc that merely repeats query terms can outrank the one that actually answers the
# question - exactly the kind of ordering error a reranker exists to fix.

# %%
CORPUS = [
    ("d1", "A reranker model scores relevance; a reranker improves relevance ranking of results."),
    ("d2", "Our search model indexes documents and returns the top results for a query."),
    ("d3", "A cross-encoder re-scores a shortlist and reorders it so the best answer lands first."),
    ("d4", "Reranking is the precise second stage that fixes a noisy first-stage candidate order."),
]
bm = BM25(CORPUS)

query = "how does a reranker improve relevance"
candidate_ids = bm.rank(query, top_k=4)
id_to_text = dict(CORPUS)
candidates = [id_to_text[cid] for cid in candidate_ids]
print("BM25 first-stage order:", candidate_ids)
for cid in candidate_ids:
    print(f"  {cid}: {id_to_text[cid]}")

# %% [markdown]
# ## Make the LLM reranker runnable offline
# `Reranker(provider="llm")` calls the generation backend with a listwise prompt and parses a
# comma-separated reorder like `3,1,2` out of the reply. To run with NO key, we register a tiny
# deterministic backend under the name `demo` that returns a fixed reorder. (The numbers are
# 1-based passage indices into the shortlist; the module converts and de-duplicates them.)

# %%
ProviderRegistry.register("generation", "demo")(
    lambda model, system, prompt, maxtok: "2,4,1,3"
)
reranker = Reranker(provider="llm", gen_provider="demo", gen_model="x")

# %% [markdown]
# ## Stage 2 - precise rerank over the shortlist
# The reranker re-scores only the survivors and promotes the genuinely explanatory passages. BM25
# put d1 first purely on keyword overlap ("reranker... relevance... reranker... relevance"); the
# demo reorder `2,4,1,3` promotes the two passages that actually explain reranking (candidates 2 and
# 4) and demotes the keyword-stuffed d1 and the off-topic search-model doc - the precision the cheap
# first stage could not deliver.

# %%
reranked = reranker.rerank(query, candidates, top_k=4)
for i, passage in enumerate(reranked, 1):
    print(f"rank {i}: {passage}")

# %% [markdown]
# ## Cost is the headline caveat
# Reranking is the highest-leverage quality lever AND, very often, the most expensive latency stage:
# in production retrieval the reranker is frequently 60-84% of total retrieval-pipeline latency. The
# candidate count is your main cost knob - cost scales linearly with how many documents the
# expensive model sees, so the first stage must hand it a tight shortlist, not the whole corpus.

# %%
print("WARNING: the reranker is often 60-84% of retrieval-pipeline latency.")
print("Cost knob = candidate count. Pull deep, rerank shallow.")
print("Shortlist handed to the reranker:", len(candidates), "candidates")

# %% [markdown]
# ## With a real key
# Swap the demo backend for a real provider and the same code calls Claude/GPT listwise:
#
# ```python
# # export ANTHROPIC_API_KEY=...   (or OPENAI_API_KEY + GEN_PROVIDER=openai)
# reranker = Reranker.default()              # LLM listwise, reuses your generation provider
# reranked = reranker.rerank(query, candidates, top_k=10)
# ```
#
# Cohere Rerank is a one-line swap (`Reranker.from_provider("cohere")`); the self-host cross-encoder
# (jina-reranker) is the `[selfhost]` tier. Next: assemble the reranked shortlist into context
# (Ch 8).
print("\nNext: assemble the reranked shortlist into context (Ch 8).")
