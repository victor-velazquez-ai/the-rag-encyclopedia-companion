# %% [markdown]
# # Chapter 16 — Performance and observability: route, cache, watch for drift
#
# The generation stage dominates both the latency budget (1-5 s) and the bill (retrieved context is
# most of the input tokens), so every cost lever points at it: send a cheaper model where the query
# allows (routing), skip the call entirely on a near-duplicate (semantic cache). And you cannot
# *operate* what you cannot see - so we also watch a scalar signal for distribution drift (PSI). Three
# pure levers, run offline here.
#
# Production code:
# [`ragkit.production.serving`](../../ragkit/production/serving/__init__.py) (`ModelRouter`,
# `SemanticCache`) and
# [`ragkit.production.observability`](../../ragkit/production/observability/__init__.py)
# (`psi`, `psi_alert`). Book section 16: "Performance and observability".

# %%
from ragkit.production.observability import psi, psi_alert
from ragkit.production.serving import ModelRouter, SemanticCache, lookup

# %% [markdown]
# ## Model routing - the strong tier must earn its place
# Most RAG queries are easy and a small model nails them; a minority need the frontier tier. Sending
# everything to the strong model pays frontier prices for questions the cheap model would have
# answered. `ModelRouter` scores complexity (length, multi-part structure, reasoning cues) and routes
# accordingly - the upfront-router shape (no extra model call). Practitioners report 45-85% cost cuts
# at ~95% quality.

# %%
router = ModelRouter()  # cheap=claude-haiku-4-5, strong=claude-opus-4-8, threshold=1.0

simple = "What is the capital of France?"
complex_q = "Compare the trade-offs between dense and sparse retrieval and explain when each wins."

print(f"simple  -> {router.route(simple)}   (complexity {router.complexity(simple):.2f})")
print(f"complex -> {router.route(complex_q)}   (complexity {router.complexity(complex_q):.2f})")
assert router.route(simple) == router.cheap_model
assert router.route(complex_q) == router.strong_model

# %% [markdown]
# ## Semantic cache - a hit skips retrieval, rerank, AND the LLM call
# The cache lives in front of the whole pipeline: a hit on a near-duplicate query returns the cached
# answer for free. `lookup` is pure cosine proximity over `[(vec, response), ...]`, returning the best
# entry's response iff its similarity clears the threshold. We use synthetic vectors so it runs with
# no embedding key.

# %%
cached = [
    ([1.0, 0.0, 0.0], "Paris is the capital of France."),     # ~ "capital of France"
    ([0.0, 1.0, 0.0], "The mitochondrion is the powerhouse."),  # unrelated
]

near_dup = [0.96, 0.10, 0.0]   # very close to the first cached vector -> HIT
unrelated = [0.10, 0.10, 0.99]  # close to nothing cached       -> MISS

hit = lookup(near_dup, cached, threshold=0.85)
miss = lookup(unrelated, cached, threshold=0.85)
print("near-duplicate query ->", hit)
print("unrelated query      ->", miss)
assert hit == "Paris is the capital of France."
assert miss is None

# %% [markdown]
# ## The vCache caveat - the threshold is a correctness control, not a tuning knob
# A single *static* similarity threshold has **no correctness guarantee**. The similarity distribution
# of pairs where the cached answer is still correct *overlaps* the distribution where it is wrong - a
# grey zone one global number cannot separate. Set the threshold too low and the cache serves a
# confidently-wrong answer. The production fix is a learned, per-prompt, error-bounded threshold
# (vCache, arXiv:2502.03771), paired with a TTL and event-based invalidation (right in March, wrong in
# June). `SemanticCache` wraps `lookup` with lazy embedding - it needs a key only on a real `get`/`put`.

# %%
print("SemanticCache default threshold:", SemanticCache().threshold, "(static baseline; see vCache)")
print("budget the speedup at 2-10x on a hit, never the marketing 100x; ship at 30%+ hit rate.")

# %% [markdown]
# ## Drift detection - PSI on a scalar signal
# `psi` (Population Stability Index) measures how far a current sample has drifted from a reference,
# over quantile buckets. No shift scores ~0; a real distribution shift crosses the operating
# thresholds: < 0.1 none, 0.1-0.2 moderate (investigate), > 0.2 significant (act). Use it on a *single
# scalar* (retrieval score, answer length, latency) - NEVER per-dimension on embeddings, where it is
# oversensitive at scale and blind to joint structure (use centroid-cosine / MMD there). PSI wants a
# real sample, not a handful of points, so the quantile buckets are populated - we draw seeded
# Gaussians (think: yesterday's vs. today's retrieval scores).

# %%
import random

rng = random.Random(0)
reference = [rng.gauss(0.0, 1.0) for _ in range(2000)]            # the baseline distribution
no_shift = [rng.gauss(0.0, 1.0) for _ in range(2000)]            # drawn from the same distribution
big_shift = [rng.gauss(1.0, 1.0) for _ in range(2000)]           # mean shifted right by one sigma

psi_none = psi(reference, no_shift)
psi_big = psi(reference, big_shift)
print(f"no shift   PSI={psi_none:.4f}  band={psi_alert(psi_none)}")
print(f"big shift  PSI={psi_big:.4f}  band={psi_alert(psi_big)}")
assert psi_none < 0.1 and psi_alert(psi_none) == "none"
assert psi_big > 0.2 and psi_alert(psi_big) == "significant"

# %% [markdown]
# ## The lesson
# Route to make the call cheaper, cache to skip it, and trace/PSI to know when the world moved under
# you. Every cost lever is charged against the quality the earlier chapters bought: the cache
# threshold is a correctness control, the router must hold quality at its cut, and drift is the signal
# that yesterday's good config is today's regression.

# %%
print("\nroute + cache for cost, PSI for drift - but never spend the quality the pipeline earned.")
