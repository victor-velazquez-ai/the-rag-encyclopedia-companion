# %% [markdown]
# # Chapter 11 — Entity resolution: the canonical-ID layer
#
# Most "agentic RAG" content sells the *loop* (plan-act-reflect, multi-hop). The loop is the minority
# path; the differentiator that actually pays for a multi-source corpus is duller and runs on *every*
# query: **entity resolution**. The same company is `Acme Inc` in the CRM, `Acme, Inc.` in contracts,
# and `acme` in support tickets. Without a canonical id, retrieval fragments the entity across systems
# and silently under-recalls. This walkthrough builds that crosswalk offline.
#
# The matcher runs the chapter's non-negotiable ordering: **deterministic / exact first, then
# fuzzy / probabilistic** for the remainder — a Fellegi-Sunter-style stack with a *review band* for
# the uncertain middle.
#
# Production code: [`professional_rag_kit.architectures.agentic`](../../professional_rag_kit/architectures/agentic/__init__.py)
# (`EntityResolver`). Book section 11: "Entity resolution and the canonical layer".

# %%
from professional_rag_kit.architectures.agentic import EntityResolver, normalize

# Build the crosswalk: one canonical id, many surface forms. The id itself is always an alias.
resolver = EntityResolver()
resolver.add_canonical("acme", "Acme Inc", "Acme, Inc.")
resolver.add_canonical("ibm", "IBM", "International Business Machines")

print("canonical ids:", resolver.canonical_ids)

# %% [markdown]
# ## Stage 1 - deterministic / exact match
# Normalization does the heavy lifting *before* any fuzzy scoring: lowercase, strip punctuation, drop
# legal suffixes (`Inc.`, `Corp.`, `GmbH`), collapse whitespace. Many "fuzzy" cases become exact ones.
# So `acme inc`, `ACME Corporation`, and `  acme  ` all collapse to the same key and resolve for free.

# %%
print("normalize('Acme, Inc.')   ->", repr(normalize("Acme, Inc.")))
print("normalize('ACME')         ->", repr(normalize("ACME")))

res = resolver.resolve_detail("acme inc")
print("resolve('acme inc')       ->", res.canonical_id, "| method:", res.method, "| score:", res.score)
assert res.method == "deterministic"

# %% [markdown]
# ## Stage 2 - fuzzy / probabilistic match
# What deterministic leaves unresolved goes to a similarity score (`difflib.SequenceMatcher`). The
# classic case: an acronym that differs only in spacing/dots. `I.B.M.` normalizes to `i b m`, which is
# not an exact alias of `ibm` - but the fuzzy stage compares the despaced forms and scores 1.0, so it
# resolves. This is the "probabilistic" half of deterministic-then-Fellegi-Sunter.

# %%
res = resolver.resolve_detail("I.B.M.")
print("resolve('I.B.M.')         ->", res.canonical_id, "| method:", res.method, "| score:", round(res.score, 3))
assert res.canonical_id == "ibm"
assert res.method == "fuzzy"

# %% [markdown]
# ## A clean miss
# An unrelated mention scores below the review floor and returns no id. The resolver fails *closed*:
# a low-confidence guess that silently merges two different companies is worse than no merge at all,
# because over-merging is the asymmetric error (you cannot un-leak a contract attached to the wrong
# entity).

# %%
res = resolver.resolve_detail("Globex Corporation")
print("resolve('Globex Corp.')   ->", res.canonical_id, "| method:", res.method, "| score:", round(res.score, 3))
assert res.canonical_id is None

# %% [markdown]
# ## The review band - the three-way decision
# Fellegi-Sunter is not match/non-match; it is match / **review** / non-match. Ratios in
# `[review_low, threshold)` are returned as *uncertain*: the resolver surfaces the candidate but does
# not commit, routing the case to a human. We tighten the band here so a near-but-not-exact mention
# lands in it. The convenience `resolve()` returns `None` for review-band hits - an uncertain match
# is not a decision.

# %%
strict = EntityResolver(threshold=0.92, review_low=0.70)
strict.add_canonical("acme", "Acme Inc")

res = strict.resolve_detail("Acne Inc")  # one-character typo: close, but not confident
print("resolve_detail('Acne Inc')->", res.canonical_id, "| method:", res.method,
      "| score:", round(res.score, 3), "| review:", res.review)
print("resolve('Acne Inc')       ->", strict.resolve("Acne Inc"), "(None: review is not a decision)")
assert res.review is True
assert strict.resolve("Acne Inc") is None

# %% [markdown]
# ## Why this is the substrate, not a nicety
# `EntityResolver` is the precision layer *under* multi-source retrieval, graph RAG, and the agentic
# loop. The plan-act-reflect loop (`AgenticRAG`, also in this module) needs a real generation key to
# run; its call shape is:
#
# ```python
# from professional_rag_kit.architectures.agentic import AgenticRAG
# agent = AgenticRAG.default(retriever=my_retriever, generator=my_generator)  # both injected
# result = agent.run("Which firm did Acme's CEO found before Acme?")          # decompose->retrieve->reflect
# ```
#
# Normalize entities first (this notebook), and *then* let the loop chain hops - each sub-question
# retrieves against canonical ids instead of fragmented surface forms.

# %%
print("\nentity resolution runs on every query; the loop runs on the compositional minority.")
print("normalize -> exact -> fuzzy -> review: the canonical-ID layer is the precision substrate.")
