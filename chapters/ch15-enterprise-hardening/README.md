# Chapter 15 — Enterprise Hardening: Security, Access Control, and Multi-Tenancy

> 📖 Companion to **Book Chapter 15: Enterprise Hardening**. Read the chapter, then run these
> top-to-bottom. Everything for this chapter is in this folder — walkthroughs, the experiment, and
> links to the production code. Nothing to hunt for.

## What you'll be able to do after this folder

Give a RAG system the properties it needs to survive a security review — none of which show up in an
offline nDCG@10 sweep, all of which show up in an incident report. You'll trim access *at retrieval
time* so the model never sees a chunk the caller isn't entitled to, isolate tenants with physical
namespaces that fail closed, redact PII on *both* sides (and treat the embedding store as
reconstructible PII), and defend against indirect prompt injection with defense in depth — because no
single control, least of all "a smarter model," is load-bearing.

## Walkthroughs (run in order)

Each notebook narrates one technique and runs live against the sample corpus. The "key" ones are
also exported as plain scripts you can lift directly.

| # | Notebook | Technique (book §) | Production code |
|---|---|---|---|
| 1 | `01_retrieval_acl_trimming.ipynb` · `01_retrieval_acl_trimming.py` | Stamp source ACLs, inject a server-side identity filter at query time (OWASP LLM08:2025) (§ "Access control at retrieval time") | [`ragkit/production/security/`](../../ragkit/production/security/) |
| 2 | `02_multi_tenancy.ipynb` | Physical namespace-per-tenant (fails closed) vs. shared-index + `tenant_id` filter (§ "Multi-tenancy isolation") | [`ragkit/production/security/`](../../ragkit/production/security/) |
| 3 | `03_two_sided_pii.ipynb` · `03_two_sided_pii.py` | Redact before indexing AND re-redact retrieved chunks; embeddings as PII (Vec2Text) (§ "PII, redaction, and data residency") | [`ragkit/production/security/`](../../ragkit/production/security/) |
| 4 | `04_injection_defense_in_depth.ipynb` · `04_injection_defense_in_depth.py` | Spotlighting, hardened model/detector, tool filtering, output controls (OWASP LLM01:2025) (§ "Indirect prompt injection") | [`ragkit/production/security/`](../../ragkit/production/security/) |

## The experiment — `reproduce.py`

```bash
make ch15            # launch the notebooks
python chapters/ch15-enterprise-hardening/reproduce.py   # or just the headline experiment
```

Reproduces the chapter's central claims on the golden set: a **retrieval-time ACL-trimming** test —
the same query run as an entitled and an unentitled caller, confirming the unentitled caller's chunks
never enter the prompt, *and* measuring the recall cost of the security pre-filter (the Chapter 5
filtered-ANN hazard, made visible) — and an **indirect-injection defense-in-depth** test that fires a
poisoned retrieved document at the pipeline with layers off vs. on, reporting attack-success rate (ASR)
as each layer is added (the AgentDojo lesson: tool filtering alone cut targeted ASR 53.1%→7.5%). It
prints a *security* delta (ASR, leaked-chunk count) **and** a *quality/cost* delta (recall under the
filter, added latency), because hardening that destroys retrieval is not hardening you can ship.

## Lift it into your project

```python
from ragkit.production.security import SecureRetriever

secure = SecureRetriever.wrap(retriever, tenant_scope=session.tenant, principal=session.user)
hits = secure.search(query, top_k=50)             # ACL-trimmed, tenant-scoped, PII re-redacted,
                                                  #   injection-screened — none of it optional
```

## Ship-this verdict (from the book)

> Harden a RAG system as five committed properties, not an afterthought: retrieval-time access
> control with a documented staleness window; physical multi-tenant isolation by default; two-sided
> PII redaction with the embedding store treated as reconstructible PII and kept in-region; and
> layered, defense-in-depth injection mitigation in which no single control — least of all "a
> smarter model" — is load-bearing. Every one of these failures is invisible to nDCG@10 and visible
> to an auditor or an attacker; that is exactly why this is the chapter that decides whether the
> system is shippable inside a real enterprise, not merely accurate in a demo.

## Prerequisites

`make setup && make up` (installs `ragkit`, loads sample data, starts Qdrant). The metadata filters
this chapter uses as a *security* primitive are the filtered-ANN mechanics of
[Chapter 5](../ch05-vector-stores/) (and its recall warning); the honest case where retrieved evidence
is a *fact*, not an *instruction*, is [Chapter 13](../ch13-grounded-generation/); and the operational
surface — performance, cost, observability — is [Chapter 16](../ch16-performance-observability/).
