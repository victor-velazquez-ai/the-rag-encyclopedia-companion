"""ragkit.production.security — access control, multi-tenancy, PII, injection defense (Book Ch 15).

The properties a RAG system needs to survive an enterprise security review — none of which show up
in an offline metric, all of which show up in an incident report. The committed posture is defense
in depth: no single control is load-bearing, least of all "a smarter model" (more-capable models
are often *more* vulnerable to injection).

    access.py      retrieval-time security trimming — stamp each chunk's source ACL at index time,
                   inject the caller's resolved identity as a SERVER-SIDE, non-optional filter at
                   query time. The leak happens the instant an unentitled chunk enters the prompt
                   (OWASP LLM08:2025), so trim before the model sees it — never filter citations
                   after. Owns the ACL re-sync staleness window: drive it from permission-change
                   events, document the worst-case window. (Native token-based ACL/RBAC is the swap.)
    tenancy.py     multi-tenant isolation — physical namespace/partition per tenant by default
                   (fails closed: a coding mistake breaks one tenant's search, it does not leak
                   everyone's). Shared-index + tenant_id filter only at high cardinality, with the
                   scope auto-injected at one chokepoint and unscoped queries rejected outright.
    pii.py         two-sided redaction (Presidio-style): before indexing AND re-redact retrieved
                   chunks at query time — the retrieved side is the half teams forget and the half
                   that leaks (OWASP LLM02:2025). Treats the embedding store as reconstructible PII
                   (Vec2Text recovers up to 92% of a short input): same residency + access class as
                   the source text; noise injection / quantization raise inversion cost, keep recall.
    injection.py   indirect-injection defense-in-depth — the threat retrieval uniquely introduces
                   (OWASP LLM01:2025). Layers data/instruction separation (spotlighting), an
                   alignment-hardened model or inline detector (StruQ/SecAlign line), least-privilege
                   tools + human-in-the-loop (tool filtering cut targeted ASR 53.1% -> 7.5% in
                   AgentDojo — the highest-leverage single control), and output exfiltration controls.

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class SecureRetriever:
#     """Wraps a retriever with the Ch 15 controls: ACL-trimmed, tenant-scoped, PII-redacted,
#     injection-screened — none of them optional."""
#     @classmethod
#     def wrap(cls, retriever, *, tenant_scope: str, principal) -> "SecureRetriever": ...
#     def search(self, query: str, top_k: int = 50) -> list:
#         """Retrieve only chunks the principal is entitled to, within their tenant namespace,
#         with retrieved-side PII re-redacted and injection-screened before return."""
#         ...

__all__ = ["SecureRetriever"]  # populated in Phase 2
