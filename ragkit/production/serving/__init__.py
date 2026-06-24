"""ragkit.production.serving — throughput, latency, and cost levers (Book Ch 16).

The generation stage dominates the latency budget (1-5 s) and the bill (retrieved context is most
of the input tokens), so every lever here points at it. Tune the *dominant* term against a written
per-stage budget, set the SLA on the tail (P99, not the mean), and make the expensive option earn
its place per query.

    batching.py    vLLM-style continuous (in-flight) batching + PagedAttention — admit/retire at the
                   token level so the GPU stays saturated; pages the KV cache like virtual memory.
                   ~2-4x throughput, ~55% less wasted KV-cache VRAM on RAG's long-prompt/variable-
                   output shape. Pair with CHUNKED PREFILL so larger batches do not inflate TTFT and
                   break the P99 promise. (Self-hosted only; a managed API makes this the vendor's job.)
    prompt_cache.py  the biggest single cost lever in RAG. The one architectural rule: STABLE PREFIX
                   FIRST (system instructions + recurring reference docs), PER-QUERY SUFFIX LAST
                   (the user question + per-request chunks) — get the order backwards and you cache
                   nothing. Provider discounts are volatile (re-pull live, dated): cache reads ~0.1x.
    routing.py     model/effort routing + cascades — FrugalGPT-style escalate-on-quality-signal or
                   RouteLLM-style upfront routing. Most queries are easy; ~45-85% cost cut at ~95%
                   quality in practice (treat the 98% headline as a ceiling). A *cost* decision about
                   which generator — retrieval routing (which index/strategy) is Ch 10's `routing/`.
    cache.py       semantic cache, vCache-style. Ship only at a measured hit rate of 30%+ (below ~15%
                   it is not worth the surface); budget 2-10x on a hit, never the marketing 100x. The
                   threshold is a CORRECTNESS control, not a tuning knob: a single static value sits
                   in a grey zone where correct and incorrect hits overlap, so use a LEARNED,
                   PER-PROMPT, error-bounded threshold (vCache, arXiv:2502.03771) and pair with a TTL
                   plus event-based invalidation where provenance is traceable.

Phase-1 scaffold: the surfaces are sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class Router:
#     """Per-query model/effort routing (RouteLLM-style) or escalate-on-signal cascade (FrugalGPT)."""
#     @classmethod
#     def default(cls) -> "Router": ...
#     def route(self, query: str, context: list) -> str:   # -> model id / effort tier
#         ...
# class SemanticCache:
#     """Embedding-proximity cache with a learned, error-bounded per-prompt threshold (vCache) + TTL."""
#     @classmethod
#     def default(cls, *, target_error: float = 0.02, ttl_seconds: int | None = None) -> "SemanticCache": ...
#     def get(self, query: str): ...        # returns cached answer only inside the error bound
#     def put(self, query: str, answer) -> None: ...

__all__ = ["Router", "SemanticCache"]  # populated in Phase 2
