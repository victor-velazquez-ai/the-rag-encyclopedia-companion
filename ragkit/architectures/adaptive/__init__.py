"""ragkit.architectures.adaptive — composable runtime control policies (Book Ch 10).

Flat RAG retrieves the same way every time: always, once, and trusting whatever came back. The four
policies here are not rival systems — they are guards over *different decision points* on the same
pipeline (before retrieval, during generation, after retrieval, after generation), so they stack.
Each ships as a *pattern* — a control graph over the frontier model you already run — not as the
papers' 7B–13B checkpoints; the policy is the durable asset, the model is yours.

    adaptive_rag.py   Adaptive-RAG (arXiv:2403.14403) — route by query complexity to
                      no-retrieval / single-step / multi-step. The cost optimizer; add it first when
                      cost is the dominant failure (near-full accuracy at ~half the steps).
    crag.py           CRAG (arXiv:2401.15884) — grade the retrieval, then correct it: refine
                      (knowledge strips) / web-search fallback / both. Add when retrieval is the weak link.
    self_rag.py       Self-RAG (arXiv:2310.11511) — segment-level support critique (retrieve-on-demand,
                      grade relevance, grade support, re-draft). Add when grounding is the priority.
    flare.py          FLARE (arXiv:2305.06983) — retrieve on the model's own low-confidence spans
                      mid-generation. Add for long-form answers whose information needs surface as you write.

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class AdaptiveRouter:
#     """Composable control layer. `AdaptiveRouter.default()` → complexity routing (Adaptive-RAG)."""
#     @classmethod
#     def default(cls) -> "AdaptiveRouter": ...
#     def with_policy(self, name: str) -> "AdaptiveRouter":   # "crag" | "self_rag" | "flare"
#         """Attach a guard to its decision point; policies compose left-to-right along the pipeline."""
#         ...
#     def run(self, query: str) -> dict:
#         """Route, retrieve, (optionally) grade/critique/re-retrieve, and generate."""
#         ...

__all__ = ["AdaptiveRouter"]  # populated in Phase 2
