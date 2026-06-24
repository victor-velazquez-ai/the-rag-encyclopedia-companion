"""ragkit.production.observability — tracing, drift detection, feedback loop (Book Ch 16).

You cannot operate what you cannot see. An un-traced RAG pipeline is unoperable, and retrofitting
tracing after an incident is the expensive path — so instrument from day one. The trace is the
substrate every other lever in the chapter measures against.

    tracing.py     OpenTelemetry GenAI semantic conventions (vendor-neutral `gen_ai.*`, with
                   `retrieval` a first-class operation) — pre-release, so PIN A VERSION and expect
                   churn — plus OpenInference's RAG-aware span kinds (RETRIEVER -> RERANKER -> LLM,
                   with flattened `retrieval.documents.{i}.{id,score,content}`). The flattening is
                   what lets a trace answer: retrieval failure (doc never fetched) or generation
                   failure (fetched and ignored)?
    drift.py       three-surface drift detection — document/embedding, data/content, query-
                   distribution — the one failure invisible per request and visible only in
                   aggregate. PSI (0.1 / 0.2 thresholds) for scalar signals, but NEVER univariate KS
                   on embeddings (oversensitive at scale, blind to joint structure): use centroid-
                   cosine drift, MMD, or a classifier-based detector against a reference window
                   established NOW, while the baseline still exists.
    feedback.py    the production loop as plumbing — thumbs -> annotation queue -> eval dataset, so
                   every real-world failure becomes a permanent regression gate on the next deploy.
                   Online LLM-as-judge on a sample (TruLens RAG Triad: context relevance,
                   groundedness, answer relevance) for continuous quality; a distilled GUARDRAIL span
                   as a real-time inline gate where a failure must never reach the user. The judge's
                   biases and calibration are `ragkit.eval` (Ch 14); here it is the wiring.

Phase-1 scaffold: the surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class Tracer:
#     """Emits OTel GenAI + OpenInference spans for the whole RAG request tree; the substrate every
#     other production lever measures against."""
#     @classmethod
#     def default(cls) -> "Tracer": ...
#     def span(self, kind: str):   # "RETRIEVER" | "RERANKER" | "LLM" | "GUARDRAIL" | ...
#         """Context manager opening one typed span in the request trace."""
#         ...
# class DriftMonitor:
#     """Three-surface drift against a reference window — centroid-cosine / MMD / classifier, never
#     univariate KS on embeddings."""
#     @classmethod
#     def default(cls) -> "DriftMonitor": ...

__all__ = ["Tracer", "DriftMonitor"]  # populated in Phase 2
