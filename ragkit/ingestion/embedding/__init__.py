"""ragkit.ingestion.embedding — text (and pages) into meaning-bearing vectors (Book Ch 4).

The single most consequential retrieval decision, and the one most often made by accident — you
inherit a model's dimensionality, license, and asymmetry for the life of the index, and changing
it means re-embedding everything. The chapter builds the stage as five *separable, measured*
choices, not one model pick: model, asymmetry, dimensionality (MRL), quantization, fine-tuning.

    encoders.py    model registry: Qwen3-Embedding-8B (Apache-2.0) self-host default; Gemini/Voyage managed
    instruct.py    instruction-tuned asymmetric encoding — query vs. doc prefixes, pinned in shared code
    matryoshka.py  MRL truncation: store full width, truncate on read, ALWAYS re-normalize after
    quantize.py    int8 + float rescore (default win); binary + rescore when memory binds; never binary raw
    finetune.py    domain fine-tune with positive-aware hard-negative mining (NV-Retriever TopK-PercPos)

One ``Embedder`` facade. The defaults are the chapter's verdicts: self-host → Qwen3-Embedding-8B
(the open SOTA you can legally ship — NV-Embed-v2's 72.31 is CC-BY-NC, a trap not a target);
managed → Gemini for balance, Voyage for price. The two failure modes the facade designs out:
a *prefix mismatch* (index/query prefixes must match — recall drops silently, nothing errors), and
*MRL truncation without re-normalization* (cosine assumes unit vectors). int8+rescore is on by
default; stack MRL × binary for ~128× compression, the largest lever on vector-store cost.

Phase-1 scaffold: the facade's surface is sketched below; implementations land in Phase 2.
"""

# --- Phase-2 target (spec) ----------------------------------------------------
# class Embedder:
#     """Facade over the embedding stack. `Embedder.default()` → Qwen3-Embedding-8B, self-host."""
#     @classmethod
#     def default(cls) -> "Embedder":
#         """Qwen3-Embedding-8B (Apache-2.0), instruction-tuned, int8+rescore on."""
#         ...
#     @classmethod
#     def from_provider(cls, provider: str, *, dims: int | None = None) -> "Embedder":
#         # "qwen3" | "gemini" | "voyage"  ; dims truncates via MRL (re-normalized)
#         ...
#     def embed_query(self, text: str): ...      # applies the QUERY instruction prefix (core)
#     def embed_documents(self, texts: list): ... # applies the DOC prefix; never mix the two

__all__ = ["Embedder"]  # populated in Phase 2
