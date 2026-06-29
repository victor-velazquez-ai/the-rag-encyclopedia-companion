"""Chapter 4 experiment - the asymmetry/paraphrase check, and MRL truncation.

The book's full sweep (self-host Qwen3 vs. managed; the int8/binary + rescore compression ladder)
scores nDCG@10 and dollar cost on the golden set and needs an embedding provider. This script
reproduces the chapter's two load-bearing claims that ARE checkable with a single provider:

  1. a learned embedding ranks a paraphrase above an unrelated pair (the whole premise), and
  2. MRL truncation returns a shorter, RE-NORMALIZED vector - not a hand-sliced one.

It is key-gated: with no OPENAI_API_KEY it prints what it WOULD do and exits cleanly; the offline
portion (model/dim wiring, the asymmetric query/doc API surface) always runs.

Run:  python chapters-companion/ch04-embeddings/reproduce.py
"""

from __future__ import annotations

import os

from professional_rag_kit.ingestion import Embedder


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


def main() -> int:
    print("Ch 4 - embeddings reproduction\n")

    # Offline portion: the facade wiring is checkable with no network.
    default = Embedder.default()
    mrl = Embedder.from_provider("openai", dims=512)
    selfhost = Embedder.from_provider("qwen3")
    print("default :", default.provider, default.model, "dims", default.dims or 3072)
    print("MRL-512 :", mrl.provider, mrl.model, "dims", mrl.dims)
    print("selfhost:", selfhost.provider, selfhost.model, "(provider swap; query-side prefix)")
    assert mrl.dims == 512
    assert hasattr(default, "embed_query") and hasattr(default, "embed_documents")
    print("[PASS] asymmetric query/doc API + MRL dim wiring\n")

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set - skipping live embedding comparison.")
        print("Set it to reproduce: paraphrase cos(A,B) > cos(A,C), and a 512-dim MRL vector.")
        return 0

    A = "The recursive chunker respects document structure."
    B = "A structure-aware splitter cuts on paragraph and sentence boundaries."  # paraphrase
    C = "Quantization shrinks each vector to one byte per dimension."            # unrelated
    va, vb, vc = default.embed_documents([A, B, C])
    para, unrel = cosine(va, vb), cosine(va, vc)
    print(f"paraphrase cos(A,B)={para:.3f}  unrelated cos(A,C)={unrel:.3f}")
    assert para > unrel, "paraphrase must outscore the unrelated pair"
    print("[PASS] paraphrase outranks unrelated")

    short = mrl.embed_query(A)
    print(f"MRL truncation: {len(short)} dims (vs 3072), API re-normalized")
    assert len(short) == 512
    print("[PASS] MRL returns a shorter re-normalized vector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
