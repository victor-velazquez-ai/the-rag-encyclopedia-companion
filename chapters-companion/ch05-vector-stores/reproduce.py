"""Chapter 5 experiment - the filtered-ANN ACL guarantee on Qdrant.

The book's full sweep (HNSW vs. int8/binary/RaBitQ recall; the filtered-ANN recall collapse vs. an
exhaustive baseline) needs a populated Qdrant and an embedding provider. This script reproduces the
chapter's most consequential single claim that is checkable end-to-end on a tiny corpus: a
security-bearing `allowed_groups` filter applied via FILTERABLE HNSW returns only authorized chunks -
no cross-tenant leak - while the quantize-first-pass-then-rescore path stays on by default.

It is gated on both Qdrant and OPENAI_API_KEY (to embed the chunks). With either missing it prints a
clear message and exits cleanly; the offline portion (schema + store wiring) always runs.

Run:  docker compose up -d   # or `make up`
      python chapters-companion/ch05-vector-stores/reproduce.py
"""

from __future__ import annotations

import os

from professional_rag_kit.core.schema import Chunk
from professional_rag_kit.ingestion import Embedder, VectorStore


def main() -> int:
    print("Ch 5 - vector store reproduction\n")

    chunks = [
        Chunk(id="c1", doc_id="hb", text="Reimbursement requests are filed in the finance portal.",
              allowed_groups=["finance"]),
        Chunk(id="c2", doc_id="hb", text="The on-call rotation is published every Monday.",
              allowed_groups=["engineering"]),
        Chunk(id="c3", doc_id="hb", text="Expense approvals over $5k need a director sign-off.",
              allowed_groups=["finance"]),
    ]

    # Offline portion: wiring is checkable with no Qdrant.
    store = VectorStore.default(collection="ch05_repro", dim=3072, quantization="int8")
    assert store.quantization == "int8" and store.dim == 3072
    print("store wired: filterable HNSW + int8 + rescore (dim 3072)")
    print("[PASS] store/schema wiring\n")

    have_key = bool(os.environ.get("OPENAI_API_KEY"))
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    if not have_key:
        print("OPENAI_API_KEY not set - skipping live index test (need it to embed the chunks).")
        print("Set the key and start Qdrant (`docker compose up -d`) to run the ACL leak test.")
        return 0

    try:
        store = VectorStore.connect(url, collection="ch05_repro", dim=3072, quantization="int8")
    except Exception as e:  # noqa: BLE001 - Qdrant unreachable is an expected, non-crash exit
        print(f"Qdrant not reachable at {url} ({type(e).__name__}).")
        print("Start it with `docker compose up -d` (or `make up`), then re-run.")
        return 0

    embedder = Embedder.default()
    for c, v in zip(chunks, embedder.embed_documents([c.text for c in chunks])):
        c.embedding = v
    store.upsert(chunks)

    qvec = embedder.embed_query("how do I get an expense approved?")
    hits = store.search(qvec, top_k=5, allowed_groups=["finance"])
    leaked = [h for h in hits if "finance" not in h.chunk.allowed_groups]
    for h in hits:
        print(f"  {h.score:.3f}  {h.chunk.allowed_groups}  {h.chunk.text[:40]}")
    assert not leaked, "ACL filter leaked a non-finance chunk (Ch 15 security failure)"
    print("[PASS] filterable-HNSW ACL filter returned only authorized chunks - no leak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
