"""Chapter 3 reproduction — does chunk size change retrieval quality? (offline, no API key)

Chunks the sample corpus at two granularities with the recursive baseline, retrieves with BM25 over
the chunks (mapping each retrieved chunk back to its document), and scores recall on the golden set.
A real, key-free measurement of the chapter's point: the chunk-size knob has a measurable effect, so
tune it on YOUR data. The semantic-vs-recursive and Contextual-Retrieval ablations need embeddings/an
LLM (a key) — the method is identical, just swap the chunker and re-score.

    python chapters/ch03-chunking/reproduce.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ragkit.eval import GoldenSet, Harness
from ragkit.ingestion.chunking import Chunker
from ragkit.retrieval.hybrid import BM25

_DATA = Path(__file__).resolve().parents[2] / "data"


def _retriever_for(target_tokens: int):
    docs = [json.loads(l) for l in (_DATA / "corpus_small" / "docs.jsonl").read_text("utf-8").splitlines() if l.strip()]
    chunker = Chunker(target_tokens=target_tokens, overlap_tokens=max(2, target_tokens // 8))
    chunks, chunk_to_doc = [], {}
    for d in docs:
        for ch in chunker.split(d["text"], doc_id=d["id"]):
            chunks.append((ch.id, ch.text))
            chunk_to_doc[ch.id] = d["id"]
    bm = BM25(chunks)

    def retrieve(query: str) -> list[str]:
        seen, docs_ranked = set(), []
        for chunk_id in bm.rank(query, top_k=20):
            doc = chunk_to_doc[chunk_id]
            if doc not in seen:  # collapse chunks back to documents, best-first
                seen.add(doc)
                docs_ranked.append(doc)
        return docs_ranked

    return retrieve, len(chunks)


def main() -> None:
    golden = GoldenSet.from_jsonl(_DATA / "golden" / "qa.jsonl")
    h = Harness.default()
    print(f"\nChunk-size sweep on the golden set (n={len(golden)}, recall@5):\n")
    for target in (15, 40, 120):
        retrieve, n_chunks = _retriever_for(target)
        card = h.score_retrieval(f"recursive ~{target} tok", retrieve, golden, k=5)
        print(f"  {card.name:<22} ({n_chunks:>2} chunks)  recall@5 {card.recall:.3f}  nDCG@5 {card.ndcg:.3f}")
    print("\n  Chunk size is a knob with a measured effect — sweep it on your corpus, don't guess.")
    print("  Semantic vs. recursive and Contextual Retrieval need a key; same method, swap the chunker.\n")


if __name__ == "__main__":
    main()
