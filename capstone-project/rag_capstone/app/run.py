"""Run the assembled capstone pipeline over the sample corpus.

    python -m rag_capstone.app.run                 # answers a few demo questions
    python -m rag_capstone.app.run "your question"  # answers your own

With ANTHROPIC_API_KEY (or OPENAI_API_KEY) set, the answer is synthesized and cited by Claude/GPT
through the grounded generator. With no key, it falls back to the offline extractive answerer so the
whole pipeline still runs end-to-end. Either way the retrieval → fold flow is identical.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rag_capstone.app.pipeline import ExtractiveAnswerer, RAGPipeline

# Resolve the repo-root `data/` dir by walking up — robust to where this package nests.
_DATA = next(
    (p / "data" for p in Path(__file__).resolve().parents if (p / "data" / "corpus_small").is_dir()),
    Path(__file__).resolve().parents[3] / "data",
)

DEMO_QUESTIONS = [
    "refund thirty days",
    "ERR_CONN_4021 connection",
    "rate limit 429",
    "reset forgotten password",
    "what is the airspeed velocity of an unladen swallow",  # not in corpus → should abstain
]


def _load_docs() -> list[tuple[str, str]]:
    path = _DATA / "corpus_small" / "docs.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(r["id"], r["text"]) for r in rows]


def _pick_generator():
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        from professional_rag_kit.production.generation import GroundedGenerator

        return GroundedGenerator.from_config(), "Claude/GPT (grounded)"
    return ExtractiveAnswerer(), "offline extractive (set ANTHROPIC_API_KEY for synthesis)"


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    docs = _load_docs()
    generator, mode = _pick_generator()
    pipe = RAGPipeline.from_corpus(docs, generator, top_k=3)

    print(f"Capstone RAG pipeline — generator: {mode}\n")
    questions = argv or DEMO_QUESTIONS
    for q in questions:
        ans = pipe.answer(q)
        tag = "ABSTAINED" if ans.abstained else f"cites {ans.citations}"
        print(f"Q: {q}\nA: {ans.text}\n   [{tag}]\n")


if __name__ == "__main__":
    main()
