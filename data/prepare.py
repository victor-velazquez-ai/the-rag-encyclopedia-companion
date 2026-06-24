"""Prepare / verify the sample corpora + golden sets (Book companion).

The starter corpus (``corpus_small``) and golden set (``golden/qa.jsonl``) are small and committed
to the repo, so `make data` just confirms they're present and reports their size — nothing to
download, no paid key. Swap in your own documents and golden set the moment you want a real answer
(the book's whole point is to measure on *your* corpus). Larger/multimodal datasets land in Phase 2+.
"""

from __future__ import annotations

import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent


def _count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    corpus = _DATA / "corpus_small" / "docs.jsonl"
    golden = _DATA / "golden" / "qa.jsonl"
    missing = [p for p in (corpus, golden) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing sample data: {missing}. Re-clone the repo.")

    # sanity-check the JSONL parses
    for line in golden.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)

    print(f"Sample data ready (no download needed):")
    print(f"  corpus_small : {_count(corpus)} documents  ({corpus.relative_to(_DATA)})")
    print(f"  golden/qa    : {_count(golden)} queries    ({golden.relative_to(_DATA)})")
    print("\nRun the offline reproduction with:  make reproduce")


if __name__ == "__main__":
    main()
