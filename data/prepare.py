"""Prepare the sample corpora + golden sets (Book companion).

Phase 2 fetches the openly-licensed sample documents, builds the small corpora described in
README.md (corpus_small, corpus_multisystem, pdfs), and writes the golden Q/A sets used by the
eval harness (Ch 14). Everything is small enough to run on a laptop with no paid API key.

Phase-1 scaffold: this entry point exists so `make data` / `python -m data.prepare` resolve.
"""


def main() -> None:
    print("Phase 2 — data preparation not yet implemented. See data/README.md for the spec.")


if __name__ == "__main__":
    main()
