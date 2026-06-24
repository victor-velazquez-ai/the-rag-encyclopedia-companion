"""Chapter 2 experiment — VLM vs. classical OCR vs. hybrid parsing.

Reproduces the chapter's headline comparison on the sample documents. Parses the same hardest
pages (multi-column, dense financial tables, charts) three ways — detector-first hybrid
(MinerU 2.5), pure end-to-end VLM (dots.ocr), and zero-hallucination classical OCR — and prints
a quality delta (OmniDocBench-style overall + table TEDS + in-cell accuracy) alongside a cost
delta (pages/sec, GPU-seconds or $/1k pages, emitted-token count). The verdict it makes runnable:
hybrid wins the broad composite, pure VLM buys simplicity, classical buys zero hallucination —
you choose by your corpus's hardest pages and your cost structure, not by a leaderboard rank.

Phase 2 wires this to ragkit.ingestion.parsing and the shared eval harness.
"""

if __name__ == "__main__":
    print("Phase 2 — see README.md")
