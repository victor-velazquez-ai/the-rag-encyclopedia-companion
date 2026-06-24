"""Chapter 6 reproduction — does hybrid fusion beat a single leg? (offline, no API key)

Scores body-only BM25 against a BM25 body+title RRF fusion on the golden set, printing the quality
delta with a cost column. Fusion recovers queries whose vocabulary lives only in one field — the same
mechanism by which adding a dense leg recovers paraphrase that BM25 misses. The dense-vs-hybrid and
HyDE-on-strong-vs-weak comparisons need embeddings (a key); the method is identical — add the leg,
re-score, read the delta. This delegates to the shared reproduction suite (ragkit.eval.suite).

    python chapters/ch06-retrieval/reproduce.py
"""

from ragkit.eval.suite import main

if __name__ == "__main__":
    main()
