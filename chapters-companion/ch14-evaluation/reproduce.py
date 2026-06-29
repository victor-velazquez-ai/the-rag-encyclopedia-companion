"""Chapter 14 reproduction — score two retrieval configs on the golden set (offline, no API key).

Runs the harness comparison from `professional_rag_kit.eval.suite`: a body-only BM25 retriever vs. a BM25
body+title RRF fusion, scored with nDCG@k / Recall@k / MRR on the version-controlled golden set,
printing the quality delta. The point is the *method* — name two configs, score them on the same
golden set, read the delta — which extends to the keyed comparisons (chunking, rerankers, graph)
that use real embeddings.

    python chapters-companion/ch14-evaluation/reproduce.py        # or: make reproduce
"""

from professional_rag_kit.eval.suite import main

if __name__ == "__main__":
    main()
