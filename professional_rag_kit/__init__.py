"""professional_rag_kit — the production-grade, all-open RAG library behind *The RAG Encyclopedia*.

The library mirrors the book's four Parts, plus a shared eval harness and a thin core of
contracts. Import the component you need; every default is the verdict-recommended, self-hostable
choice from its chapter, with managed providers available as one-line swaps.

    core          configuration, the chunk schema, the provider registry  (cross-cutting)
    ingestion     parsing · chunking · embedding · indexing                (Part I,  Ch 2–5)
    retrieval     hybrid+RRF · query · routing · rerank · context          (Part II, Ch 6–8)
    architectures graph · hierarchical · adaptive · agentic · multimodal    (Part III, Ch 9–12)
    production    generation · security · serving · observability           (Part IV, Ch 13,15,16)
    eval          metrics · golden-set runner · judge calibration          (Ch 14)

Phase-1 scaffold: public surfaces are declared here; implementations land in Phase 2.
"""

__version__ = "0.1.0"
