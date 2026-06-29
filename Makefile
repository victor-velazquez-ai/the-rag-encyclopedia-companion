# The RAG Encyclopedia — companion repo task runner.
# Each `make chNN` opens that chapter's walkthroughs; `make reproduce` runs the book's experiments.

.DEFAULT_GOAL := help
.PHONY: help setup up down data reproduce capstone test $(addprefix ch,02 03 04 05 06 07 08 09 10 11 12 13 14 15 16)

help:  ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

up:  ## Start Qdrant locally (Docker), no API keys needed
	docker compose up -d

down:  ## Stop local services
	docker compose down

setup:  ## Install professional_rag_kit + dev deps and load the sample data
	pip install -e ".[dev]"
	$(MAKE) data

data:  ## Download/prepare the sample corpus + golden sets (small, no keys)
	python -m data.prepare

reproduce:  ## Run the "expensive doesn't always win" experiment suite (offline)
	python -m professional_rag_kit.eval.suite --all

capstone:  ## Run the assembled end-to-end RAG pipeline over the sample corpus
	PYTHONPATH=capstone-project python -m rag_capstone.app.run

test:  ## Run the unit tests (no API key needed)
	python -m pytest tests/ -q

# --- Per-chapter walkthroughs (launch the chapter's notebooks) ---------------
ch02:  ## Chapter 2 — Document Processing
	jupyter lab chapters-companion/ch02-document-processing
ch03:  ## Chapter 3 — Chunking
	jupyter lab chapters-companion/ch03-chunking
ch04:  ## Chapter 4 — Embeddings
	jupyter lab chapters-companion/ch04-embeddings
ch05:  ## Chapter 5 — Vector Stores
	jupyter lab chapters-companion/ch05-vector-stores
ch06:  ## Chapter 6 — Retrieval & Routing
	jupyter lab chapters-companion/ch06-retrieval
ch07:  ## Chapter 7 — Reranking
	jupyter lab chapters-companion/ch07-reranking
ch08:  ## Chapter 8 — Context Construction
	jupyter lab chapters-companion/ch08-context-construction
ch09:  ## Chapter 9 — GraphRAG
	jupyter lab chapters-companion/ch09-graphrag
ch10:  ## Chapter 10 — Hierarchical & Adaptive
	jupyter lab chapters-companion/ch10-adaptive-rag
ch11:  ## Chapter 11 — Agentic & Multi-System
	jupyter lab chapters-companion/ch11-agentic-rag
ch12:  ## Chapter 12 — Multimodal
	jupyter lab chapters-companion/ch12-multimodal
ch13:  ## Chapter 13 — Grounded Generation
	jupyter lab chapters-companion/ch13-grounded-generation
ch14:  ## Chapter 14 — Evaluation
	jupyter lab chapters-companion/ch14-evaluation
ch15:  ## Chapter 15 — Enterprise Hardening
	jupyter lab chapters-companion/ch15-enterprise-hardening
ch16:  ## Chapter 16 — Performance & Observability
	jupyter lab chapters-companion/ch16-performance-observability
