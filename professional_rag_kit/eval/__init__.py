"""professional_rag_kit.eval — the evaluation harness: the book's spine, made runnable (Book Ch 14).

*Measurement licenses complexity.* No technique earns a place because it is sophisticated — only
because a number on a held-out set moved, and stayed moved, when you turned it on. This module
produces that number. The one non-negotiable conceptual move: retrieval failure and generation
failure are orthogonal, so the harness scores RETRIEVAL separately against its own ground truth.

``GoldenSet`` is the version-controlled query set (queries + relevant-doc ground truth). ``Harness``
runs a retriever over it and returns a ``Scorecard`` — mean nDCG@k / Recall@k / MRR plus a cost
number, because the book's argument is never quality alone. Generation-side metrics (faithfulness,
answer relevance) and LLM-judge calibration are Phase 2+; retrieval scoring is implemented here.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from professional_rag_kit.eval.metrics import mrr, ndcg_at_k, recall_at_k


@dataclass
class GoldenItem:
    query: str
    relevant: set[str]
    gains: dict[str, float] = field(default_factory=dict)  # graded relevance; defaults to 1.0 each
    reference_answer: str | None = None

    def gain_map(self) -> dict[str, float]:
        return self.gains or {d: 1.0 for d in self.relevant}


@dataclass
class GoldenSet:
    items: list[GoldenItem]

    def __iter__(self) -> Iterator[GoldenItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def from_items(cls, rows: list[dict]) -> "GoldenSet":
        return cls(
            [
                GoldenItem(
                    query=r["query"],
                    relevant=set(r["relevant"]),
                    gains={k: float(v) for k, v in r.get("gains", {}).items()},
                    reference_answer=r.get("reference_answer"),
                )
                for r in rows
            ]
        )

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "GoldenSet":
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls.from_items(rows)


@dataclass
class Scorecard:
    name: str
    n: int
    ndcg: float
    recall: float
    mrr: float
    cost_usd: float = 0.0

    def row(self) -> str:
        return (f"{self.name:<28} nDCG@k {self.ndcg:.3f}  Recall@k {self.recall:.3f}  "
                f"MRR {self.mrr:.3f}  ${self.cost_usd:.4f}  (n={self.n})")


class Harness:
    """Runs a retriever over a GoldenSet and scores retrieval quality (Ch 14)."""

    @classmethod
    def default(cls) -> "Harness":
        return cls()

    def score_retrieval(
        self,
        name: str,
        retriever: Callable[[str], list[str]],
        golden: GoldenSet,
        *,
        k: int = 5,
        cost_usd: float = 0.0,
    ) -> Scorecard:
        """Score a retriever (``query -> ranked doc ids``) on the golden set, averaged over queries."""
        n = len(golden)
        if n == 0:
            return Scorecard(name, 0, 0.0, 0.0, 0.0, cost_usd)
        s_ndcg = s_recall = s_mrr = 0.0
        for item in golden:
            ranked = retriever(item.query)
            s_ndcg += ndcg_at_k(ranked, item.gain_map(), k)
            s_recall += recall_at_k(ranked, item.relevant, k)
            s_mrr += mrr(ranked, item.relevant)
        return Scorecard(name, n, s_ndcg / n, s_recall / n, s_mrr / n, cost_usd)

    def compare(self, named_retrievers: dict[str, Callable[[str], list[str]]], golden: GoldenSet,
                *, k: int = 5) -> list[Scorecard]:
        """Score several retrievers on the same golden set; return cards sorted by nDCG (best first)."""
        cards = [self.score_retrieval(name, r, golden, k=k) for name, r in named_retrievers.items()]
        cards.sort(key=lambda c: -c.ndcg)
        return cards


__all__ = ["Harness", "GoldenSet", "GoldenItem", "Scorecard"]
