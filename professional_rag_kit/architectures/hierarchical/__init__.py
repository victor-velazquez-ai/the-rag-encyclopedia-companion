"""professional_rag_kit.architectures.hierarchical — retrieval at the right altitude (Book Ch 10).

Flat retrieval returns equal-sized chunks; hierarchy lets retrieval return a unit sized to the
question (Ch 10). Two patterns, opposite cost profiles:

- ``ParentChildIndex`` — parent-child / small-to-big / auto-merging, *the default hierarchy in every
  RAG system*. It exploits one asymmetry — small chunks retrieve well, large chunks read well — by
  indexing small **child** chunks for precision and *returning the larger **parent*** for generation
  context. No LLM summarization, no summary drift, trivial incremental updates. PURE and tested.

- ``RaptorTree`` — RAPTOR (arXiv:2401.18059): recursively cluster chunks by embedding similarity,
  LLM-summarize each cluster into a new layer, repeat to a root. Queried in **collapsed-tree** mode
  (flatten every node at every layer into one flat top-k pool — the paper's mode, which beats
  tree-traversal). The summarization is a lazy LLM call; clustering is a simple embedding-distance
  pass (a documented hook you can swap for the paper's UMAP + soft-GMM/BIC).

Embeddings come from ``professional_rag_kit.ingestion.embedding.Embedder`` (lazy) and summaries from the generation
backend (lazy) — no vendor SDK is imported at module top level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from professional_rag_kit.core.config import ProviderRegistry, load_config

# Ch 10 (RAPTOR): each internal node is an LLM summary of its cluster; watch for "summary drift"
# (cite the leaf, not just the node), so keep summaries tight and faithful to the inputs.
SUMMARIZE_SYSTEM = (
    "You write a single dense summary that faithfully integrates the supplied passages. "
    "Cover every distinct point; introduce no facts the passages do not contain. "
    "Output only the summary."
)


@dataclass
class ParentChildIndex:
    """Small-to-big retrieval: index children for precision, return parents for context (Ch 10). PURE.

    Register parents with ``add(parent_id, text, children)`` where ``children`` is a list of
    ``(child_id, child_text)``; child chunks are the embedding targets in a real pipeline, but the
    structural job — map a retrieved child back up to its parent — is deterministic and needs no model.
    """

    parents: dict[str, str] = field(default_factory=dict)
    children: dict[str, str] = field(default_factory=dict)
    _child_to_parent: dict[str, str] = field(default_factory=dict)

    def add(self, parent_id: str, text: str, children) -> "ParentChildIndex":
        """Index a parent and its child chunks. Returns self for chaining."""
        self.parents[parent_id] = text
        for child_id, child_text in children:
            self.children[child_id] = child_text
            self._child_to_parent[child_id] = parent_id
        return self

    def parent_of(self, child_id: str) -> str | None:
        """The parent id containing ``child_id`` (pure)."""
        return self._child_to_parent.get(child_id)

    def retrieve(self, child_ids) -> list[str]:
        """Map retrieved child ids → their parent documents, *de-duplicated*, order-preserving (Ch 10).

        This is the "auto-merging" payoff: several children that hit the same parent collapse to one
        coherent passage instead of scattered fragments. Unknown child ids are skipped.
        """
        out: list[str] = []
        seen: set[str] = set()
        for cid in child_ids:
            pid = self._child_to_parent.get(cid)
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            out.append(self.parents[pid])
        return out


@dataclass
class _Node:
    text: str
    layer: int  # 0 = leaf chunk; higher = summary-of-summaries


@dataclass
class RaptorTree:
    """RAPTOR collapsed-tree index (Book Ch 10): cluster + LLM-summarize bottom-up, query flat.

    ``build`` grows the tree; ``query`` pools *every* node (leaves and summaries alike) and runs a
    single flat top-k — the collapsed-tree mode the paper shows beats tree-traversal. Summarization
    uses the lazy generation backend; clustering defaults to a simple embedding-distance grouping
    (override ``cluster_fn`` for the paper's UMAP + soft-GMM/BIC).
    """

    nodes: list[_Node] = field(default_factory=list)
    provider: str = "anthropic"
    model: str = ""
    embed_provider: str = "openai"
    cluster_size: int = 2  # target leaves per cluster at each layer
    max_layers: int = 3

    @classmethod
    def build(
        cls,
        chunks,
        *,
        provider: str = "anthropic",
        model: str = "",
        embed_provider: str = "openai",
        cluster_size: int = 2,
        max_layers: int = 3,
        cluster_fn=None,
        max_tokens: int = 1024,
    ) -> "RaptorTree":
        """Recursively cluster + LLM-summarize ``chunks`` into a collapsed-tree index (Ch 10).

        Leaves are the raw ``chunks``; each successive layer clusters the layer below, summarizes
        each cluster with one LLM call, and adds the summaries as a new layer — stopping at a single
        node, at ``max_layers``, or when a layer can no longer be clustered. The LLM/embedding
        backends are resolved lazily, so building requires no SDK until a real cluster is summarized.
        """
        tree = cls(
            provider=provider,
            model=model or load_config().generation.model,
            embed_provider=embed_provider,
            cluster_size=cluster_size,
            max_layers=max_layers,
        )
        current = [_Node(text=str(c), layer=0) for c in chunks]
        tree.nodes.extend(current)
        backend = ProviderRegistry.get("generation", provider)
        cluster = cluster_fn or tree._cluster_by_embedding

        layer = 1
        while len(current) > 1 and layer < max_layers:
            groups = cluster([n.text for n in current])
            if len(groups) >= len(current):  # no consolidation possible — stop
                break
            summaries: list[_Node] = []
            for group in groups:
                joined = "\n\n".join(current[i].text for i in group)
                summary = backend(tree.model, SUMMARIZE_SYSTEM, joined, max_tokens).strip()
                summaries.append(_Node(text=summary, layer=layer))
            tree.nodes.extend(summaries)
            current = summaries
            layer += 1
        return tree

    def _cluster_by_embedding(self, texts: list[str]) -> list[list[int]]:
        """Default clustering hook: greedy nearest-neighbor grouping by embedding cosine distance.

        A deliberately simple stand-in for RAPTOR's UMAP + soft-GMM(BIC) (Ch 10) — enough to drive a
        real tree, swappable via ``cluster_fn``. Uses the lazy ``Embedder``; pure-ish (one embed call).
        """
        from professional_rag_kit.ingestion.embedding import Embedder

        vecs = Embedder.from_provider(self.embed_provider).embed_documents(texts)
        return _greedy_cluster(vecs, self.cluster_size)

    def query(
        self,
        query: str,
        top_k: int = 10,
        *,
        embed_provider: str | None = None,
    ) -> list[str]:
        """Collapsed-tree retrieval: flat top-k over *all* nodes, leaves and summaries alike (Ch 10).

        Whichever altitude best answers the query wins on its own merits. Scoring is embedding cosine
        similarity via the lazy ``Embedder``.
        """
        if not self.nodes:
            return []
        from professional_rag_kit.ingestion.embedding import Embedder

        emb = Embedder.from_provider(embed_provider or self.embed_provider)
        node_vecs = emb.embed_documents([n.text for n in self.nodes])
        q_vec = emb.embed_query(query)
        scored = sorted(
            range(len(self.nodes)),
            key=lambda i: -_cosine(q_vec, node_vecs[i]),
        )
        return [self.nodes[i].text for i in scored[:top_k]]


# --- pure clustering / similarity helpers ------------------------------------
def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _greedy_cluster(vecs, cluster_size: int) -> list[list[int]]:
    """Greedy nearest-neighbor clustering into groups of ~``cluster_size`` — PURE.

    Walks unused indices, and for each seed pulls in its nearest still-unused neighbors until the
    group reaches ``cluster_size``. Deterministic; a documented placeholder for UMAP + soft-GMM.
    """
    n = len(vecs)
    used = [False] * n
    groups: list[list[int]] = []
    size = max(1, cluster_size)
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        group = [i]
        while len(group) < size:
            best_j, best_sim = None, -2.0
            for j in range(n):
                if used[j]:
                    continue
                sim = _cosine(vecs[i], vecs[j])
                if sim > best_sim:
                    best_j, best_sim = j, sim
            if best_j is None:
                break
            used[best_j] = True
            group.append(best_j)
        groups.append(group)
    return groups


__all__ = ["ParentChildIndex", "RaptorTree", "SUMMARIZE_SYSTEM"]
