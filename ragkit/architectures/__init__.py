"""ragkit.architectures — Part III: structures and control loops beyond flat retrieval (Book Ch 9–12).

    graph/         knowledge-graph RAG — LightRAG default · LazyGraphRAG · full GraphRAG   (Ch 9)
    hierarchical/  parent-child / small-to-big default · RAPTOR collapsed-tree            (Ch 10)
    adaptive/      Adaptive-RAG routing · CRAG · Self-RAG · FLARE as composable patterns   (Ch 10)
    agentic/       plan-act-reflect loop · Self-Ask / IRCoT · MCP wrappers · EntityResolver (Ch 11)
    multimodal/    ColPali / ColQwen2.5 visual document retrieval (token pooling)          (Ch 12)

The discipline of this Part: every structure here is expensive, so it must earn its cost against a
flat-retrieval baseline. A graph, a summary tree, a control loop, or a multi-vector index is added
only when the query log (or the corpus) proves flat retrieval provably cannot do the job — never by
default.

Phase-1 scaffold. Phase 2 exports the top-level conveniences listed below.
"""

from ragkit.architectures.adaptive import AdaptiveRAG
from ragkit.architectures.agentic import AgenticRAG, EntityResolver, MCPSource
from ragkit.architectures.graph import GraphIndex
from ragkit.architectures.hierarchical import ParentChildIndex, RaptorTree
from ragkit.architectures.multimodal import VisualRetriever

__all__ = [
    "GraphIndex",
    "ParentChildIndex",
    "RaptorTree",
    "AdaptiveRAG",
    "AgenticRAG",
    "EntityResolver",
    "MCPSource",
    "VisualRetriever",
]
