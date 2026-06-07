from docustra.retrieval.adaptive import AdaptiveRAG
from docustra.retrieval.agentic import AgenticRAG
from docustra.retrieval.base import RAGPattern, RAGResponse
from docustra.retrieval.branched import BranchedRAG
from docustra.retrieval.corrective import CorrectiveRAG
from docustra.retrieval.graph import GraphRAG
from docustra.retrieval.hybrid import HybridRAG
from docustra.retrieval.hyde import HyDERAG
from docustra.retrieval.multimodal import MultimodalRAG
from docustra.retrieval.self_rag import SelfRAG

STRATEGY_REGISTRY: dict[RAGPattern, type] = {
    RAGPattern.ADAPTIVE: AdaptiveRAG,
    RAGPattern.AGENTIC: AgenticRAG,
    RAGPattern.BRANCHED: BranchedRAG,
    RAGPattern.CORRECTIVE: CorrectiveRAG,
    RAGPattern.GRAPH: GraphRAG,
    RAGPattern.HYBRID: HybridRAG,
    RAGPattern.HYDE: HyDERAG,
    RAGPattern.MULTIMODAL: MultimodalRAG,
    RAGPattern.SELF_RAG: SelfRAG,
}


def get_strategy(pattern: RAGPattern | str):
    if isinstance(pattern, str):
        pattern = RAGPattern(pattern)
    cls = STRATEGY_REGISTRY.get(pattern)
    if cls is None:
        raise ValueError(f"Unknown RAG pattern: {pattern}")
    return cls()


__all__ = [
    "RAGPattern",
    "RAGResponse",
    "AdaptiveRAG",
    "AgenticRAG",
    "BranchedRAG",
    "CorrectiveRAG",
    "GraphRAG",
    "HybridRAG",
    "HyDERAG",
    "MultimodalRAG",
    "SelfRAG",
    "STRATEGY_REGISTRY",
    "get_strategy",
]
