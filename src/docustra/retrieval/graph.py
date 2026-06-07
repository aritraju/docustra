"""
Graph RAG
─────────
Queries the Neo4j knowledge graph to find entity relationships,
then augments vector-retrieved context with graph-derived context.
Excels at multi-hop questions: "How does regulation X affect vendor Y?"

Prompts loaded from prompts/<version>/graph.yaml and shared.yaml.
"""

from docustra.core import get_logger
from docustra.core.prompts import get_prompt, get_prompt_version
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.storage.graph_store import GraphStore
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)


class GraphRAG(BaseRAGStrategy):
    pattern = RAGPattern.GRAPH

    def __init__(self) -> None:
        super().__init__()
        self._vector_store = VectorStore(get_embeddings())
        self._graph_store = GraphStore()

    def query(self, question: str, **kwargs) -> RAGResponse:
        logger.info("Graph RAG query", question=question[:80])

        entities = self._extract_entities(question)
        logger.info("Extracted entities", entities=entities)

        docs = self._vector_store.similarity_search(question)
        graph_context = ""
        if entities:
            graph_context = self._graph_store.get_entity_context(entities)

        if not graph_context:
            logger.info("No graph context found, using vector only")

        text_context = "\n\n".join(d.page_content for d in docs)
        answer = (
            (get_prompt("graph", "graph_answer") | self._llm)
            .invoke(
                {
                    "question": question,
                    "text_context": text_context,
                    "graph_context": graph_context
                    or "No relationship data found in knowledge graph.",
                }
            )
            .content  # type: ignore[union-attr]
        )

        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            sources=self._format_sources(docs),
            citations=[
                {
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page"),
                    "passage_preview": d.page_content[:200],
                }
                for d in docs
            ],
            reasoning=f"Entities extracted: {entities}. Graph context lines: {len(graph_context.splitlines())}.",
            metadata={
                "entities": entities,
                "graph_context_found": bool(graph_context),
                "prompt_version": get_prompt_version(),
            },
        )

    def _extract_entities(self, question: str) -> list[str]:
        import json

        raw = (
            (get_prompt("graph", "entity_extract") | self._llm)
            .invoke({"question": question})
            .content.strip()  # type: ignore[union-attr]
        )
        # Try JSON array first, fall back to comma-separated
        try:
            entities = json.loads(raw)
            if isinstance(entities, list):
                return [str(e).strip() for e in entities]
        except (json.JSONDecodeError, ValueError):
            pass
        return [e.strip() for e in raw.split(",") if e.strip()]
