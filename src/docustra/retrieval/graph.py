"""
Graph RAG
─────────
Queries the Neo4j knowledge graph to find entity relationships,
then augments vector-retrieved context with graph-derived context.
Excels at multi-hop questions: "How does regulation X affect vendor Y?"
"""
from langchain_core.prompts import ChatPromptTemplate

from docustra.core import get_logger
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.storage.graph_store import GraphStore
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)

_ENTITY_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Extract all named entities from the question: companies, people, regulations, products, locations.
Return as a comma-separated list. Return ONLY the list.""",
        ),
        ("human", "{question}"),
    ]
)

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Answer the question using both the document context and the knowledge graph context.
The knowledge graph context shows relationships between entities.

Document Context:
{doc_context}

Knowledge Graph Context:
{graph_context}""",
        ),
        ("human", "{question}"),
    ]
)


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

        # Parallel: vector search + graph traversal
        docs = self._vector_store.similarity_search(question)
        graph_context = ""
        if entities:
            graph_context = self._graph_store.get_entity_context(entities)

        if not graph_context:
            logger.info("No graph context found, using vector only")

        doc_context = "\n\n".join(d.page_content for d in docs)
        chain = _RAG_PROMPT | self._llm
        answer = chain.invoke(
            {
                "question": question,
                "doc_context": doc_context,
                "graph_context": graph_context or "No relationship data found in knowledge graph.",
            }
        ).content

        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            sources=self._format_sources(docs),
            reasoning=f"Entities extracted: {entities}. Graph context lines: {len(graph_context.splitlines())}.",
            metadata={"entities": entities, "graph_context_found": bool(graph_context)},
        )

    def _extract_entities(self, question: str) -> list[str]:
        chain = _ENTITY_EXTRACT_PROMPT | self._llm
        raw = chain.invoke({"question": question}).content.strip()
        return [e.strip() for e in raw.split(",") if e.strip()]
