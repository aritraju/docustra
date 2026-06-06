"""
HyDE — Hypothetical Document Embedding
───────────────────────────────────────
The LLM generates a hypothetical answer to the query first.
That hypothetical answer (which stylistically resembles real documents)
is embedded and used as the search vector instead of the raw query.
This bridges the embedding space gap between short queries and long documents.
"""
from langchain_core.prompts import ChatPromptTemplate

from docustra.core import get_logger
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)

_HYPOTHETICAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Write a hypothetical document passage that would answer the following question.
Write it as if it were an excerpt from a formal enterprise document or report.
Be specific and factual in tone. Length: 2-3 sentences.""",
        ),
        ("human", "{question}"),
    ]
)

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using only the provided context.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ]
)


class HyDERAG(BaseRAGStrategy):
    pattern = RAGPattern.HYDE

    def __init__(self) -> None:
        super().__init__()
        self._vector_store = VectorStore(get_embeddings())

    def query(self, question: str, **kwargs) -> RAGResponse:
        logger.info("HyDE query", question=question[:80])

        # Step 1: Generate hypothetical document
        chain = _HYPOTHETICAL_PROMPT | self._llm
        hypothetical_doc = chain.invoke({"question": question}).content.strip()
        logger.info("Generated hypothetical doc", preview=hypothetical_doc[:100])

        # Step 2: Use the hypothetical doc as the search query
        docs = self._vector_store.similarity_search(hypothetical_doc)

        # Step 3: Answer with real retrieved docs
        context = "\n\n".join(d.page_content for d in docs)
        answer_chain = _RAG_PROMPT | self._llm
        answer = answer_chain.invoke({"context": context, "question": question}).content

        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            sources=self._format_sources(docs),
            reasoning=f"Hypothetical document used for retrieval: '{hypothetical_doc[:150]}...'",
            metadata={"hypothetical_document": hypothetical_doc},
        )
