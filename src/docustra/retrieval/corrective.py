"""
Corrective RAG (CRAG)
─────────────────────
After retrieval, each document is scored for relevance.
If the average score falls below a threshold, the system either:
  1. Reformulates the query and retries vector search, or
  2. Falls back to a live web search via Tavily.
"""
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from docustra.core import get_logger, get_settings
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)

_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Score the relevance of this document to the question.
Return ONLY a number between 0.0 and 1.0. Nothing else.""",
        ),
        ("human", "Question: {question}\n\nDocument: {document}"),
    ]
)

_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the question to improve document retrieval. Be more specific and use different keywords.",
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


class CorrectiveRAG(BaseRAGStrategy):
    pattern = RAGPattern.CORRECTIVE

    def __init__(self) -> None:
        super().__init__()
        settings = get_settings()
        self._threshold = settings.retrieval_score_threshold
        self._vector_store = VectorStore(get_embeddings())
        if settings.tavily_api_key and settings.tavily_api_key != "your_tavily_api_key_here":
            import os
            os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
            self._web_search = TavilySearchResults(max_results=4)
        else:
            self._web_search = None

    def query(self, question: str, **kwargs) -> RAGResponse:
        logger.info("CRAG query", question=question[:80])

        docs_with_scores = self._vector_store.similarity_search_with_scores(question)
        avg_score = (
            sum(s for _, s in docs_with_scores) / len(docs_with_scores)
            if docs_with_scores else 0.0
        )

        logger.info("Retrieval scores", avg=round(avg_score, 3), threshold=self._threshold)

        if avg_score >= self._threshold:
            docs = [d for d, _ in docs_with_scores]
            fallback_used = "vector_search"
        else:
            rewritten = self._rewrite_query(question)
            logger.info("Query rewritten", original=question[:60], rewritten=rewritten[:60])

            retry_docs = self._vector_store.similarity_search(rewritten, k=5)
            retry_score = self._score_docs(rewritten, retry_docs)

            if retry_score >= self._threshold or not self._web_search:
                docs = retry_docs
                fallback_used = "rewritten_vector_search"
            else:
                docs = self._web_search_fallback(question)
                fallback_used = "web_search"

        context = "\n\n".join(d.page_content for d in docs)
        chain = _RAG_PROMPT | self._llm
        answer = chain.invoke({"context": context, "question": question}).content

        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            sources=self._format_sources(docs),
            reasoning=f"Avg relevance score: {avg_score:.2f} (threshold: {self._threshold}). Fallback used: {fallback_used}.",
            metadata={"avg_score": avg_score, "fallback": fallback_used},
        )

    def _rewrite_query(self, question: str) -> str:
        chain = _REWRITE_PROMPT | self._llm
        return chain.invoke({"question": question}).content.strip()

    def _score_docs(self, question: str, docs: list[Document]) -> float:
        if not docs:
            return 0.0
        scores = []
        for doc in docs[:3]:
            try:
                chain = _RELEVANCE_PROMPT | self._llm
                raw = chain.invoke({"question": question, "document": doc.page_content[:500]}).content
                scores.append(float(raw.strip()))
            except (ValueError, Exception):
                scores.append(0.5)
        return sum(scores) / len(scores)

    def _web_search_fallback(self, question: str) -> list[Document]:
        results = self._web_search.invoke(question)
        return [
            Document(
                page_content=r.get("content", r.get("snippet", "")),
                metadata={"source": r.get("url", "web"), "type": "web"},
            )
            for r in results
        ]
