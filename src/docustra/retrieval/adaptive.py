"""
Adaptive RAG
────────────
A routing layer classifies the query complexity before retrieval:
  - trivial   → answer directly from LLM (no retrieval)
  - simple    → single-step vector retrieval
  - complex   → multi-step retrieval with sub-question decomposition

Prompts are loaded from prompts/<version>/adaptive.yaml and shared.yaml.
"""

from enum import StrEnum

from docustra.core import get_logger
from docustra.core.prompts import get_prompt, get_prompt_version
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)


class QueryComplexity(StrEnum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    COMPLEX = "complex"


class AdaptiveRAG(BaseRAGStrategy):
    pattern = RAGPattern.ADAPTIVE

    def __init__(self) -> None:
        super().__init__()
        self._vector_store = VectorStore(get_embeddings())

    def query(self, question: str, **kwargs) -> RAGResponse:
        complexity = self._classify(question)
        logger.info("Adaptive routing", question=question[:80], complexity=complexity)

        if complexity == QueryComplexity.TRIVIAL:
            return self._answer_directly(question)
        if complexity == QueryComplexity.SIMPLE:
            return self._simple_retrieval(question)
        return self._complex_retrieval(question)

    def _classify(self, question: str) -> QueryComplexity:
        chain = get_prompt("adaptive", "router") | self._llm
        result = chain.invoke({"question": question}).content.strip()  # type: ignore[union-attr].lower()
        try:
            return QueryComplexity(result)
        except ValueError:
            return QueryComplexity.SIMPLE

    def _answer_directly(self, question: str) -> RAGResponse:
        chain = get_prompt("adaptive", "direct_answer") | self._llm
        answer = chain.invoke({"question": question}).content  # type: ignore[union-attr]
        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            reasoning="Classified as trivial — answered without retrieval.",
            metadata={"complexity": "trivial", "prompt_version": get_prompt_version()},
        )

    def _simple_retrieval(self, question: str) -> RAGResponse:
        docs = self._vector_store.similarity_search(question)
        context = "\n\n".join(d.page_content for d in docs)
        chain = get_prompt("shared", "citation_rag") | self._llm
        answer = chain.invoke({"context": context, "question": question}).content  # type: ignore[union-attr]
        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            sources=self._format_sources(docs),
            citations=self._extract_citations(docs),
            reasoning="Classified as simple — single-step vector retrieval.",
            metadata={"complexity": "simple", "prompt_version": get_prompt_version()},
        )

    def _decompose(self, question: str) -> list[str]:
        raw = (
            (get_prompt("adaptive", "decompose") | self._llm).invoke({"question": question}).content  # type: ignore[union-attr]
        )
        return [line.strip() for line in raw.strip().splitlines() if line.strip()][:4]

    def _complex_retrieval(self, question: str) -> RAGResponse:
        sub_questions = self._decompose(question)

        all_docs = []
        for sq in sub_questions:
            all_docs.extend(self._vector_store.similarity_search(sq, k=3))

        seen: set[str] = set()
        unique_docs = []
        for d in all_docs:
            key = d.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        context = "\n\n".join(d.page_content for d in unique_docs)
        chain = get_prompt("shared", "citation_rag") | self._llm
        answer = chain.invoke({"context": context, "question": question}).content  # type: ignore[union-attr]
        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            sources=self._format_sources(unique_docs),
            citations=self._extract_citations(unique_docs),
            reasoning=f"Classified as complex — decomposed into {len(sub_questions)} sub-questions.",
            metadata={
                "complexity": "complex",
                "sub_questions": sub_questions,
                "prompt_version": get_prompt_version(),
            },
        )

    def _extract_citations(self, docs) -> list[dict]:
        return [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page"),
                "passage_preview": d.page_content[:200],
            }
            for d in docs
        ]
