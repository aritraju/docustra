"""
Adaptive RAG
────────────
A routing layer classifies the query complexity before retrieval:
  - trivial   → answer directly from LLM (no retrieval)
  - simple    → single-step vector retrieval
  - complex   → multi-step retrieval with sub-question decomposition
"""
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate

from docustra.core import get_logger
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)


class QueryComplexity(str, Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    COMPLEX = "complex"


_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Classify the user question into one of three categories:
- trivial: common knowledge, math, definitions that need no document lookup
- simple: factual lookup from documents, single concept
- complex: multi-hop reasoning, comparisons, or questions spanning multiple concepts

Respond with exactly one word: trivial, simple, or complex.""",
        ),
        ("human", "{question}"),
    ]
)

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer using only the provided context. Be concise.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

_DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Answer the following question concisely."), ("human", "{question}")]
)


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
        chain = _ROUTER_PROMPT | self._llm
        result = chain.invoke({"question": question}).content.strip().lower()
        try:
            return QueryComplexity(result)
        except ValueError:
            return QueryComplexity.SIMPLE

    def _answer_directly(self, question: str) -> RAGResponse:
        chain = _DIRECT_PROMPT | self._llm
        answer = chain.invoke({"question": question}).content
        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            reasoning="Classified as trivial — answered without retrieval.",
            metadata={"complexity": "trivial"},
        )

    def _simple_retrieval(self, question: str) -> RAGResponse:
        docs = self._vector_store.similarity_search(question)
        context = "\n\n".join(d.page_content for d in docs)
        chain = _RAG_PROMPT | self._llm
        answer = chain.invoke({"context": context, "question": question}).content
        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            sources=self._format_sources(docs),
            reasoning="Classified as simple — single-step vector retrieval.",
            metadata={"complexity": "simple"},
        )

    def _complex_retrieval(self, question: str) -> RAGResponse:
        decompose_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Break the question into 2-3 focused sub-questions. Return them as a numbered list.",
                ),
                ("human", "{question}"),
            ]
        )
        sub_questions_raw = (decompose_prompt | self._llm).invoke({"question": question}).content
        sub_questions = [
            line.split(". ", 1)[-1].strip()
            for line in sub_questions_raw.strip().splitlines()
            if line.strip()
        ][:3]

        all_docs = []
        for sq in sub_questions:
            all_docs.extend(self._vector_store.similarity_search(sq, k=3))

        seen = set()
        unique_docs = []
        for d in all_docs:
            key = d.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        context = "\n\n".join(d.page_content for d in unique_docs)
        chain = _RAG_PROMPT | self._llm
        answer = chain.invoke({"context": context, "question": question}).content

        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            sources=self._format_sources(unique_docs),
            reasoning=f"Classified as complex — decomposed into {len(sub_questions)} sub-questions.",
            metadata={"complexity": "complex", "sub_questions": sub_questions},
        )
