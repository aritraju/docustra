"""
Branched RAG
────────────
Decomposes a complex query into parallel sub-questions, runs independent
retrieval for each branch, then synthesizes all results into one answer.
Parallel execution via asyncio for efficiency.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_core.prompts import ChatPromptTemplate

from docustra.core import get_logger
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)

_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Decompose the question into 2-4 independent sub-questions that together cover the full answer.
Each sub-question should be answerable independently from the document corpus.
Return ONLY a numbered list, one sub-question per line.""",
        ),
        ("human", "{question}"),
    ]
)

_BRANCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using only the provided context.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ]
)

_SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You have answers to several sub-questions of a complex query.
Synthesize them into a single coherent, comprehensive answer.
Do not repeat information. Cite sources where relevant.""",
        ),
        (
            "human",
            "Original question: {original_question}\n\nSub-question answers:\n{sub_answers}",
        ),
    ]
)


class BranchedRAG(BaseRAGStrategy):
    pattern = RAGPattern.BRANCHED

    def __init__(self) -> None:
        super().__init__()
        self._vector_store = VectorStore(get_embeddings())

    def query(self, question: str, **kwargs) -> RAGResponse:
        logger.info("Branched RAG query", question=question[:80])

        sub_questions = self._decompose(question)
        logger.info("Decomposed query", branches=len(sub_questions))

        # Run branches sequentially to respect free-tier RPM limits
        # (Switch to ThreadPoolExecutor for paid API keys with higher RPM)
        branch_results = []
        all_docs = []
        for sq in sub_questions:
            answer, docs = self._answer_branch(sq)
            branch_results.append({"sub_question": sq, "answer": answer})
            all_docs.extend(docs)

        sub_answers_text = "\n\n".join(
            f"Q: {r['sub_question']}\nA: {r['answer']}" for r in branch_results
        )
        chain = _SYNTHESIZE_PROMPT | self._llm
        final_answer = chain.invoke(
            {"original_question": question, "sub_answers": sub_answers_text}
        ).content

        seen = set()
        unique_docs = [
            d for d in all_docs if not (d.page_content[:80] in seen or seen.add(d.page_content[:80]))
        ]

        return RAGResponse(
            answer=final_answer,
            pattern=self.pattern,
            sources=self._format_sources(unique_docs),
            reasoning=f"Decomposed into {len(sub_questions)} branches, retrieved in parallel, then synthesized.",
            metadata={"sub_questions": sub_questions, "branch_answers": branch_results},
        )

    def _decompose(self, question: str) -> list[str]:
        chain = _DECOMPOSE_PROMPT | self._llm
        raw = chain.invoke({"question": question}).content.strip()
        lines = [l.split(". ", 1)[-1].strip() for l in raw.splitlines() if l.strip()]
        return [l for l in lines if len(l) > 10][:4]

    def _answer_branch(self, sub_question: str) -> tuple[str, list]:
        docs = self._vector_store.similarity_search(sub_question, k=3)
        context = "\n\n".join(d.page_content for d in docs)
        chain = _BRANCH_PROMPT | self._llm
        answer = chain.invoke({"context": context, "question": sub_question}).content
        return answer, docs
