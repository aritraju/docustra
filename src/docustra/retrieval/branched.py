"""
Branched RAG
────────────
Decomposes a complex query into parallel sub-questions, runs independent
retrieval for each branch, then synthesizes all results into one answer.

Prompts loaded from prompts/<version>/branched.yaml and shared.yaml.
"""

from docustra.core import get_logger
from docustra.core.prompts import get_prompt, get_prompt_version
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)


class BranchedRAG(BaseRAGStrategy):
    pattern = RAGPattern.BRANCHED

    def __init__(self) -> None:
        super().__init__()
        self._vector_store = VectorStore(get_embeddings())

    def query(self, question: str, **kwargs) -> RAGResponse:
        logger.info("Branched RAG query", question=question[:80])

        sub_questions = self._decompose(question)
        logger.info("Decomposed query", branches=len(sub_questions))

        branch_results = []
        all_docs = []
        for sq in sub_questions:
            answer, docs = self._answer_branch(sq)
            branch_results.append({"sub_question": sq, "answer": answer})
            all_docs.extend(docs)

        sub_answers_text = "\n\n".join(
            f"Q: {r['sub_question']}\nA: {r['answer']}" for r in branch_results
        )
        final_answer = (
            (get_prompt("branched", "synthesize") | self._llm)
            .invoke({"question": question, "answers": sub_answers_text})
            .content  # type: ignore[union-attr]
        )

        seen: set[str] = set()
        unique_docs = [
            d
            for d in all_docs
            if not (d.page_content[:80] in seen or seen.add(d.page_content[:80]))  # type: ignore[func-returns-value]
        ]

        return RAGResponse(
            answer=final_answer,
            pattern=self.pattern,
            sources=self._format_sources(unique_docs),
            citations=[
                {
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page"),
                    "passage_preview": d.page_content[:200],
                }
                for d in unique_docs
            ],
            reasoning=f"Decomposed into {len(sub_questions)} branches, synthesized.",
            metadata={
                "sub_questions": sub_questions,
                "branch_answers": branch_results,
                "prompt_version": get_prompt_version(),
            },
        )

    def _decompose(self, question: str) -> list[str]:
        raw = (
            (get_prompt("branched", "decompose") | self._llm)
            .invoke({"question": question})
            .content.strip()  # type: ignore[union-attr]
        )
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        return [ln for ln in lines if len(ln) > 10][:4]

    def _answer_branch(self, sub_question: str) -> tuple[str, list]:
        docs = self._vector_store.similarity_search(sub_question, k=3)
        context = "\n\n".join(d.page_content for d in docs)
        answer = (
            (get_prompt("branched", "branch_answer") | self._llm)
            .invoke({"context": context, "question": sub_question})
            .content  # type: ignore[union-attr]
        )
        return answer, docs
