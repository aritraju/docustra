"""
Self-RAG
────────
The LLM generates reflection tokens at each step to self-critique:
  [Retrieve]  — should retrieval happen?
  [Relevant]  — is the retrieved document relevant?
  [Supported] — is the claim grounded in the context?
  [Useful]    — is the final response useful?

This makes the reasoning process transparent and auditable.
Prompts loaded from prompts/<version>/self_rag.yaml and shared.yaml.
"""

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from docustra.core import get_logger
from docustra.core.prompts import get_prompt, get_prompt_version
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)


@dataclass
class ReflectionTokens:
    retrieve: bool = True
    relevant: bool = True
    supported: bool = True
    useful: bool = True


class SelfRAG(BaseRAGStrategy):
    pattern = RAGPattern.SELF_RAG

    def __init__(self) -> None:
        super().__init__()
        self._vector_store = VectorStore(get_embeddings())

    def query(self, question: str, **kwargs) -> RAGResponse:
        logger.info("Self-RAG query", question=question[:80])
        tokens = ReflectionTokens()
        reasoning_log = []

        # [Retrieve] token
        retrieve_raw = (
            (get_prompt("self_rag", "retrieve_token") | self._llm)
            .invoke({"question": question})
            .content.strip()  # type: ignore[union-attr]
            .upper()
        )
        tokens.retrieve = "YES" in retrieve_raw
        reasoning_log.append(f"[Retrieve]: {retrieve_raw}")

        if not tokens.retrieve:
            direct = (
                (
                    ChatPromptTemplate.from_messages(
                        [("system", "Answer concisely."), ("human", "{question}")]
                    )
                    | self._llm
                )
                .invoke({"question": question})
                .content  # type: ignore[union-attr]
            )
            return RAGResponse(
                answer=direct,
                pattern=self.pattern,
                reasoning="\n".join(reasoning_log) + "\n[Retrieve]: NO — answered directly.",
                metadata={"tokens": vars(tokens), "prompt_version": get_prompt_version()},
            )

        # Retrieve docs and filter by [Relevant] token
        docs = self._vector_store.similarity_search(question, k=6)
        relevant_docs = []
        for doc in docs:
            relevance = (
                (get_prompt("self_rag", "relevance_token") | self._llm)
                .invoke({"question": question, "document": doc.page_content[:500]})
                .content.strip()  # type: ignore[union-attr]
                .upper()
            )
            if "YES" in relevance:
                relevant_docs.append(doc)
            reasoning_log.append(f"[Relevant] for chunk: {relevance}")

        if not relevant_docs:
            reasoning_log.append("[Relevant]: No relevant documents found — using all retrieved.")
            relevant_docs = docs[:3]
            tokens.relevant = False

        context = "\n\n".join(d.page_content for d in relevant_docs)
        # Use citation_rag for grounded answer generation
        answer = (
            (get_prompt("shared", "citation_rag") | self._llm)
            .invoke({"context": context, "question": question})
            .content  # type: ignore[union-attr]
        )

        # [Supported] token
        supported_raw = (
            (get_prompt("self_rag", "support_token") | self._llm)
            .invoke({"context": context[:1000], "answer": answer[:500]})
            .content.strip()  # type: ignore[union-attr]
            .upper()
        )
        tokens.supported = "YES" in supported_raw or "PARTIALLY" in supported_raw
        reasoning_log.append(f"[Supported]: {supported_raw}")

        # [Useful] token
        useful_raw = (
            (get_prompt("self_rag", "useful_token") | self._llm)
            .invoke({"question": question, "answer": answer})
            .content.strip()  # type: ignore[union-attr]
            .upper()
        )
        tokens.useful = "YES" in useful_raw
        reasoning_log.append(f"[Useful]: {useful_raw}")

        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            sources=self._format_sources(relevant_docs),
            citations=[
                {
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page"),
                    "passage_preview": d.page_content[:200],
                }
                for d in relevant_docs
            ],
            reasoning="\n".join(reasoning_log),
            metadata={
                "tokens": vars(tokens),
                "relevant_docs_count": len(relevant_docs),
                "prompt_version": get_prompt_version(),
            },
        )
