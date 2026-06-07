"""
HyDE — Hypothetical Document Embedding
───────────────────────────────────────
The LLM generates a hypothetical answer to the query first.
That hypothetical answer (which stylistically resembles real documents)
is embedded and used as the search vector instead of the raw query.
This bridges the embedding space gap between short queries and long documents.

Prompts loaded from prompts/<version>/hyde.yaml and shared.yaml.
"""

from docustra.core import get_logger
from docustra.core.prompts import get_prompt, get_prompt_version
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)


class HyDERAG(BaseRAGStrategy):
    pattern = RAGPattern.HYDE

    def __init__(self) -> None:
        super().__init__()
        self._vector_store = VectorStore(get_embeddings())

    def query(self, question: str, **kwargs) -> RAGResponse:
        logger.info("HyDE query", question=question[:80])

        # Step 1: Generate hypothetical document
        hypothetical_doc = (
            (get_prompt("hyde", "hypothetical_doc") | self._llm)
            .invoke({"question": question})
            .content.strip()  # type: ignore[union-attr]
        )
        logger.info("Generated hypothetical doc", preview=hypothetical_doc[:100])

        # Step 2: Use the hypothetical doc as the search query
        docs = self._vector_store.similarity_search(hypothetical_doc)

        # Step 3: Answer with real retrieved docs (with citation enforcement)
        context = "\n\n".join(d.page_content for d in docs)
        answer = (
            (get_prompt("shared", "citation_rag") | self._llm)
            .invoke({"context": context, "question": question})
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
            reasoning=f"Hypothetical document used for retrieval: '{hypothetical_doc[:150]}...'",
            metadata={
                "hypothetical_document": hypothetical_doc,
                "prompt_version": get_prompt_version(),
            },
        )
