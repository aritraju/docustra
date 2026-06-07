"""
Cross-Encoder Reranker
══════════════════════
Uses a sentence-transformers CrossEncoder to rescore retrieved documents.
The cross-encoder processes the (query, document) pair jointly — giving
much higher accuracy than bi-encoder cosine similarity at query time.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (free, ~80MB, strong MS MARCO scores)
Alternative high-quality free options:
  - cross-encoder/ms-marco-MiniLM-L-12-v2  (larger, more accurate)
  - cross-encoder/qnli-distilroberta-base   (faster)
"""

from __future__ import annotations

import functools

from langchain_core.documents import Document

from docustra.core import get_logger, get_settings

logger = get_logger(__name__)


@functools.lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str):
    """Load and cache the cross-encoder model (heavy — only load once)."""
    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model", model=model_name)
        return CrossEncoder(model_name)
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for reranking. "
            "Install with: pip install sentence-transformers"
        ) from e


class CrossEncoderReranker:
    """
    Reranks a list of documents against a query using a cross-encoder model.

    Unlike bi-encoders (where query and document are encoded separately),
    cross-encoders jointly encode the (query, document) pair, enabling
    fine-grained relevance scoring at the cost of higher latency.

    Typical pipeline:
        docs = vector_store.similarity_search(query, k=20)   # fast candidate retrieval
        top_docs = reranker.rerank(query, docs, top_n=5)     # precise reranking
    """

    def __init__(self, model_name: str | None = None, top_n: int | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.reranker_model
        self._top_n = top_n or settings.reranker_top_n

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int | None = None,
    ) -> list[Document]:
        """
        Rerank documents by cross-encoder relevance score.

        Parameters
        ----------
        query : str
            The user's question.
        documents : list[Document]
            Candidate documents (typically from BM25 + vector fusion).
        top_n : int | None
            Number of top documents to return. Defaults to ``Settings.reranker_top_n``.

        Returns
        -------
        list[Document]
            Reranked documents (best first) with ``reranker_score`` in metadata.
        """
        if not documents:
            return []

        top_n = top_n or self._top_n
        model = _get_cross_encoder(self._model_name)

        # Build (query, passage) pairs
        pairs = [(query, doc.page_content) for doc in documents]

        try:
            scores = model.predict(pairs)
        except Exception as e:
            logger.warning("Cross-encoder scoring failed, returning unsorted", error=str(e))
            return documents[:top_n]

        # Attach scores and sort descending
        scored = sorted(
            zip(documents, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for doc, score in scored[:top_n]:
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "reranker_score": float(score)},
            )
            result.append(doc_copy)

        logger.info(
            "Reranking complete",
            candidates=len(documents),
            returned=len(result),
            top_score=round(float(scored[0][1]), 4) if scored else None,
        )
        return result


@functools.lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    """Cached singleton reranker using settings defaults."""
    return CrossEncoderReranker()
