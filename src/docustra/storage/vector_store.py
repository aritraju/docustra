from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from docustra.core import StorageError, get_logger, get_settings

logger = get_logger(__name__)


class VectorStore:
    """Thin wrapper around Qdrant providing upsert, similarity search, and MMR."""

    def __init__(self, embeddings) -> None:
        self._settings = get_settings()
        self._embeddings = embeddings
        self._client = QdrantClient(url=self._settings.qdrant_url)
        self._store: QdrantVectorStore | None = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collection = self._settings.qdrant_collection
        existing = [c.name for c in self._client.get_collections().collections]
        if collection not in existing:
            dim = len(self._embeddings.embed_query("ping"))
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection", collection=collection, dim=dim)

        self._store = QdrantVectorStore(
            client=self._client,
            collection_name=collection,
            embedding=self._embeddings,
        )

    def add_documents(self, docs: list[Document]) -> list[str]:
        if not docs:
            return []
        try:
            ids = [str(uuid4()) for _ in docs]
            self._store.add_documents(docs, ids=ids)
            logger.info("Indexed documents", count=len(docs))
            return ids
        except Exception as e:
            raise StorageError(f"Failed to index documents: {e}") from e

    def similarity_search(
        self, query: str, k: int | None = None, score_threshold: float | None = None
    ) -> list[Document]:
        k = k or self._settings.retrieval_top_k
        try:
            if score_threshold is not None:
                return self._store.similarity_search_with_relevance_scores(
                    query, k=k, score_threshold=score_threshold
                )
            return self._store.similarity_search(query, k=k)
        except Exception as e:
            raise StorageError(f"Similarity search failed: {e}") from e

    def similarity_search_with_scores(
        self, query: str, k: int | None = None
    ) -> list[tuple[Document, float]]:
        k = k or self._settings.retrieval_top_k
        try:
            return self._store.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            raise StorageError(f"Scored search failed: {e}") from e

    def max_marginal_relevance_search(
        self, query: str, k: int | None = None, fetch_k: int = 20, lambda_mult: float = 0.5
    ) -> list[Document]:
        k = k or self._settings.retrieval_top_k
        try:
            return self._store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
            )
        except Exception as e:
            raise StorageError(f"MMR search failed: {e}") from e

    def as_retriever(self, **kwargs: Any):
        return self._store.as_retriever(**kwargs)
