"""
Hybrid RAG — BM25 + Vector Search with Cross-Encoder Reranking
═══════════════════════════════════════════════════════════════
Retrieval pipeline:

  Query
    │
    ├──▶ BM25 keyword search  ─────┐
    │    (rank-bm25 library)       │
    │                              ├──▶ Reciprocal Rank Fusion ──▶ Cross-Encoder ──▶ LLM
    └──▶ Dense vector search  ─────┘         (RRF)               (reranker)    (citation prompt)
         (Qdrant)

Why this matters
─────────────────
- BM25 excels at exact keyword / rare term matching (e.g. "Section 12(g)")
- Vector search excels at semantic similarity (paraphrases, synonyms)
- RRF fusion beats both individually on most retrieval benchmarks (BEIR)
- Cross-encoder reranking on the merged candidate set further improves
  answer precision by ~10-20% over vector-only retrieval

Citation enforcement
────────────────────
The answer prompt requires inline citations [Source: X, Page: Y] and
declines to answer when the retrieved context is insufficient.
"""

from __future__ import annotations

from langchain_core.documents import Document

from docustra.core import RetrievalError, get_logger, get_settings
from docustra.core.prompts import get_prompt, get_prompt_version
from docustra.ingestion.embedder import get_embeddings
from docustra.retrieval.base import BaseRAGStrategy, RAGPattern, RAGResponse
from docustra.retrieval.reranker import CrossEncoderReranker
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)

_CANNOT_ANSWER = "I cannot answer this question based on the provided documents."


def _reciprocal_rank_fusion(
    bm25_docs: list[Document],
    vector_docs: list[Document],
    k: int = 60,
    bm25_weight: float = 0.4,
) -> list[Document]:
    """
    Merge two ranked lists via Reciprocal Rank Fusion.

    RRF score: sum_over_lists( weight / (k + rank) )

    A high k (60 is standard) reduces the sensitivity to exact rank position,
    making fusion robust even when the two lists disagree on ordering.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    vector_weight = 1.0 - bm25_weight

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content[:200]  # dedup key
        scores[key] = scores.get(key, 0.0) + bm25_weight / (k + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(vector_docs, start=1):
        key = doc.page_content[:200]
        scores[key] = scores.get(key, 0.0) + vector_weight / (k + rank)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


class BM25Index:
    """
    Lazy-loaded BM25 index built over documents in the Qdrant collection.

    The index is rebuilt on first call and cached for the lifetime of the
    strategy instance. For production use with large corpora, this should
    be persisted to disk — but for this demo size it is fast enough.
    """

    def __init__(self) -> None:
        self._tokenized: list[list[str]] | None = None
        self._docs: list[Document] | None = None
        self._bm25 = None

    def _ensure_built(self, docs: list[Document]) -> None:
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank-bm25 is required for hybrid retrieval. "
                "Install with: pip install rank-bm25"
            ) from e

        self._docs = docs
        self._tokenized = [doc.page_content.lower().split() for doc in docs]
        self._bm25 = BM25Okapi(self._tokenized)
        logger.info("BM25 index built", num_docs=len(docs))

    def search(self, query: str, docs: list[Document], k: int = 10) -> list[Document]:
        """Return top-k documents ranked by BM25 score."""
        self._ensure_built(docs)
        if not docs:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Pair with docs and sort
        ranked = sorted(
            zip(docs, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        return [doc for doc, _ in ranked[:k]]


class HybridRAG(BaseRAGStrategy):
    """
    BM25 + Dense vector retrieval fused with RRF, reranked by cross-encoder,
    answered with mandatory citation enforcement.
    """

    pattern = RAGPattern.HYBRID

    def __init__(self) -> None:
        super().__init__()
        settings = get_settings()
        self._vector_store = VectorStore(get_embeddings())
        self._bm25 = BM25Index()
        self._reranker = CrossEncoderReranker() if settings.enable_reranking else None
        self._bm25_weight = settings.bm25_weight
        self._hybrid_top_k = settings.hybrid_top_k
        self._final_k = settings.reranker_top_n

    def query(self, question: str, **kwargs) -> RAGResponse:
        logger.info("Hybrid RAG query", question=question[:80])

        # ── 1. Vector retrieval ────────────────────────────────────────────
        vector_docs = self._vector_store.similarity_search(question, k=self._hybrid_top_k)

        # ── 2. BM25 retrieval (over same corpus) ──────────────────────────
        # We use the vector candidates as the BM25 corpus to avoid loading
        # all docs from Qdrant. For full BM25 coverage, fetch a larger k first.
        all_candidates = self._vector_store.similarity_search(question, k=50)
        bm25_docs = self._bm25.search(question, all_candidates, k=self._hybrid_top_k)

        # ── 3. Reciprocal Rank Fusion ──────────────────────────────────────
        fused_docs = _reciprocal_rank_fusion(
            bm25_docs, vector_docs, bm25_weight=self._bm25_weight
        )
        logger.info(
            "RRF fusion",
            bm25_candidates=len(bm25_docs),
            vector_candidates=len(vector_docs),
            fused=len(fused_docs),
        )

        # ── 4. Cross-encoder reranking ─────────────────────────────────────
        if self._reranker and fused_docs:
            final_docs = self._reranker.rerank(question, fused_docs, top_n=self._final_k)
        else:
            final_docs = fused_docs[: self._final_k]

        if not final_docs:
            raise RetrievalError("No documents retrieved for hybrid query.")

        # ── 5. Citation-enforced answer generation ─────────────────────────
        context = self._build_context_with_metadata(final_docs)
        prompt = get_prompt("shared", "citation_rag")
        chain = prompt | self._llm
        answer = chain.invoke({"context": context, "question": question}).content.strip()

        # Detect decline — don't attach sources if model couldn't answer
        declined = answer.startswith("I cannot answer")

        citations = [] if declined else self._extract_citations(final_docs)

        return RAGResponse(
            answer=answer,
            pattern=self.pattern,
            sources=self._format_sources(final_docs) if not declined else [],
            citations=citations,
            reasoning=(
                f"Hybrid retrieval: BM25 weight={self._bm25_weight}, "
                f"vector weight={1 - self._bm25_weight}, "
                f"reranked={self._reranker is not None}, "
                f"final_docs={len(final_docs)}, declined={declined}"
            ),
            metadata={
                "retrieval_method": "hybrid_bm25_vector_rrf",
                "reranking": self._reranker is not None,
                "bm25_weight": self._bm25_weight,
                "docs_retrieved": len(final_docs),
                "prompt_version": get_prompt_version(),
                "declined": declined,
            },
        )

    def _build_context_with_metadata(self, docs: list[Document]) -> str:
        """Build context string with rich source metadata for citation grounding."""
        parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            score = doc.metadata.get("reranker_score")
            header = f"[Passage {i} | Source: {source}" + (f", Page: {page}" if page else "") + "]"
            if score is not None:
                header += f"  relevance: {score:.3f}"
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(parts)

    def _extract_citations(self, docs: list[Document]) -> list[dict]:
        """Build structured citation list from retrieved documents."""
        return [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "passage_preview": doc.page_content[:200],
                "reranker_score": doc.metadata.get("reranker_score"),
            }
            for doc in docs
        ]
