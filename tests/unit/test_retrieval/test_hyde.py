from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from docustra.retrieval.base import RAGPattern
from docustra.retrieval.hyde import HyDERAG


@pytest.fixture
def hyde_strategy(mock_llm, mock_vector_store):
    with (
        patch("docustra.retrieval.hyde.get_llm", return_value=mock_llm),
        patch("docustra.retrieval.hyde.VectorStore", return_value=mock_vector_store),
        patch("docustra.retrieval.hyde.get_embeddings", return_value=MagicMock()),
    ):
        strategy = HyDERAG()
        strategy._llm = mock_llm
        strategy._vector_store = mock_vector_store
        yield strategy


def test_hyde_uses_hypothetical_doc_for_retrieval(hyde_strategy, mock_vector_store):
    mock_llm = hyde_strategy._llm
    mock_llm.invoke.side_effect = [
        MagicMock(content="Apple Inc reported strong revenue from iPhone sales in Q1 2024."),
        MagicMock(content="Final answer based on retrieved context."),
    ]

    response = hyde_strategy.query("What drove Apple's revenue in Q1?")

    assert response.pattern == RAGPattern.HYDE
    assert "hypothetical" in response.reasoning.lower()
    assert mock_vector_store.similarity_search.called


def test_hyde_returns_sources(hyde_strategy):
    hyde_strategy._llm.invoke.side_effect = [
        MagicMock(content="Hypothetical document text."),
        MagicMock(content="The answer is X."),
    ]
    response = hyde_strategy.query("Test question")
    assert isinstance(response.sources, list)
    assert response.metadata.get("hypothetical_document") is not None
