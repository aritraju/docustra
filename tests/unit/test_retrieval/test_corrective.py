from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from docustra.retrieval.base import RAGPattern
from docustra.retrieval.corrective import CorrectiveRAG


@pytest.fixture
def crag_strategy(mock_llm, mock_vector_store):
    with (
        patch("docustra.retrieval.corrective.get_llm", return_value=mock_llm),
        patch("docustra.retrieval.corrective.VectorStore", return_value=mock_vector_store),
        patch("docustra.retrieval.corrective.get_embeddings", return_value=MagicMock()),
        patch("docustra.retrieval.corrective.get_settings") as mock_settings,
    ):
        settings = MagicMock()
        settings.retrieval_score_threshold = 0.7
        settings.tavily_api_key = ""
        mock_settings.return_value = settings

        strategy = CorrectiveRAG()
        strategy._llm = mock_llm
        strategy._vector_store = mock_vector_store
        strategy._threshold = 0.7
        strategy._web_search = None
        yield strategy


def test_crag_passes_when_score_above_threshold(crag_strategy, mock_vector_store):
    mock_vector_store.similarity_search_with_scores.return_value = [
        (Document(page_content="Relevant content."), 0.9)
    ]
    crag_strategy._llm.invoke.return_value = MagicMock(content="Answer.")

    response = crag_strategy.query("What are the risk factors?")

    assert response.pattern == RAGPattern.CORRECTIVE
    assert "vector_search" in response.metadata["fallback"]


def test_crag_rewrites_query_when_score_below_threshold(crag_strategy, mock_vector_store):
    mock_vector_store.similarity_search_with_scores.return_value = [
        (Document(page_content="Irrelevant content."), 0.2)
    ]
    mock_vector_store.similarity_search.return_value = [
        Document(page_content="Better content after rewrite.")
    ]
    crag_strategy._llm.invoke.side_effect = [
        MagicMock(content="Rewritten query text"),  # rewrite call
        MagicMock(content="0.8"),                    # score call
        MagicMock(content="Final answer."),           # answer call
    ]

    response = crag_strategy.query("Vague question?")

    assert response.pattern == RAGPattern.CORRECTIVE
