from unittest.mock import MagicMock, patch

import pytest

from docustra.retrieval.adaptive import AdaptiveRAG, QueryComplexity
from docustra.retrieval.base import RAGPattern


@pytest.fixture
def adaptive_strategy(mock_llm, mock_vector_store):
    with (
        patch("docustra.retrieval.adaptive.get_llm", return_value=mock_llm),
        patch("docustra.retrieval.adaptive.VectorStore", return_value=mock_vector_store),
        patch("docustra.retrieval.adaptive.get_embeddings", return_value=MagicMock()),
    ):
        strategy = AdaptiveRAG()
        strategy._llm = mock_llm
        strategy._vector_store = mock_vector_store
        yield strategy


def test_trivial_question_skips_retrieval(adaptive_strategy, mock_vector_store):
    adaptive_strategy._llm.invoke.side_effect = [
        MagicMock(content="trivial"),
        MagicMock(content="4"),
    ]
    response = adaptive_strategy.query("What is 2 + 2?")

    assert response.pattern == RAGPattern.ADAPTIVE
    assert response.metadata["complexity"] == "trivial"
    mock_vector_store.similarity_search.assert_not_called()


def test_simple_question_uses_vector_search(adaptive_strategy, mock_vector_store):
    adaptive_strategy._llm.invoke.side_effect = [
        MagicMock(content="simple"),
        MagicMock(content="Answer from context."),
    ]
    response = adaptive_strategy.query("What is Apple's revenue?")

    assert response.metadata["complexity"] == "simple"
    mock_vector_store.similarity_search.assert_called_once()


def test_complex_question_decomposes(adaptive_strategy, mock_vector_store):
    adaptive_strategy._llm.invoke.side_effect = [
        MagicMock(content="complex"),
        MagicMock(content="1. Sub Q1\n2. Sub Q2\n3. Sub Q3"),
        MagicMock(content="Synthesized answer."),
    ]
    response = adaptive_strategy.query("How do supply chain risks affect Apple's margins and investor outlook?")

    assert response.metadata["complexity"] == "complex"
    assert len(response.metadata["sub_questions"]) > 0
