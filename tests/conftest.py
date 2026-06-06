import pytest
from langchain_core.documents import Document
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            page_content="Apple Inc. reported revenue of $94.9 billion in Q1 2024, driven by iPhone sales.",
            metadata={"source": "apple_10k.pdf", "page": 1, "type": "text"},
        ),
        Document(
            page_content="The SEC requires all public companies to disclose material risk factors in annual filings.",
            metadata={"source": "sec_guidance.pdf", "page": 3, "type": "text"},
        ),
        Document(
            page_content="Supply chain disruptions remain a significant risk factor for hardware manufacturers.",
            metadata={"source": "apple_10k.pdf", "page": 12, "type": "text"},
        ),
    ]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="Mocked LLM response.")
    llm.bind_tools.return_value = llm
    return llm


@pytest.fixture
def mock_vector_store(sample_docs):
    store = MagicMock()
    store.similarity_search.return_value = sample_docs
    store.similarity_search_with_scores.return_value = [(d, 0.85) for d in sample_docs]
    return store
