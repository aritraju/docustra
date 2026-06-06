import pytest
from langchain_core.documents import Document

from docustra.ingestion.chunker import DocumentChunker


def test_chunk_splits_long_document():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    docs = [Document(page_content="word " * 200, metadata={"source": "test.pdf"})]
    chunks = chunker.chunk(docs)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 150 for c in chunks)


def test_chunk_preserves_metadata():
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    docs = [Document(page_content="Hello world. " * 50, metadata={"source": "test.pdf", "page": 1})]
    chunks = chunker.chunk(docs)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1


def test_chunk_text_creates_documents():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_text("Sample text. " * 50, metadata={"type": "text"})
    assert len(chunks) >= 1
    assert all(c.metadata.get("type") == "text" for c in chunks)


def test_chunk_empty_input():
    chunker = DocumentChunker()
    result = chunker.chunk([])
    assert result == []
