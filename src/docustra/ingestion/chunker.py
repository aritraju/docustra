from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docustra.core import get_settings


class DocumentChunker:
    """Splits documents into overlapping chunks for indexing."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        settings = get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        # Preserve type metadata after splitting
        for chunk in chunks:
            chunk.metadata.setdefault("type", "text")
        return chunks

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[Document]:
        return self._splitter.create_documents([text], metadatas=[metadata or {}])
