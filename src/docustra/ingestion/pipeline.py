from pathlib import Path

from docustra.core import IngestionError, get_logger
from docustra.graph.builder import KnowledgeGraphBuilder
from docustra.ingestion.chunker import DocumentChunker
from docustra.ingestion.embedder import get_embeddings
from docustra.ingestion.parser import DocumentParser
from docustra.storage.vector_store import VectorStore

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion:
      parse → chunk → embed → index (vector) → extract entities (graph)
    """

    def __init__(self) -> None:
        self._parser = DocumentParser()
        self._chunker = DocumentChunker()
        self._vector_store = VectorStore(get_embeddings())
        self._graph_builder = KnowledgeGraphBuilder()

    def ingest(self, file_path: str | Path, build_graph: bool = True) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise IngestionError(f"File not found: {path}")

        logger.info("Starting ingestion", file=path.name)

        try:
            parsed = self._parser.parse(path)
        except Exception as e:
            raise IngestionError(f"Parsing failed for {path.name}: {e}") from e

        all_docs = parsed.text_chunks + parsed.tables
        chunks = self._chunker.chunk(all_docs)
        ids = self._vector_store.add_documents(chunks)

        graph_entities = 0
        if build_graph and chunks:
            graph_entities = self._graph_builder.build_from_documents(chunks)

        logger.info(
            "Ingestion complete",
            file=path.name,
            chunks=len(chunks),
            images=len(parsed.images),
            graph_entities=graph_entities,
        )

        return {
            "file": path.name,
            "chunks_indexed": len(chunks),
            "images_found": len(parsed.images),
            "graph_entities": graph_entities,
            "doc_ids": ids,
        }

    def ingest_batch(self, file_paths: list[str | Path], build_graph: bool = True) -> list[dict]:
        results = []
        for path in file_paths:
            try:
                results.append(self.ingest(path, build_graph=build_graph))
            except IngestionError as e:
                logger.error("Ingestion failed", file=str(path), error=str(e))
                results.append({"file": str(path), "error": str(e)})
        return results
