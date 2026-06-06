from langchain_core.documents import Document

from docustra.core import get_logger
from docustra.graph.extractor import EntityExtractor
from docustra.storage.graph_store import GraphStore

logger = get_logger(__name__)


class KnowledgeGraphBuilder:
    """Builds the Neo4j knowledge graph from ingested documents."""

    def __init__(self) -> None:
        self._extractor = EntityExtractor()
        self._graph = GraphStore()

    def build_from_documents(self, docs: list[Document], batch_size: int = 10) -> int:
        total_entities = 0
        batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]

        for batch in batches:
            extractions = self._extractor.extract_from_documents(batch)
            for extraction in extractions:
                for entity in extraction.get("entities", []):
                    self._graph.upsert_entity(
                        label=entity.get("type", "Entity"),
                        name=entity["name"],
                        properties={"source": extraction.get("source", "")},
                    )
                    total_entities += 1

                for rel in extraction.get("relationships", []):
                    try:
                        self._graph.upsert_relationship(
                            from_label="Entity",
                            from_name=rel["from"],
                            rel_type=rel["type"].upper().replace(" ", "_"),
                            to_label="Entity",
                            to_name=rel["to"],
                        )
                    except Exception as e:
                        logger.warning("Relationship upsert failed", rel=rel, error=str(e))

        logger.info("Knowledge graph built", entities=total_entities)
        return total_entities
