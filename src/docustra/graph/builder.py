import time

from langchain_core.documents import Document

from docustra.core import get_logger
from docustra.graph.extractor import EntityExtractor
from docustra.storage.graph_store import GraphStore

logger = get_logger(__name__)

# Gemini free tier: 15 RPM. Process chunks in batches and pause between batches
# to stay safely within the rate limit.
_RATE_LIMIT_BATCH = 10  # chunks per batch
_RATE_LIMIT_PAUSE = (
    5.0  # seconds to wait between batches (10 chunks = 10 LLM calls; pause keeps RPM ≤ 12)
)


class KnowledgeGraphBuilder:
    """Builds the Neo4j knowledge graph from ingested documents."""

    def __init__(self) -> None:
        self._extractor = EntityExtractor()
        self._graph = GraphStore()

    def build_from_documents(
        self, docs: list[Document], batch_size: int = _RATE_LIMIT_BATCH
    ) -> int:
        total_entities = 0
        batches = [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]
        total_batches = len(batches)

        for idx, batch in enumerate(batches, start=1):
            logger.info("Processing graph batch", batch=idx, total=total_batches, chunks=len(batch))
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

            # Pause between batches to respect free-tier RPM limit
            if idx < total_batches:
                time.sleep(_RATE_LIMIT_PAUSE)

        logger.info("Knowledge graph built", entities=total_entities)
        return total_entities
