from neo4j import GraphDatabase, Driver

from docustra.core import GraphError, get_logger, get_settings

logger = get_logger(__name__)


class GraphStore:
    """Neo4j wrapper for knowledge graph operations."""

    def __init__(self) -> None:
        settings = get_settings()
        self._driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self._verify_connection()

    def _verify_connection(self) -> None:
        try:
            self._driver.verify_connectivity()
            logger.info("Neo4j connection verified")
        except Exception as e:
            raise GraphError(f"Cannot connect to Neo4j: {e}") from e

    def run_query(self, cypher: str, params: dict | None = None) -> list[dict]:
        try:
            with self._driver.session() as session:
                result = session.run(cypher, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            raise GraphError(f"Cypher query failed: {e}") from e

    def upsert_entity(self, label: str, name: str, properties: dict | None = None) -> None:
        props = properties or {}
        cypher = f"""
        MERGE (e:{label} {{name: $name}})
        SET e += $props
        """
        self.run_query(cypher, {"name": name, "props": props})

    def upsert_relationship(
        self,
        from_label: str,
        from_name: str,
        rel_type: str,
        to_label: str,
        to_name: str,
        properties: dict | None = None,
    ) -> None:
        props = properties or {}
        cypher = f"""
        MERGE (a:{from_label} {{name: $from_name}})
        MERGE (b:{to_label} {{name: $to_name}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r += $props
        """
        self.run_query(cypher, {"from_name": from_name, "to_name": to_name, "props": props})

    def find_related_entities(self, entity_name: str, depth: int = 2) -> list[dict]:
        cypher = """
        MATCH path = (start {name: $name})-[*1..$depth]-(related)
        RETURN DISTINCT related.name AS name, labels(related) AS labels,
               [r in relationships(path) | type(r)] AS relationship_types
        LIMIT 50
        """
        return self.run_query(cypher, {"name": entity_name, "depth": depth})

    def get_entity_context(self, entity_names: list[str]) -> str:
        cypher = """
        MATCH (e) WHERE e.name IN $names
        OPTIONAL MATCH (e)-[r]-(neighbor)
        RETURN e.name AS entity, labels(e) AS type,
               collect(DISTINCT {rel: type(r), neighbor: neighbor.name}) AS connections
        """
        rows = self.run_query(cypher, {"names": entity_names})
        lines = []
        for row in rows:
            connections = [f"{c['rel']} -> {c['neighbor']}" for c in row["connections"] if c["neighbor"]]
            lines.append(f"{row['entity']} ({', '.join(row['type'])}): {'; '.join(connections)}")
        return "\n".join(lines)

    def clear_all(self) -> None:
        self.run_query("MATCH (n) DETACH DELETE n")
        logger.warning("Cleared entire Neo4j graph")

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
