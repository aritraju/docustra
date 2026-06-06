"""Pre-built Cypher queries for common knowledge graph lookups."""
from docustra.storage.graph_store import GraphStore


def get_all_entities(graph: GraphStore) -> list[dict]:
    return graph.run_query("MATCH (n) RETURN n.name AS name, labels(n) AS types LIMIT 200")


def get_entity_neighborhood(graph: GraphStore, entity: str, depth: int = 2) -> list[dict]:
    return graph.find_related_entities(entity, depth=depth)


def get_relationship_types(graph: GraphStore) -> list[str]:
    rows = graph.run_query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel_type")
    return [r["rel_type"] for r in rows]


def get_shortest_path(graph: GraphStore, from_entity: str, to_entity: str) -> list[dict]:
    cypher = """
    MATCH (a {name: $from}), (b {name: $to})
    MATCH path = shortestPath((a)-[*..6]-(b))
    RETURN [node in nodes(path) | node.name] AS path_nodes,
           [rel in relationships(path) | type(rel)] AS path_rels
    """
    return graph.run_query(cypher, {"from": from_entity, "to": to_entity})


def get_entity_stats(graph: GraphStore) -> dict:
    node_count = graph.run_query("MATCH (n) RETURN count(n) AS count")[0]["count"]
    rel_count = graph.run_query("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
    return {"nodes": node_count, "relationships": rel_count}
