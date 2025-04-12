"""
Module for interacting with the graph database (e.g., Neo4j).
"""
from neo4j import GraphDatabase # Example using Neo4j driver
from typing import List, Dict, Any
from ...configs import config # Use relative import

class GraphDBConnector:
    """Handles connection and operations with the graph database."""

    def __init__(self, uri: str = config.GRAPH_DB_URI, user: str = config.GRAPH_DB_USER, password: str = config.GRAPH_DB_PASSWORD):
        """Initialize the database connection."""
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the database connection."""
        self._driver.close()

    def run_query(self, query: str, parameters: Dict = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def add_node(self, label: str, properties: Dict):
        """Add a node to the graph."""
        # Implement logic to create or merge a node
        # Example: MERGE (n:label {id: $id}) SET n += $props
        query = f"MERGE (n:{label} {{ {config.PROP_ID}: $id }}) SET n += $props RETURN n"
        props_with_id = properties.copy()
        props_with_id['id'] = properties.get(config.PROP_ID, None) # Ensure ID is present
        if not props_with_id['id']:
            # Handle cases where ID might be missing or needs generation
            print(f"Warning: Node of type {label} is missing an ID property.")
            return None
        parameters = {'id': props_with_id['id'], 'props': properties}
        return self.run_query(query, parameters)

    def add_relationship(self, start_node_label: str, start_node_id: Any,
                         end_node_label: str, end_node_id: Any,
                         rel_type: str, properties: Dict = None):
        """Add a relationship between two nodes."""
        # Implement logic to create or merge a relationship
        # Example: MATCH (a:start_label {id: $start_id}), (b:end_label {id: $end_id})
        #          MERGE (a)-[r:REL_TYPE]->(b) SET r += $props
        query = (
            f"MATCH (a:{start_node_label} {{ {config.PROP_ID}: $start_id }}), "
            f"(b:{end_node_label} {{ {config.PROP_ID}: $end_id }}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
        )
        if properties:
            query += "SET r += $props "
        query += "RETURN type(r)"

        parameters = {
            'start_id': start_node_id,
            'end_id': end_node_id,
            'props': properties or {}
        }
        return self.run_query(query, parameters)

    def get_neighbors(self, node_label: str, node_id: Any, relationship_types: List[str] = None) -> List[Dict[str, Any]]:
        """Retrieve neighbors of a given node."""
        rel_match = ""
        if relationship_types:
            rel_match = ":" + "|".join(relationship_types)

        query = (
            f"MATCH (a:{node_label} {{ {config.PROP_ID}: $node_id }})-[r{rel_match}]-(neighbor) "
            "RETURN neighbor, type(r) as relationship"
        )
        parameters = {'node_id': node_id}
        return self.run_query(query, parameters)

    # Add more specific query methods as needed (e.g., find_cited_by, find_applying_statutes)

