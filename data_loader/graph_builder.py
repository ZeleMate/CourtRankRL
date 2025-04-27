"""
Script to build a NetworkX graph from document metadata stored in a Parquet file.

This script reads document metadata (like court, legal area, and other metadata fields)
from a specified Parquet file. It uses placeholder classes (`GraphExtractor`, `GraphDBConnector`)
to represent the extraction and graph storage logic. The `GraphExtractor` currently only
processes metadata columns, not the document text itself. The `GraphDBConnector` uses
NetworkX to build an in-memory graph.

The script processes documents in batches and finally saves the resulting graph
to a GraphML file named 'output_graph.graphml'.

Command-line arguments control the input Parquet file path, batch size, and
whether to clear the graph before starting.
"""
import os
import sys
import argparse
import pandas as pd # Import pandas
import networkx as nx
import json

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import configuration settings
from configs import config
from tqdm import tqdm # For progress bar


class GraphExtractor:
    """Extracts graph nodes and relationships from document metadata.

    This class processes individual document data dictionaries to identify
    entities (like documents, courts, legal areas) and their connections.
    """
    def __init__(self):
        """Initializes the GraphExtractor."""
        print("Initializing Metadata GraphExtractor...")
        pass

    def process_document(self, doc_data):
        """
        Extracts graph elements (nodes and relationships) from a single document's metadata.

        Args:
            doc_data (dict): A dictionary representing a single document's data,
                             typically a row from the input DataFrame.

        Returns:
            dict: A dictionary containing two lists: 'nodes' and 'relationships'.
                  Each node is a dict with 'label' and 'properties'.
                  Each relationship is a dict specifying start/end nodes, type, and properties.
                  Returns empty lists if essential data like 'doc_id' is missing.
        """
        nodes = []
        relationships = []

        # Dokumentum csomópont (mindig létrehozzuk)
        doc_id = doc_data.get('doc_id')
        if not doc_id:
            print(f"Warning: Missing 'doc_id' in data: {doc_data}. Skipping document.")
            return {'nodes': [], 'relationships': []}

        doc_props = {'id': doc_id}
        if 'text' in doc_data and pd.notna(doc_data['text']):
            doc_props['text_length'] = len(doc_data['text'])
        nodes.append({'label': 'Document', 'properties': doc_props})

        # Bíróság csomópont és kapcsolat
        birosag = doc_data.get('birosag')
        if birosag and pd.notna(birosag):
            court_id = f"court_{birosag.lower().replace(' ', '_')}"
            nodes.append({'label': 'Court', 'properties': {'id': court_id, 'name': birosag}})
            relationships.append({
                'start_node_label': 'Document', 'start_node_id_prop': 'id', 'start_node_id_val': doc_id,
                'end_node_label': 'Court', 'end_node_id_prop': 'id', 'end_node_id_val': court_id,
                'type': 'ISSUED_BY', 'properties': {}
            })

        # Jogterület csomópont és kapcsolat
        jogterulet = doc_data.get('jogterulet')
        if jogterulet and pd.notna(jogterulet):
            area_id = f"area_{jogterulet.lower().replace(' ', '_')}"
            nodes.append({'label': 'LegalArea', 'properties': {'id': area_id, 'name': jogterulet}})
            relationships.append({
                'start_node_label': 'Document', 'start_node_id_prop': 'id', 'start_node_id_val': doc_id,
                'end_node_label': 'LegalArea', 'end_node_id_prop': 'id', 'end_node_id_val': area_id,
                'type': 'BELONGS_TO_AREA', 'properties': {}
            })

        # Metadata feldolgozása
        metadata = doc_data.get('metadata')
        if metadata and pd.notna(metadata):
            try:
                metadata_dict = json.loads(metadata)
                for key, value in metadata_dict.items():
                    if value and isinstance(value, str):
                        meta_node_id = f"meta_{key.lower()}_{value.lower().replace(' ', '_')}"
                        nodes.append({'label': 'Metadata', 'properties': {'id': meta_node_id, 'key': key, 'value': value}})
                        relationships.append({
                            'start_node_label': 'Document', 'start_node_id_prop': 'id', 'start_node_id_val': doc_id,
                            'end_node_label': 'Metadata', 'end_node_id_prop': 'id', 'end_node_id_val': meta_node_id,
                            'type': f'HAS_METADATA_{key.upper()}', 'properties': {}
                        })
            except json.JSONDecodeError:
                print(f"Warning: Failed to decode metadata for document {doc_id}.")

        return {'nodes': nodes, 'relationships': relationships}

class GraphDBConnector:
    """Manages the connection and operations for an in-memory NetworkX graph.

    This class provides methods to add nodes and relationships to a NetworkX
    MultiDiGraph, clear the graph, save it to a file, and report basic stats.
    """
    def __init__(self):
        """Initializes the GraphDBConnector with an empty NetworkX MultiDiGraph."""
        print("Initializing NetworkX GraphDBConnector...")
        self.graph = nx.MultiDiGraph()  # Irányított, többszörös éleket is enged
        self.nodes_added = 0
        self.rels_added = 0

    def clear_database(self):
        """Clears all nodes and edges from the in-memory graph."""
        print("Clearing the in-memory NetworkX graph...")
        self.graph = nx.MultiDiGraph()
        self.nodes_added = 0
        self.rels_added = 0

    def add_node(self, label, properties):
        """Adds a node to the graph.

        Args:
            label (str): The label for the node (e.g., 'Document', 'Court').
            properties (dict): A dictionary of properties for the node.
                               Must contain an 'id' key for unique identification.
                               If 'id' is missing, a generic one is generated.
        """
        node_id = properties.get('id', None)
        if node_id is None:
            node_id = f"{label}_{self.nodes_added}"
        self.graph.add_node(node_id, label=label, **properties)
        self.nodes_added += 1

    def add_relationship(self, start_node_label, start_node_id_prop, start_node_id_val,
                         end_node_label, end_node_id_prop, end_node_id_val,
                         rel_type, properties=None):
        """Adds a directed relationship (edge) between two nodes in the graph.

        Args:
            start_node_label (str): Label of the starting node.
            start_node_id_prop (str): Property key used to identify the start node (usually 'id').
            start_node_id_val (any): Value of the identifying property for the start node.
            end_node_label (str): Label of the ending node.
            end_node_id_prop (str): Property key used to identify the end node (usually 'id').
            end_node_id_val (any): Value of the identifying property for the end node.
            rel_type (str): The type of the relationship (e.g., 'ISSUED_BY'). Used as edge key and label.
            properties (dict, optional): Additional properties for the relationship. Defaults to None.
        """
        if properties is None:
            properties = {}
        self.graph.add_edge(start_node_id_val, end_node_id_val, key=rel_type, label=rel_type, **properties)
        self.rels_added += 1

    def save_graph(self, path):
        """Saves the current graph to a file in GraphML format.

        Args:
            path (str): The file path where the graph should be saved.
        """
        print(f"Saving graph to {path} (GraphML format)...")
        nx.write_graphml(self.graph, path)
        print("Graph saved.")

    def close(self):
        """Prints the final node and relationship counts for the in-memory graph."""
        print(f"NetworkX graph in memory. Nodes: {self.nodes_added}, Relationships: {self.rels_added}")

# --- End Placeholder Classes ---

def parse_args():
    """Parses command-line arguments for the graph building script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
                            Includes doc_path, batch_size, and clear_db.
    """
    parser = argparse.ArgumentParser(description="Populate Graph Database from Documents")
    parser.add_argument("--doc_path", type=str, default=config.PROCESSED_PARQUET_DATA_PATH, # Use correct config variable
                        help="Path to the processed documents Parquet file")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing documents and embeddings")
    parser.add_argument("--clear_db", action='store_true',
                        help="Clear the graph database before populating")
    return parser.parse_args()

def main():
    """Main function to orchestrate the graph building process.

    Parses arguments, initializes the graph connector and extractor,
    loads documents from a Parquet file, processes them in batches,
    adds nodes and relationships to the graph, and saves the final graph.
    """
    args = parse_args()

    print("Initializing components...")
    # Initialize Graph Database connection
    db_connector = GraphDBConnector()
    if args.clear_db:
        print("Clearing the graph database...")
        db_connector.clear_database()
        print("Database cleared.")

    # Load documents from Parquet file
    print(f"Loading documents from: {args.doc_path}")
    try:
        df = pd.read_parquet(args.doc_path)
        # Convert DataFrame to list of dictionaries
        documents_with_meta = df.to_dict('records')
        print(f"Loaded {len(documents_with_meta)} documents from Parquet.")
    except FileNotFoundError:
        print(f"Error: Parquet file not found at {args.doc_path}. Exiting.")
        db_connector.close()
        return
    except Exception as e:
        print(f"Error loading Parquet file {args.doc_path}: {e}. Exiting.")
        db_connector.close()
        return

    if not documents_with_meta:
        print(f"Error: No documents found in {args.doc_path}. Exiting.")
        db_connector.close()
        return

    print(f"Loaded {len(documents_with_meta)} documents.")

    extractor = GraphExtractor() # Initialize NLP extractor

    print(f"Starting graph population from {len(documents_with_meta)} documents...")

    nodes_batch = []
    rels_batch = []

    try:
        # Iterate over the list of dictionaries from the DataFrame
        for i, doc_data in enumerate(tqdm(documents_with_meta, desc="Processing Documents")):
            # Access data using dictionary keys (column names from Parquet)
            doc_text = doc_data.get('text', '')
            # Use 'doc_id' column if available, otherwise generate one
            doc_id = doc_data.get('doc_id', f"doc_{i}")

            if not doc_text or pd.isna(doc_text): # Check for empty or NaN text
                print(f"Warning: Skipping document {doc_id} due to empty content.")
                continue

            # Extract graph elements from the document
            graph_elements = extractor.process_document(doc_data)

            nodes_batch.extend(graph_elements.get('nodes', []))
            rels_batch.extend(graph_elements.get('relationships', []))

            # Commit in batches
            if (i + 1) % args.batch_size == 0 or (i + 1) == len(documents_with_meta):
                print(f"\nCommitting batch {i // args.batch_size + 1} to database...")
                # --- Add nodes to DB ---
                for node in tqdm(nodes_batch, desc="Adding Nodes", leave=False):
                    db_connector.add_node(node['label'], node['properties'])
                # --- Add relationships to DB ---
                for rel in tqdm(rels_batch, desc="Adding Relationships", leave=False):
                    # Ensure start/end nodes exist before adding relationship
                    # This might require more complex batching or error handling
                    db_connector.add_relationship(
                        rel['start_node_label'], rel['start_node_id_prop'], rel['start_node_id_val'],
                        rel['end_node_label'], rel['end_node_id_prop'], rel['end_node_id_val'],
                        rel['type'], rel.get('properties')
                    )
                print(f"Batch committed. Nodes: {len(nodes_batch)}, Relationships: {len(rels_batch)}")
                nodes_batch = []
                rels_batch = []

    except Exception as e:
        print(f"\nAn error occurred during graph population: {e}")
    finally:
        # Save the graph at the end
        db_connector.save_graph("output_graph.graphml")
        db_connector.close()
        print("Graph population finished. Graph saved to output_graph.graphml.")

if __name__ == "__main__":
    main()
