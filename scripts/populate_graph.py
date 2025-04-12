"""
Script to extract entities and relationships from documents and populate the graph database.
"""
import os
import sys
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader.legal_docs import load_documents_with_metadata # Use metadata loader
from src.graph.extractor import GraphExtractor
from src.graph.graph_db import GraphDBConnector
from configs import config
from tqdm import tqdm # For progress bar

def parse_args():
    parser = argparse.ArgumentParser(description="Populate Graph Database from Documents")
    parser.add_argument("--doc_path", type=str, default=config.RAW_DATA_PATH,
                        help="Path to folder containing documents")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of documents to process before committing to DB")
    # Add arguments for specific NLP models if needed
    return parser.parse_args()

def main():
    args = parse_args()

    print("Initializing components...")
    # Load documents with metadata (filename can serve as initial ID)
    documents_with_meta = load_documents_with_metadata(args.doc_path)
    if not documents_with_meta:
        print("No documents found. Exiting.")
        return

    extractor = GraphExtractor() # Initialize NLP extractor
    db_connector = GraphDBConnector() # Initialize DB connector

    print(f"Starting graph population from {len(documents_with_meta)} documents...")

    nodes_batch = []
    rels_batch = []

    try:
        for i, doc_data in enumerate(tqdm(documents_with_meta, desc="Processing Documents")):
            doc_text = doc_data.get('text', '')
            # Use filename or a dedicated ID from metadata
            doc_id = doc_data.get('metadata', {}).get('filename', f"doc_{i}")

            if not doc_text:
                print(f"Warning: Skipping document {doc_id} due to empty content.")
                continue

            # Extract graph elements from the document
            graph_elements = extractor.process_document(doc_id, doc_text)

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
                        rel['start_node_label'], rel['start_node_id'],
                        rel['end_node_label'], rel['end_node_id'],
                        rel['type'], rel.get('properties')
                    )
                print(f"Batch committed. Nodes: {len(nodes_batch)}, Relationships: {len(rels_batch)}")
                nodes_batch = []
                rels_batch = []

    except Exception as e:
        print(f"\nAn error occurred during graph population: {e}")
    finally:
        db_connector.close()
        print("Graph population finished. Database connection closed.")

if __name__ == "__main__":
    main()
