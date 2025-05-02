import os
import sys
import argparse
import pandas as pd
import networkx as nx
import json
from tqdm import tqdm
import logging
from collections import Counter
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import configuration settings
try:
    from configs import config
except ImportError:
    logging.error("Could not import config. Make sure configs/config.py exists.")
    # Define minimal fallbacks
    class MockConfig:
        PROCESSED_PARQUET_DATA_PATH = 'processed_data/processed_documents.parquet'
        GRAPH_OUTPUT_JSON_PATH = 'processed_data/graph_data/graph.json'
        GRAPH_OUTPUT_GRAPHML_PATH = 'processed_data/graph_data/graph.graphml'
        GRAPH_METADATA_PATH = 'processed_data/graph_data/graph_metadata.json'
    config = MockConfig()

# --- Helper Functions ---

def parse_list_string(data_string, separator=';'):
    """Parses a string containing a list of items separated by a separator."""
    if not data_string or pd.isna(data_string):
        return []
    
    # Handle JSON list format
    if isinstance(data_string, str) and data_string.strip().startswith('[') and data_string.strip().endswith(']'):
        try:
            parsed_list = json.loads(data_string)
            if isinstance(parsed_list, list):
                return [str(item).strip() for item in parsed_list if item]
        except json.JSONDecodeError:
            pass
    
    # Handle regular string with separator
    if isinstance(data_string, str):
        return [item.strip() for item in data_string.split(separator) if item.strip()]
    
    # Handle direct list input
    if isinstance(data_string, list):
        return [str(item).strip() for item in data_string if item]
    
    return []

def is_valid_doc_id(doc_id):
    """Basic validation for document IDs."""
    return isinstance(doc_id, str) and bool(doc_id.strip())

# --- Main Graph Building Logic ---

def build_graph(df, stop_jogszabalyok):
    """Builds the NetworkX graph based on the DataFrame."""
    G = nx.DiGraph()
    logging.info("Starting graph construction...")

    for _, doc_data in tqdm(df.iterrows(), total=df.shape[0], desc="Building Graph"):
        doc_id = doc_data.get('doc_id')
        if not is_valid_doc_id(doc_id):
            continue

        # Extract data fields
        jogszabalyhelyek = parse_list_string(doc_data.get('Jogszabalyhelyek', ''))
        kapcsolodo_hatarozatok = parse_list_string(doc_data.get('KapcsolodoHatarozatok', ''))
        kapcsolodo_birosagok = parse_list_string(doc_data.get('AllKapcsolodoBirosag', ''))
        
        # Add document node with metadata
        node_attrs = {
            "type": "dokumentum",
            "jogterulet": doc_data.get('jogterulet') if pd.notna(doc_data.get('jogterulet')) else None,
            "birosag": doc_data.get('birosag') if pd.notna(doc_data.get('birosag')) else None,
            "ev": int(doc_data.get('HatarozatEve')) if pd.notna(doc_data.get('HatarozatEve')) and str(doc_data.get('HatarozatEve')).isdigit() else None,
        }
        node_attrs = {k: v for k, v in node_attrs.items() if v is not None}
        
        if not G.has_node(doc_id):
            G.add_node(doc_id, **node_attrs)
        else:
            nx.set_node_attributes(G, {doc_id: {**G.nodes[doc_id], **node_attrs}})

        def add_or_increment_edge(u, v, rel_type):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, relation_type=rel_type, weight=1)

        # Add references to other decisions
        for hatarozat_id in kapcsolodo_hatarozatok:
            if is_valid_doc_id(hatarozat_id):
                if not G.has_node(hatarozat_id):
                    G.add_node(hatarozat_id, type="dokumentum")
                add_or_increment_edge(doc_id, hatarozat_id, "hivatkozik")

        # Add court connections
        for birosag_name in kapcsolodo_birosagok:
            if birosag_name and isinstance(birosag_name, str):
                birosag_node_id = f"birosag_{birosag_name.lower().replace(' ', '_')}"
                if not G.has_node(birosag_node_id):
                    G.add_node(birosag_node_id, type="birosag", name=birosag_name)
                add_or_increment_edge(doc_id, birosag_node_id, "targyalta")

        # Handle legal references
        for jsz in jogszabalyhelyek:
            if jsz and isinstance(jsz, str) and jsz not in stop_jogszabalyok:
                jsz_node_id = f"jogszabaly_{jsz.lower().replace(' ', '_').replace('.', '').replace('§', 'par').replace('(', '').replace(')', '')}"
                if not G.has_node(jsz_node_id):
                    G.add_node(jsz_node_id, type="jogszabaly", reference=jsz)
                add_or_increment_edge(doc_id, jsz_node_id, "hivatkozik_jogszabalyra")

    logging.info(f"Graph construction finished. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def save_graph(G, json_path, graphml_path):
    """Saves the graph in both JSON and GraphML formats."""
    # Save JSON format
    try:
        logging.info(f"Saving graph to {json_path} (JSON format)...")
        graph_data = nx.node_link_data(G)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=4)
        logging.info("Graph saved successfully in JSON format.")
    except Exception as e:
        logging.error(f"Failed to save graph to JSON ({json_path}): {e}")

    # Save GraphML format
    try:
        logging.info(f"Saving graph to {graphml_path} (GraphML format)...")
        os.makedirs(os.path.dirname(graphml_path), exist_ok=True)
        nx.write_graphml(G, graphml_path)
        logging.info("Graph saved successfully in GraphML format.")
    except Exception as e:
        logging.error(f"Failed to save graph to GraphML ({graphml_path}): {e}")

def save_graph_metadata(G, stop_jogszabalyok_len, output_path):
    """Saves metadata about the generated graph to a JSON file."""
    logging.info(f"Saving graph metadata to {output_path}...")
    try:
        relation_types = {data.get('relation_type') for _, _, data in G.edges(data=True) if 'relation_type' in data}
        
        metadata = {
            "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "stop_jogszabalyok_count": stop_jogszabalyok_len,
            "relation_types": sorted(list(relation_types))
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        logging.info("Graph metadata saved.")
    except Exception as e:
        logging.error(f"Failed to save graph metadata to {output_path}: {e}")

def determine_stop_jogszabalyok(df, column_name='Jogszabalyhelyek', threshold_percentage=0.005):
    """Determines frequent legal references to be used as stop words."""
    logging.info(f"Determining stop jogszabalyok with threshold {threshold_percentage*100}%...")
    
    if column_name not in df.columns:
        logging.warning(f"Column '{column_name}' not found in DataFrame. Cannot determine stop words.")
        return set()
    
    all_references = []
    for references_str in df[column_name].dropna():
        all_references.extend(parse_list_string(references_str))
    
    if not all_references:
        logging.warning("No legal references found to analyze.")
        return set()
    
    reference_counts = Counter(all_references)
    threshold_count = len(df) * threshold_percentage
    stop_set = {ref for ref, count in reference_counts.items() if count > threshold_count}
    
    logging.info(f"Found {len(stop_set)} stop jogszabalyok occurring in more than {threshold_percentage*100}% of documents.")
    return stop_set

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Build NetworkX Graph from Document Metadata")
    parser.add_argument(
        "--input",
        type=str,
        default=getattr(config, 'PROCESSED_PARQUET_DATA_PATH', 'processed_data/processed_documents.parquet'),
        help="Path to the input Parquet file"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=getattr(config, 'GRAPH_OUTPUT_JSON_PATH', 'processed_data/graph_data/graph.json'),
        help="Path to save the output graph in JSON format"
    )
    parser.add_argument(
        "--output-graphml",
        type=str,
        default=getattr(config, 'GRAPH_OUTPUT_GRAPHML_PATH', 'processed_data/graph_data/graph.graphml'),
        help="Path to save the output graph in GraphML format"
    )
    parser.add_argument(
        "--output-metadata",
        type=str,
        default=getattr(config, 'GRAPH_METADATA_PATH', 'processed_data/graph_data/graph_metadata.json'),
        help="Path to save the graph metadata JSON"
    )
    parser.add_argument(
        "--stopword-threshold",
        type=float,
        default=0.005,  # Default to 0.5%
        help="Threshold percentage (0.0 to 1.0) for determining stop jogszabalyok (default: 0.005 = 0.5%)"
    )
    parser.add_argument(
        "--stopword-column",
        type=str,
        default='Jogszabalyhelyek',
        help="Column name containing legal references for stop word analysis"
    )
    return parser.parse_args()

def main():
    """Main function to load data, build graph, and save outputs."""
    args = parse_args()

    # Validate threshold
    if not 0.0 <= args.stopword_threshold <= 1.0:
        logging.error("Stopword threshold must be between 0.0 and 1.0.")
        sys.exit(1)

    # Load data
    try:
        df = pd.read_parquet(args.input)
        logging.info(f"Loaded {len(df)} documents from {args.input}.")
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Process data and build graph
    stop_jogszabalyok = determine_stop_jogszabalyok(df, args.stopword_column, args.stopword_threshold)
    G = build_graph(df, stop_jogszabalyok)

    # Save outputs
    save_graph(G, args.output_json, args.output_graphml)
    save_graph_metadata(G, len(stop_jogszabalyok), args.output_metadata)
    logging.info("Graph building process finished.")

if __name__ == "__main__":
    main()