import os
import sys
import argparse
import pandas as pd
import networkx as nx
import json
from tqdm import tqdm
import logging
from collections import Counter
from datetime import datetime

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
    # Define fallbacks or exit if config is essential
    class MockConfig:
        PROCESSED_PARQUET_DATA_PATH = 'processed_data/processed_documents.parquet'
        GRAPH_OUTPUT_GML_PATH = 'output_graph.graphml'
        GRAPH_OUTPUT_JSON_PATH = 'output_graph.json'
    config = MockConfig()

# --- Helper Functions ---

def parse_list_string(data_string, separator=';'):
    """Parses a string containing a list of items separated by a separator or a JSON list."""
    if not data_string or pd.isna(data_string):
        return []
    try:
        if isinstance(data_string, str) and data_string.strip().startswith('[') and data_string.strip().endswith(']'):
            parsed_list = json.loads(data_string)
            if isinstance(parsed_list, list):
                return [str(item).strip() for item in parsed_list if item]
        if isinstance(data_string, str):
            return [item.strip() for item in data_string.split(separator) if item.strip()]
        if isinstance(data_string, list):
            return [str(item).strip() for item in data_string if item]
    except json.JSONDecodeError:
        if isinstance(data_string, str):
            return [item.strip() for item in data_string.split(separator) if item.strip()]
    except TypeError:
        logging.warning(f"Could not parse data: {data_string}. Returning empty list.")
        return []
    logging.warning(f"Unexpected data type or format for parsing: {data_string}. Returning empty list.")
    return []

def is_valid_doc_id(doc_id):
    """Basic validation for document IDs (example). Checks if non-empty string."""
    return isinstance(doc_id, str) and bool(doc_id.strip())

# --- Main Graph Building Logic ---

def build_graph(df, stop_jogszabalyok):
    """Builds the NetworkX graph based on the DataFrame."""
    G = nx.DiGraph()
    logging.info("Starting graph construction...")

    for _, doc_data in tqdm(df.iterrows(), total=df.shape[0], desc="Building Graph"):
        doc_id = doc_data.get('doc_id')
        if not is_valid_doc_id(doc_id):
            logging.warning(f"Skipping row due to invalid or missing doc_id: {doc_data.get('doc_id')}")
            continue

        # Extract data fields
        jogszabalyhelyek_str = doc_data.get('Jogszabalyhelyek', '')
        kapcsolodo_hatarozatok_str = doc_data.get('KapcsolodoHatarozatok', '')
        kapcsolodo_birosagok_str = doc_data.get('AllKapcsolodoBirosag', '')
        jogterulet = doc_data.get('jogterulet')
        birosag = doc_data.get('birosag')
        ev = doc_data.get('HatarozatEve')  # Changed from 'ev'

        # Parse list-like strings
        jogszabalyhelyek = parse_list_string(jogszabalyhelyek_str)
        kapcsolodo_hatarozatok = parse_list_string(kapcsolodo_hatarozatok_str)
        kapcsolodo_birosagok = parse_list_string(kapcsolodo_birosagok_str)

        # 1. Add/Update Document Node with Year
        node_attrs = {
            "type": "dokumentum",
            "jogterulet": jogterulet if pd.notna(jogterulet) else None,
            "birosag": birosag if pd.notna(birosag) else None,
            "ev": int(ev) if pd.notna(ev) and str(ev).isdigit() else None,
        }
        node_attrs = {k: v for k, v in node_attrs.items() if v is not None}
        if not G.has_node(doc_id):
            G.add_node(doc_id, **node_attrs)
        else:
            current_attrs = G.nodes[doc_id]
            current_attrs.update(node_attrs)
            nx.set_node_attributes(G, {doc_id: current_attrs})

        def add_or_increment_edge(u, v, rel_type):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, relation_type=rel_type, weight=1)

        # 2. Add References to Other Decisions (Directed Edge)
        for hatarozat_id in kapcsolodo_hatarozatok:
            if is_valid_doc_id(hatarozat_id):
                if not G.has_node(hatarozat_id):
                    G.add_node(hatarozat_id, type="dokumentum")
                add_or_increment_edge(doc_id, hatarozat_id, "hivatkozik")

        # 3. Add Court Connections (Directed Edge from doc to court)
        for birosag_name in kapcsolodo_birosagok:
            if birosag_name and isinstance(birosag_name, str):
                birosag_node_id = f"birosag_{birosag_name.lower().replace(' ', '_')}"
                if not G.has_node(birosag_node_id):
                    G.add_node(birosag_node_id, type="birosag", name=birosag_name)
                add_or_increment_edge(doc_id, birosag_node_id, "targyalta")

        # 4. Handle Legal References (Directed Edge from doc to legal ref)
        for jsz in jogszabalyhelyek:
            if jsz and isinstance(jsz, str) and jsz not in stop_jogszabalyok:
                jsz_node_id = f"jogszabaly_{jsz.lower().replace(' ', '_').replace('.', '').replace('§', 'par').replace('(', '').replace(')', '')}"
                if not G.has_node(jsz_node_id):
                    G.add_node(jsz_node_id, type="jogszabaly", reference=jsz)
                add_or_increment_edge(doc_id, jsz_node_id, "hivatkozik_jogszabalyra")

    logging.info(f"Graph construction finished. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def save_graph(G, gml_path, json_path):
    """Saves the graph in GML and JSON formats."""
    try:
        logging.info(f"Saving graph to {gml_path} (GML format)...")
        nx.write_graphml(G, gml_path)
        logging.info("GML Graph saved.")
    except Exception as e:
        logging.error(f"Failed to save graph to GML ({gml_path}): {e}")

    try:
        logging.info(f"Saving graph to {json_path} (JSON node-link format)...")
        graph_data = nx.node_link_data(G)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=4)
        logging.info("JSON Graph saved.")
    except Exception as e:
        logging.error(f"Failed to save graph to JSON ({json_path}): {e}")

def save_graph_metadata(G, stop_jogszabalyok_len, output_path):
    """Saves metadata about the generated graph to a JSON file."""
    logging.info(f"Saving graph metadata to {output_path}...")
    try:
        relation_types = set()
        for _, _, data in G.edges(data=True):
            if 'relation_type' in data:
                relation_types.add(data['relation_type'])

        metadata = {
            "generation_timestamp_utc": datetime.utcnow().isoformat(),
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "stop_jogszabalyok_count": stop_jogszabalyok_len,
            "relation_types": sorted(list(relation_types))
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        logging.info("Graph metadata saved.")
    except Exception as e:
        logging.error(f"Failed to save graph metadata to {output_path}: {e}")

def determine_stop_jogszabalyok(df, column_name='Jogszabalyhelyek', threshold_percentage=0.5):
    """Determines frequent legal references to be used as stop words."""
    logging.info(f"Determining stop jogszabalyok from column '{column_name}' with threshold {threshold_percentage*100}%...")
    all_references = []
    if column_name not in df.columns:
        logging.warning(f"Column '{column_name}' not found in DataFrame. Cannot determine stop words.")
        return set()

    for references_str in df[column_name].dropna():
        all_references.extend(parse_list_string(references_str))

    if not all_references:
        logging.warning("No legal references found to analyze for stop words.")
        return set()

    reference_counts = Counter(all_references)
    total_documents = len(df)
    threshold_count = total_documents * threshold_percentage

    stop_set = {ref for ref, count in reference_counts.items() if count > threshold_count}
    logging.info(f"Found {len(stop_set)} stop jogszabalyok occurring in more than {threshold_percentage*100}% of documents.")
    return stop_set

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Build NetworkX Graph from Document Metadata")
    parser.add_argument("--doc_path", type=str, default=config.PROCESSED_PARQUET_DATA_PATH,
                        help="Path to the processed documents Parquet file")
    parser.add_argument("--gml_output", type=str, default=getattr(config, 'GRAPH_OUTPUT_GML_PATH', 'output_graph.graphml'),
                        help="Output path for the graph in GML format")
    parser.add_argument("--json_output", type=str, default=getattr(config, 'GRAPH_OUTPUT_JSON_PATH', 'output_graph.json'),
                        help="Output path for the graph in JSON node-link format")
    parser.add_argument("--metadata_output", type=str, default=None,
                        help="Output path for the graph metadata JSON file. Defaults to [json_output_path]_metadata.json")
    return parser.parse_args()

def main():
    """Main function to orchestrate the graph building process."""
    args = parse_args()

    logging.info(f"Loading documents from: {args.doc_path}")
    try:
        df = pd.read_parquet(args.doc_path)
        logging.info(f"Loaded {len(df)} documents from Parquet.")
    except FileNotFoundError:
        logging.error(f"Error: Parquet file not found at {args.doc_path}. Exiting.")
        return
    except Exception as e:
        logging.error(f"Error loading Parquet file {args.doc_path}: {e}. Exiting.")
        return

    if df.empty:
        logging.error(f"Error: No documents found in {args.doc_path}. Exiting.")
        return

    dynamic_stop_jogszabalyok = determine_stop_jogszabalyok(df, threshold_percentage=0.5)

    G = build_graph(df, dynamic_stop_jogszabalyok)
    save_graph(G, args.gml_output, args.json_output)

    metadata_output_path = args.metadata_output
    if metadata_output_path is None:
        base, _ = os.path.splitext(args.json_output)
        metadata_output_path = f"{base}_metadata.json"

    save_graph_metadata(G, len(dynamic_stop_jogszabalyok), metadata_output_path)

    logging.info("Graph building process finished.")

if __name__ == "__main__":
    main()
