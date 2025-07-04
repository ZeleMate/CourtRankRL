# Ez a szkript felelős a dokumentumok metaadataiból egy hálózati gráf felépítéséért,
# amely a dokumentumok, jogszabályok és bíróságok közötti kapcsolatokat reprezentálja.
import os
import sys
import argparse
import pandas as pd
import networkx as nx
import json
from tqdm import tqdm
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone

# Loggolás alapbeállítása (a config.py felülírhatja, ha ott is van basicConfig)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Konfigurációs beállítások importálása
from configs import config

# --- Segédfüggvények ---

def parse_list_string(data_string, separator=';'):
    """Egy stringként tárolt, elválasztóval tagolt listaelemet alakít át valódi listává."""
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
    """Alapvető ellenőrzés a dokumentumazonosítókra (legyen string és ne legyen üres)."""
    return isinstance(doc_id, str) and bool(doc_id.strip())

# --- Fő gráfépítő logika ---

def build_graph(df, stop_jogszabalyok):
    """Felépíti a NetworkX gráfot a bemeneti DataFrame alapján - optimalizált verzió."""
    G = nx.DiGraph() # Irányított gráf létrehozása
    logging.info("Optimalizált gráfépítés megkezdése...")

    # Batch műveletek előkészítése
    batch_nodes = []
    batch_edges = []
    edge_weights = defaultdict(int)
    
    for _, doc_data in tqdm(df.iterrows(), total=df.shape[0], desc="Gráf építése"): # tqdm progress bar
        doc_id = doc_data.get('doc_id')
        if not is_valid_doc_id(doc_id):
            logging.debug(f"Érvénytelen vagy hiányzó doc_id ({doc_id}), a sor kihagyva.")
            continue

        # Adatmezők kinyerése és listává alakítása
        jogszabalyhelyek = parse_list_string(doc_data.get('Jogszabalyhelyek', ''))
        kapcsolodo_hatarozatok = parse_list_string(doc_data.get('KapcsolodoHatarozatok', ''))
        kapcsolodo_birosagok = parse_list_string(doc_data.get('AllKapcsolodoBirosag', ''))
        
        # Dokumentum csomópont batch-hez
        node_attrs = {
            "type": "dokumentum",
            "jogterulet": doc_data.get('jogterulet') if pd.notna(doc_data.get('jogterulet')) else None,
            "birosag": doc_data.get('birosag') if pd.notna(doc_data.get('birosag')) else None,
            "ev": int(doc_data.get('HatarozatEve')) if pd.notna(doc_data.get('HatarozatEve')) and str(doc_data.get('HatarozatEve')).isdigit() else None,
        }
        node_attrs = {k: v for k, v in node_attrs.items() if v is not None}
        batch_nodes.append((doc_id, node_attrs))

        # Hivatkozások batch-hez
        for hatarozat_id in kapcsolodo_hatarozatok:
            if is_valid_doc_id(hatarozat_id):
                batch_nodes.append((hatarozat_id, {"type": "dokumentum"}))
                edge_key = (doc_id, hatarozat_id, "hivatkozik")
                edge_weights[edge_key] += 1

        # Bírósági kapcsolatok batch-hez
        for birosag_name in kapcsolodo_birosagok:
            if birosag_name and isinstance(birosag_name, str):
                birosag_node_id = f"birosag_{birosag_name.lower().replace(' ', '_')}"
                batch_nodes.append((birosag_node_id, {"type": "birosag", "name": birosag_name}))
                edge_key = (doc_id, birosag_node_id, "targyalta")
                edge_weights[edge_key] += 1

        # Jogszabályhelyek batch-hez
        for jsz in jogszabalyhelyek:
            if jsz and isinstance(jsz, str) and jsz not in stop_jogszabalyok:
                jsz_node_id = f"jogszabaly_{jsz.lower().replace(' ', '_').replace('.', '').replace('§', 'par').replace('(', '').replace(')', '')}"
                batch_nodes.append((jsz_node_id, {"type": "jogszabaly", "reference": jsz}))
                edge_key = (doc_id, jsz_node_id, "hivatkozik_jogszabalyra")
                edge_weights[edge_key] += 1

    # Batch csomópont hozzáadás - duplikátumok kezelése
    logging.info("Csomópontok batch hozzáadása...")
    unique_nodes = {}
    for node_id, attrs in batch_nodes:
        if node_id in unique_nodes:
            # Attribútumok egyesítése
            unique_nodes[node_id].update({k: v for k, v in attrs.items() if v is not None})
        else:
            unique_nodes[node_id] = attrs
    
    G.add_nodes_from(unique_nodes.items())

    # Batch él hozzáadás súlyokkal
    logging.info("Élek batch hozzáadása...")
    edges_with_attrs = []
    for (u, v, rel_type), weight in edge_weights.items():
        edges_with_attrs.append((u, v, {"relation_type": rel_type, "weight": weight}))
    
    G.add_edges_from(edges_with_attrs)

    logging.info(f"Optimalizált gráfépítés befejezve. Csomópontok száma: {G.number_of_nodes()}, Élek száma: {G.number_of_edges()}")
    return G

def save_graph(G, json_path, graphml_path):
    """Elmenti a gráfot JSON és GraphML formátumban is."""
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

def save_graph_metadata(G, stop_jogszabalyok_set, output_path):
    """Elmenti a generált gráf metaadatait egy JSON fájlba."""
    logging.info(f"Saving graph metadata to {output_path}...")
    try:
        relation_types = {data.get('relation_type') for _, _, data in G.edges(data=True) if 'relation_type' in data}
        
        metadata = {
            "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "stop_jogszabalyok_count": len(stop_jogszabalyok_set),
            "stop_jogszabalyok_list": sorted(list(stop_jogszabalyok_set)),
            "relation_types": sorted(list(relation_types))
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        logging.info("Graph metadata saved.")
    except Exception as e:
        logging.error(f"Failed to save graph metadata to {output_path}: {e}")

def determine_stop_jogszabalyok(df, column_name='Jogszabalyhelyek', threshold_percentage=0.005):
    """Meghatározza a gyakori jogszabályhelyeket, amelyek "stop szavakként" funkcionálnak."""
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
    """Parancssori argumentumok feldolgozása."""
    parser = argparse.ArgumentParser(description="NetworkX Gráf Építése Dokumentum Metaadatokból")
    parser.add_argument(
        "--input",
        type=str,
        default=config.PROCESSED_PARQUET_DATA_PATH,
        help="Path to the input Parquet file (default: from config.py)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=config.GRAPH_OUTPUT_JSON_PATH,
        help="Path to save the output graph in JSON format (default: from config.py)"
    )
    parser.add_argument(
        "--output-graphml",
        type=str,
        default=config.GRAPH_OUTPUT_GRAPHML_PATH,
        help="Path to save the output graph in GraphML format (default: from config.py)"
    )
    parser.add_argument(
        "--output-metadata",
        type=str,
        default=config.GRAPH_METADATA_PATH,
        help="Path to save the graph metadata JSON (default: from config.py)"
    )
    parser.add_argument(
        "--stopword-threshold",
        type=float,
        default=0.01,  # Default to 1%
        help="Threshold percentage (0.0 to 1.0) for determining stop jogszabalyok (default: 0.01 = 1%)"
    )
    parser.add_argument(
        "--stopword-column",
        type=str,
        default='Jogszabalyhelyek',
        help="Column name containing legal references for stop word analysis (default: Jogszabalyhelyek)"
    )
    return parser.parse_args()

def main():
    """Fő függvény az adatok betöltéséhez, a gráf felépítéséhez és a kimenetek mentéséhez."""
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
    save_graph_metadata(G, stop_jogszabalyok, args.output_metadata)
    logging.info("Graph building process finished.")

if __name__ == "__main__":
    main()