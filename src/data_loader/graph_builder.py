# Ez a szkript felelős a dokumentumok metaadataiból egy hálózati gráf felépítéséért,
# amely a dokumentumok, jogszabályok és bíróságok közötti kapcsolatokat reprezentálja.
import sys
import pandas as pd
import networkx as nx
import json
import io
import pickle
from tqdm import tqdm
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Loggolás alapbeállítása
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Konfigurációs beállítások és segédprogramok importálása
try:
    from configs import config
    from src.utils.azure_blob_storage import AzureBlobStorage
except ImportError as e:
    print(f"HIBA: Modul importálása sikertelen: {e}")
    sys.exit(1)

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

def build_graph(df: pd.DataFrame, stop_jogszabalyok: set) -> nx.DiGraph:
    """Felépíti a NetworkX gráfot a bemeneti DataFrame alapján - optimalizált verzió."""
    G = nx.DiGraph()
    logging.info("Gráfépítés megkezdése...")

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

    # Gráf metaadatok hozzáadása
    G.graph['creation_timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    G.graph['document_count'] = sum(1 for _, attrs in G.nodes(data=True) if attrs.get('type') == 'dokumentum')
    G.graph['stop_jogszabalyok_count'] = len(stop_jogszabalyok)

    logging.info(f"Gráfépítés befejezve. Csomópontok: {G.number_of_nodes()}, Élek: {G.number_of_edges()}")
    return G

def determine_stop_jogszabalyok(df: pd.DataFrame, column_name='Jogszabalyhelyek', threshold_percentage=0.01) -> set:
    """Meghatározza a gyakori jogszabályhelyeket, amelyek "stop szavakként" funkcionálnak."""
    logging.info(f"Stop jogszabályok meghatározása {threshold_percentage*100:.2f}% küszöbértékkel...")
    
    if column_name not in df.columns:
        logging.warning(f"Column '{column_name}' not found in DataFrame. Cannot determine stop words.")
        return set()
    
    all_references = []
    for references_str in df[column_name].dropna():
        all_references.extend(parse_list_string(references_str))
    
    if not all_references:
        logging.warning("Nincsenek jogszabályi hivatkozások az elemzéshez.")
        return set()
    
    reference_counts = Counter(all_references)
    threshold_count = len(df) * threshold_percentage
    stop_set = {ref for ref, count in reference_counts.items() if count > threshold_count}
    
    logging.info(f"Talált {len(stop_set)} stop jogszabály, amelyek az iratok több mint {threshold_percentage*100:.2f}%-ában előfordulnak.")
    return stop_set

def main():
    """
    Fő függvény a gráf építéséhez, amely beolvassa a dokumentumokat,
    kiszámítja a hasonlóságokat, és feltölti a gráfot az Azure-ba.
    """
    logging.info("GRÁFÉPÍTŐ INDÍTÁSA AZURE BLOB STORAGE ALAPJÁN")
    
    blob_storage = AzureBlobStorage(container_name=config.AZURE_CONTAINER_NAME)
    
    input_blob_path = config.BLOB_DOCUMENTS_WITH_EMBEDDINGS_PARQUET
    try:
        data = blob_storage.download_data(input_blob_path)
        df = pd.read_parquet(io.BytesIO(data))
        logging.info(f"Adatok betöltve: {len(df):,} dokumentum.")
    except Exception as e:
        logging.error(f"Hiba az adatok letöltésekor: {e}", exc_info=True)
        sys.exit(1)
    
    # 2. Stop szavak meghatározása
    stop_jogszabalyok = determine_stop_jogszabalyok(df)

    # 3. Gráf építése
    G = build_graph(df, stop_jogszabalyok)

    # 4. Gráf mentése és feltöltése
    output_blob_path = config.BLOB_GRAPH
    logging.info(f"Gráf feltöltése: {output_blob_path}")
    try:
        buffer = io.BytesIO()
        pickle.dump(G, buffer)
        buffer.seek(0)
        blob_storage.upload_data(buffer.getvalue(), output_blob_path)
        logging.info("Gráf sikeresen feltöltve.")
    except Exception as e:
        logging.error(f"Hiba a gráf építése során: {e}", exc_info=True)

    logging.info("\nGRÁFÉPÍTÉS BEFEJEZVE!")

if __name__ == '__main__':
    main()