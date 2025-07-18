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
except ImportError as e:
    print(f"HIBA: Modul importálása sikertelen: {e}")
    sys.exit(1)

# Az Azure SDK naplózási szintjének beállítása, hogy ne legyen túl beszédes
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

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
    """
    Felépíti a NetworkX gráfot a bemeneti DataFrame alapján.
    A csomópontokat és éleket iteratívan adja hozzá a memóriakezelés javítása érdekében.
    """
    G = nx.DiGraph()
    logging.info("Gráfépítés megkezdése...")

    for _, doc_data in tqdm(df.iterrows(), total=df.shape[0], desc="Gráf építése"):
        doc_id = doc_data.get('doc_id')
        if not is_valid_doc_id(doc_id):
            logging.debug(f"Érvénytelen vagy hiányzó doc_id ({doc_id}), a sor kihagyva.")
            continue

        # 1. Dokumentum csomópont hozzáadása vagy frissítése
        node_attrs = {
            "type": "dokumentum",
            "jogterulet": doc_data.get('jogterulet') if pd.notna(doc_data.get('jogterulet')) else None,
            "birosag": doc_data.get('birosag') if pd.notna(doc_data.get('birosag')) else None,
            "ev": int(doc_data.get('HatarozatEve')) if pd.notna(doc_data.get('HatarozatEve')) and str(doc_data.get('HatarozatEve')).isdigit() else None,
        }
        clean_attrs = {k: v for k, v in node_attrs.items() if v is not None}
        
        if G.has_node(doc_id):
            G.nodes[doc_id].update(clean_attrs)
        else:
            G.add_node(doc_id, **clean_attrs)

        # 2. Kapcsolatok (élek) feldolgozása
        # Adatmezők kinyerése és listává alakítása
        jogszabalyhelyek = parse_list_string(doc_data.get('Jogszabalyhelyek', ''))
        kapcsolodo_hatarozatok = parse_list_string(doc_data.get('AllKapcsolodoUgyszam', ''))
        kapcsolodo_birosagok = parse_list_string(doc_data.get('AllKapcsolodoBirosag', ''))

        # Hivatkozott határozatok élei
        for hatarozat_id in kapcsolodo_hatarozatok:
            if is_valid_doc_id(hatarozat_id):
                if not G.has_node(hatarozat_id):
                    G.add_node(hatarozat_id, type="dokumentum") # Alapértelmezett attribútum
                
                if G.has_edge(doc_id, hatarozat_id):
                    G[doc_id][hatarozat_id]['weight'] += 1
                else:
                    G.add_edge(doc_id, hatarozat_id, relation_type="hivatkozik", weight=1)

        # Bírósági kapcsolatok élei
        for birosag_name in kapcsolodo_birosagok:
            if birosag_name and isinstance(birosag_name, str):
                birosag_node_id = f"birosag_{birosag_name.lower().replace(' ', '_')}"
                if not G.has_node(birosag_node_id):
                    G.add_node(birosag_node_id, type="birosag", name=birosag_name)
                
                if G.has_edge(doc_id, birosag_node_id):
                    G[doc_id][birosag_node_id]['weight'] += 1
                else:
                    G.add_edge(doc_id, birosag_node_id, relation_type="targyalta", weight=1)

        # Jogszabályhelyek élei
        for jsz in jogszabalyhelyek:
            if jsz and isinstance(jsz, str) and jsz not in stop_jogszabalyok:
                jsz_node_id = f"jogszabaly_{jsz.lower().replace(' ', '_').replace('.', '').replace('§', 'par').replace('(', '').replace(')', '')}"
                if not G.has_node(jsz_node_id):
                    G.add_node(jsz_node_id, type="jogszabaly", reference=jsz)

                if G.has_edge(doc_id, jsz_node_id):
                    G[doc_id][jsz_node_id]['weight'] += 1
                else:
                    G.add_edge(doc_id, jsz_node_id, relation_type="hivatkozik_jogszabalyra", weight=1)

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
    kiszámítja a hasonlóságokat, és elmenti a gráfot lokálisan.
    """
    logging.info("GRÁFÉPÍTŐ INDÍTÁSA LOKÁLIS ADATOK ALAPJÁN")

    input_path = config.DOCUMENTS_WITH_EMBEDDINGS_PARQUET
    try:
        df = pd.read_parquet(input_path)
        logging.info(f"Adatok betöltve: {len(df):,} dokumentum innen: {input_path}")
    except Exception as e:
        logging.error(f"Hiba az adatok betöltésekor: {e}", exc_info=True)
        sys.exit(1)

    # 2. Stop szavak meghatározása
    stop_jogszabalyok = determine_stop_jogszabalyok(df)

    # 3. Gráf építése
    G = build_graph(df, stop_jogszabalyok)

    # 4. Gráf mentése
    output_path = config.GRAPH_PATH
    logging.info(f"Gráf mentése ide: {output_path}")
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
        logging.info("Gráf sikeresen elmentve.")
    except Exception as e:
        logging.error(f"Hiba a gráf mentése során: {e}", exc_info=True)

    logging.info("\nGRÁFÉPÍTÉS BEFEJEZVE!")

if __name__ == '__main__':
    main()