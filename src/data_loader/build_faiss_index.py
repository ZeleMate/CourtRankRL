# src/data_loader/build_faiss_index.py
"""
FAISS index építő szkript a dokumentum embeddingek alapján.
Beolvassa az embeddingeket tartalmazó Parquet fájlt az Azure Blob Storage-ból,
létrehozza a FAISS indexet és a hozzá tartozó ID-leképezést, majd feltölti
azokat is a Blob Storage-ba.
"""
import pandas as pd
import numpy as np
import faiss
import logging
import sys
import gc
import time
import json
import io
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Konfiguráció és segédprogramok importálása
try:
    from configs import config
except ImportError as e:
    print(f"HIBA: Modul importálása sikertelen: {e}")
    sys.exit(1)

# Loggolás beállítása
logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

# Az Azure SDK naplózási szintjének beállítása, hogy ne legyen túl beszédes
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

def create_faiss_index(vectors: np.ndarray, vector_count: int) -> Any:
    """
    Létrehoz egy FAISS indexet a megadott vektorokból.
    A vektorok száma alapján választ IndexFlatL2 vagy IndexIVFFlat típust.
    """
    vector_dimension = vectors.shape[1]
    logging.info(f"FAISS index létrehozása {vector_count} vektorral, dimenzió: {vector_dimension}")
    
    if not vectors.flags.c_contiguous:
        logging.warning("A bemeneti vektorok nem C-folyamatosak, átalakítás...")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    
    # Kisebb adathalmazokhoz a IndexFlatL2 gyorsabb és egyszerűbb.
    # A training lépést is kihagyhatjuk.
    # Nagyobb adathalmazoknál az IndexIVFFlat gyorsabb keresést tesz lehetővé a particionálás miatt.
    nlist = 100 # A particionálás (klaszterek) száma
    if vector_count < 10000:
        logging.info(f"Kis adathalmaz ({vector_count} vektor), IndexFlatL2 használata.")
        index = faiss.IndexFlatL2(vector_dimension)
    else:
        logging.info(f"Nagyobb adathalmaz ({vector_count} vektor), IndexIVFFlat létrehozása {nlist} klaszterrel.")
        quantizer = faiss.IndexFlatL2(vector_dimension)
        index = faiss.IndexIVFFlat(quantizer, vector_dimension, nlist, faiss.METRIC_L2)
        
        logging.info("IndexIVFFlat betanítása...")
        if not index.is_trained and vector_count > 0:
            index.train(vectors)
        else:
            logging.info("Az index már betanított vagy nincs adat a tanításhoz.")
            
        index.nprobe = 10 # Hány klasztert vizsgáljon kereséskor
    
    logging.info("Vektorok hozzáadása az indexhez...")
    start_time = time.time()
    if vector_count > 0:
        index.add(vectors)
    else:
        logging.warning("Nincsenek vektorok az indexhez adáshoz.")
    logging.info(f"Vektorok indexelése befejezve. Feldolgozott vektorok: {index.ntotal}, idő: {time.time() - start_time:.2f} mp")
    
    return index

def test_search(index: Any, vectors: np.ndarray, id_mapping: Dict[int, Any], k: int = 5) -> None:
    """
    Leteszteli a FAISS indexet egy egyszerű kereséssel az első vektor alapján.
    """
    if vectors.shape[0] == 0:
        logging.warning("Nincsenek vektorok a keresés teszteléséhez.")
        return

    logging.info(f"FAISS index tesztelése: {k} legközelebbi szomszéd keresése az első vektorhoz...")
    query_vector = np.ascontiguousarray(vectors[0:1], dtype=np.float32)
    
    start_time = time.time()
    distances, indices = index.search(query_vector, k)
    search_time = time.time() - start_time
    
    logging.info(f"Keresési idő: {search_time*1000:.2f} ms")
    if indices.size > 0:
        logging.info(f"Találatok FAISS indexei: {indices[0].tolist()}")
        try:
            # Fontos: Az id_mapping kulcsai int-ek, a keresés eredménye (indices) int64 lehet.
            doc_ids = [id_mapping[int(idx)] for idx in indices[0].tolist() if int(idx) in id_mapping]
            logging.info(f"Találatok eredeti dokumentum ID-jai: {doc_ids}")
        except KeyError as e:
            logging.error(f"Hiba a dokumentum ID-k visszakeresésekor a leképezésből: hiányzó FAISS index {e}")
    else:
        logging.info("A tesztkeresés nem adott vissza találatot.")

def load_data_from_local(filepath: Path) -> pd.DataFrame:
    """Adatok betöltése lokális Parquet fájlból."""
    if not filepath.exists():
        logging.error(f"A bemeneti fájl nem található: {filepath}")
        sys.exit(1)
    logging.info(f"Adatok betöltése innen: {filepath}")
    return pd.read_parquet(filepath)

def build_index(embeddings: np.ndarray) -> faiss.Index:
    """FAISS index építése a megadott embeddingekből."""
    vector_count = embeddings.shape[0]
    logging.info(f"FAISS index építése {vector_count} vektorra...")
    return create_faiss_index(embeddings, vector_count)

def save_artifacts(index: faiss.Index, doc_id_map: Dict[int, Any], index_path: Path, map_path: Path):
    """FAISS index és ID leképezés mentése lokális fájlokba."""
    logging.info(f"FAISS index mentése ide: {index_path}")
    faiss.write_index(index, str(index_path))
    
    logging.info(f"Dokumentum ID leképezés mentése ide: {map_path}")
    with open(map_path, 'w') as f:
        json.dump(doc_id_map, f)

def main():
    """Fő függvény a FAISS index építéséhez lokális adatokból."""
    logging.info("===== FAISS INDEX ÉPÍTÉSE LOKÁLISAN =====")
    start_time = time.time()

    # 1. Adatok betöltése lokálisan
    df = load_data_from_local(config.DOCUMENTS_WITH_EMBEDDINGS_PARQUET)
    
    if 'embedding' not in df.columns or df['embedding'].isnull().any():
        logging.error("A 'embedding' oszlop hiányzik vagy hibás adatokat tartalmaz.")
        sys.exit(1)

    # 2. Embeddingek és ID-k kinyerése
    logging.info("Embeddingek és ID-k előkészítése...")
    embeddings = np.vstack(df['embedding'].to_numpy()).astype('float32')
    doc_ids = df['doc_id'].tolist()
    
    # Memóriatakarékosság
    del df
    gc.collect()

    # 3. FAISS index építése
    logging.info(f"FAISS index építése {embeddings.shape[0]} vektorra...")
    index = build_index(embeddings)

    # 4. ID leképezés létrehozása
    doc_id_map = {i: doc_ids[i] for i in range(len(doc_ids))}

    # 5. Artefaktumok mentése lokálisan
    save_artifacts(index, doc_id_map, config.FAISS_INDEX_PATH, config.FAISS_DOC_ID_MAP_PATH)

    end_time = time.time()
    logging.info(f"FAISS index sikeresen létrehozva és elmentve. Időtartam: {end_time - start_time:.2f} másodperc.")
    logging.info("========================================")

if __name__ == '__main__':
    main()