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
    from src.utils.azure_blob_storage import AzureBlobStorage
except ImportError as e:
    print(f"HIBA: Modul importálása sikertelen: {e}")
    sys.exit(1)

# Loggolás beállítása
logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

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

def main():
    """ Fő függvény a FAISS index létrehozásához Azure Blob Storage integrációval. """
    logging.info("🚀 FAISS INDEX ÉPÍTÉSE AZURE BLOB STORAGE ALAPJÁN")
    
    # Azure Blob Storage kliens inicializálása
    try:
        blob_storage = AzureBlobStorage(container_name=config.AZURE_CONTAINER_NAME)
    except ValueError as e:
        logging.error(e)
        sys.exit(1)

    # ===== 1. ADATOK LETÖLTÉSE AZURE-BÓL =====
    input_blob_path = config.BLOB_DOCUMENTS_WITH_EMBEDDINGS_PARQUET
    logging.info(f"Embeddingek letöltése: {input_blob_path}")
    try:
        data = blob_storage.download_data(input_blob_path)
        df = pd.read_parquet(io.BytesIO(data))
        logging.info(f"✅ Sikeresen letöltve és beolvasva {len(df):,} dokumentum.")
    except Exception as e:
        logging.error(f"Hiba az embeddingek letöltése vagy feldolgozása közben: {e}", exc_info=True)
        sys.exit(1)

    try:
        # ===== 2. ADATOK VALIDÁLÁSA ÉS TISZTÍTÁSA =====
        logging.info(f"Adatok validálása...")
        
        if 'embedding' not in df.columns or 'doc_id' not in df.columns:
            logging.error("A DataFrame nem tartalmazza a szükséges 'embedding' vagy 'doc_id' oszlopot.")
            sys.exit(1)

        missing_count = df['embedding'].isna().sum()
        if missing_count:
            logging.warning(f"{missing_count} sorban nincs embedding, ezek kiszűrése...")
            df = df.dropna(subset=['embedding']).reset_index(drop=True)
        
        if df.empty:
            logging.error("Nincsenek érvényes embeddingek az index építéséhez.")
            sys.exit(1)
        
        logging.info(f"Validálás után maradt {len(df):,} dokumentum.")

        # ===== 3. VEKTOROK ÉS ID-LEKÉPEZÉS LÉTREHOZÁSA =====
        logging.info("Vektorok és ID-leképezés előkészítése...")
        
        # A vektorokat egyetlen NumPy tömbbe fűzzük
        vectors = np.vstack(df['embedding'].values).astype('float32')
        
        # Ellenőrizzük a dimenziót
        if vectors.shape[1] != config.EMBEDDING_DIMENSION:
            logging.error(
                f"Embedding dimenzió eltérés: "
                f"Várt: {config.EMBEDDING_DIMENSION}, Kapott: {vectors.shape[1]}"
            )
            sys.exit(1)
        
        # ID-leképezés létrehozása: FAISS index -> eredeti doc_id
        # A FAISS egyszerű, 0-tól n-1-ig terjedő indexeket használ.
        id_mapping = {i: doc_id for i, doc_id in enumerate(df['doc_id'])}
        vector_count = len(df)
        
        del df; gc.collect()

        # ===== 4. FAISS INDEX LÉTREHOZÁSA =====
        index = create_faiss_index(vectors, vector_count)

        # ===== 5. TESZTELÉS =====
        test_search(index, vectors, id_mapping)
        del vectors; gc.collect()

        # ===== 6. INDEX ÉS LEKÉPEZÉS FELTÖLTÉSE AZURE-BA =====
        # FAISS index mentése memóriába és feltöltése
        logging.info(f"FAISS index feltöltése ide: {config.BLOB_FAISS_INDEX}")
        index_buffer = io.BytesIO()
        faiss.write_index(index, faiss.PyCallbackIOWriter(index_buffer.write))
        blob_storage.upload_data(index_buffer.getvalue(), config.BLOB_FAISS_INDEX)
        logging.info("✅ FAISS index sikeresen feltöltve.")

        # ID-leképezés mentése JSON-ként és feltöltése
        logging.info(f"ID-leképezés feltöltése ide: {config.BLOB_FAISS_DOC_ID_MAP}")
        map_buffer = io.BytesIO(json.dumps(id_mapping, ensure_ascii=False).encode('utf-8'))
        blob_storage.upload_data(map_buffer.getvalue(), config.BLOB_FAISS_DOC_ID_MAP)
        logging.info("✅ ID-leképezés sikeresen feltöltve.")

    except Exception as e:
        logging.error(f"Váratlan hiba történt az index építése során: {e}", exc_info=True)
        sys.exit(1)

    logging.info("\n🎉 FAISS INDEX ÉPÍTÉS BEFEJEZVE!")

if __name__ == '__main__':
    main()