# src/data_loader/build_faiss_index.py
"""
FAISS index építő script a jogi dokumentumok embeddingjeihez.

Ez a script beolvassa a parquet fájlból az előre elkészített embeddinget és létrehoz
belőlük egy FAISS indexet a gyors hasonlósági kereséshez.
"""
import pandas as pd
import numpy as np
import faiss
import logging
import os
import sys
import gc
import time
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Konfiguráció importálása
from configs import config

# ------------------------------------------------------------------
# Konfiguráció betöltése
# ------------------------------------------------------------------
OUT_DIR = config.OUT_DIR
PROCESSED_PARQUET_DATA_PATH = config.PROCESSED_PARQUET_DATA_PATH
EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION

# FAISS specifikus beállítások hozzáadása a konfigurációhoz
FAISS_INDEX_PATH = OUT_DIR / "faiss_index.bin"
FAISS_MAPPING_PATH = OUT_DIR / "faiss_id_mapping.pkl"
FAISS_NLIST = 100    # Hány darab clusterre osszuk az adatokat
FAISS_NPROBE = 10    # Hány clustert vizsgálunk kereséskor

# Kimeneti könyvtár létrehozása, ha nem létezik
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging beállítása
logging.basicConfig(
    level=config.LOGGING_LEVEL,
    format=config.LOGGING_FORMAT
)

def create_faiss_index(vectors: np.ndarray) -> Any:
    """
    Létrehoz egy FAISS indexet a megadott vektorokból.
    
    Args:
        vectors: A beágyazási vektorok numpy tömbje
    
    Returns:
        A létrehozott FAISS index
    """
    vector_dimension = vectors.shape[1]
    vector_count = vectors.shape[0]
    logging.info(f"Vektorok száma: {vector_count}, dimenziószáma: {vector_dimension}")
    
    # Ellenőrizzük, hogy az adatok megfelelő formátumban vannak-e
    if not vectors.flags.c_contiguous:
        logging.warning("Az adatok nem C-contiguous formátumban vannak, átalakítás...")
        vectors = np.ascontiguousarray(vectors)
    
    # Index típus kiválasztása az adatok mérete alapján
    if vector_count < 10000:
        # Kis adathalmazhoz egyszerű indexet használunk
        logging.info("Kis adathalmaz, IndexFlatL2 használata")
        index = faiss.IndexFlatL2(vector_dimension)
    else:
        # Nagyobb adathalmazhoz IVF indexet használunk
        logging.info(f"Nagy adathalmaz, IVF index létrehozása {FAISS_NLIST} clusterrel")
        # IndexFlatL2 létrehozása kvantálóként
        quantizer = faiss.IndexFlatL2(vector_dimension)
        # IndexIVFFlat létrehozása a megfelelő paraméterekkel
        index = faiss.IndexIVFFlat(quantizer, vector_dimension, FAISS_NLIST, faiss.METRIC_L2)
        
        # IVF index betanítása
        logging.info("IVF index betanítása...")
        
        # Nagy adathalmaznál mintavételezés a tanításhoz
        if vector_count > 1_000_000:
            logging.info("Túl sok vektor, mintavételezés a betanításhoz...")
            sample_indices = np.random.choice(vector_count, 1_000_000, replace=False)
            train_vectors = vectors[sample_indices]
            index.train(train_vectors)  # type: ignore
            del sample_indices
            del train_vectors
            gc.collect()
        else:
            index.train(vectors)  # type: ignore
        
        # Keresési paraméter beállítása
        index.nprobe = FAISS_NPROBE
    
    # Adatok hozzáadása az indexhez
    start_time = time.time()
    index.add(vectors)  # type: ignore
    logging.info(f"Indexelés kész, {vector_count} vektor feldolgozva, időtartam: {time.time() - start_time:.2f} mp")
    
    return index

def test_search(index: Any, vectors: np.ndarray, id_mapping: Dict[int, Any], k: int = 5) -> None:
    """
    Az indexet teszteli egy egyszerű kereséssel.
    
    Args:
        index: A FAISS index
        vectors: A vektorok, amelyekből az index készült
        id_mapping: Az azonosító leképezés
        k: Hány legközelebbi szomszédot keressünk
    """
    logging.info("Index tesztelése egyszerű kereséssel...")
    # Első vektort használjuk lekérdezésként
    query_vector = vectors[0].reshape(1, -1)
    
    start_time = time.time()
    distances, indices = index.search(query_vector, k)
    search_time = time.time() - start_time
    
    logging.info(f"Keresési idő: {search_time*1000:.1f} ms")
    logging.info(f"Találatok FAISS ID-jei: {indices[0].tolist()}")
    
    # Eredeti dokumentum ID-k visszakeresése
    doc_ids = [id_mapping[idx] for idx in indices[0].tolist()]
    logging.info(f"Találatok dokumentum ID-jei: {doc_ids}")

def main():
    # Bemeneti fájl ellenőrzése
    if not PROCESSED_PARQUET_DATA_PATH.exists():
        logging.error(f"A bemeneti fájl nem található: {PROCESSED_PARQUET_DATA_PATH}")
        raise SystemExit("Először futtasd a generate_embeddings.py szkriptet!")

    try:
        # Beágyazások beolvasása
        logging.info(f"Beágyazások beolvasása: {PROCESSED_PARQUET_DATA_PATH}")
        df = pd.read_parquet(PROCESSED_PARQUET_DATA_PATH, columns=['doc_id', 'embedding'])
        logging.info(f"Beolvasva: {len(df)} dokumentum")
        
        # Hiányzó embeddinges sorok kezelése
        missing_count = df['embedding'].isna().sum()
        if missing_count:
            logging.warning(f"{missing_count} sorban nincs embedding, ezek kiszűrése...")
            df = df.dropna(subset=['embedding']).reset_index(drop=True)
            logging.info(f"Szűrés után maradt {len(df)} dokumentum")
        
        # ID-leképezés létrehozása
        id_mapping = dict(enumerate(df['doc_id']))
        
        # Vektorok konvertálása numpy tömbbé
        vectors = np.stack(df['embedding'].tolist()).astype(np.float32)
        
        # Memória felszabadítása
        del df
        gc.collect()
        
        # Dimenziószám ellenőrzése
        if vectors.shape[1] != EMBEDDING_DIMENSION:
            logging.warning(f"A vektorok dimenziója {vectors.shape[1]}, konfigurációban: {EMBEDDING_DIMENSION}")

        # FAISS index létrehozása
        logging.info("FAISS index létrehozása...")
        index = create_faiss_index(vectors)
        
        # Index és ID-leképezés mentése
        logging.info(f"Index mentése: {FAISS_INDEX_PATH}")
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        
        logging.info(f"ID-leképezés mentése: {FAISS_MAPPING_PATH}")
        with open(FAISS_MAPPING_PATH, 'wb') as f:
            pickle.dump(id_mapping, f)
        
        # Index tesztelése
        test_search(index, vectors, id_mapping)
        
        print(f"✅ FAISS index létrehozva: {FAISS_INDEX_PATH}")
        print(f"✅ ID-leképezés mentve: {FAISS_MAPPING_PATH}")
        
    except Exception as e:
        logging.error(f"Hiba történt: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise SystemExit("Hiba a FAISS index építése során!")

if __name__ == '__main__':
    main()