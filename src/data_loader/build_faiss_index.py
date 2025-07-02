# src/data_loader/build_faiss_index.py
"""
FAISS index építő szkript a jogi dokumentumok embeddingjeihez.

ÚJDONSÁG: Chunked parquet támogatás memory-safe FAISS index építéshez.
Ez a szkript először chunked parquet fájlokat keres, majd fallback az egyesített parquet-re.
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
import glob
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
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

# FAISS specifikus elérési utak és paraméterek a konfigurációból
FAISS_INDEX_PATH = OUT_DIR / "faiss_index.bin"
FAISS_MAPPING_PATH = OUT_DIR / "faiss_id_mapping.pkl"
FAISS_NLIST = config.FAISS_INDEX_NLIST
FAISS_NPROBE = config.FAISS_INDEX_NPROBE

# Kimeneti könyvtár létrehozása, ha nem létezik
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Loggolás beállítása
logging.basicConfig(
    level=config.LOGGING_LEVEL,
    format=config.LOGGING_FORMAT
)

def create_faiss_index(vectors: np.ndarray) -> Any:
    """
    Létrehoz egy FAISS indexet a megadott vektorokból.
    A vektorok száma alapján választ IndexFlatL2 vagy IndexIVFFlat típust.

    Args:
        vectors: A beágyazási vektorok NumPy tömbje (float32).
    
    Returns:
        A létrehozott FAISS index (faiss.Index objektum).
    """
    vector_dimension = vectors.shape[1]
    vector_count = vectors.shape[0]
    logging.info(f"FAISS index létrehozása {vector_count} vektorral, dimenzió: {vector_dimension}")
    
    if not vectors.flags.c_contiguous:
        logging.warning("A bemeneti vektorok nem C-folyamatosak (C-contiguous), átalakítás...")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    
    if vector_count < 10000:
        logging.info(f"Kis adathalmaz ({vector_count} vektor), IndexFlatL2 használata.")
        index = faiss.IndexFlatL2(vector_dimension)
    else:
        logging.info(f"Nagyobb adathalmaz ({vector_count} vektor), IndexIVFFlat létrehozása {FAISS_NLIST} klaszterrel.")
        quantizer = faiss.IndexFlatL2(vector_dimension)
        index = faiss.IndexIVFFlat(quantizer, vector_dimension, FAISS_NLIST, faiss.METRIC_L2)
        
        logging.info("IndexIVFFlat betanítása...")
        if not index.is_trained and vector_count > 0:
            index.train(vectors)  # type: ignore
        else:
            logging.info("Az index már betanított vagy nincs adat a tanításhoz.")
            
        index.nprobe = FAISS_NPROBE
    
    logging.info("Vektorok hozzáadása az indexhez...")
    start_time = time.time()
    if vector_count > 0:
        index.add(vectors)  # type: ignore
    else:
        logging.warning("Nincsenek vektorok az indexhez adáshoz.")
    logging.info(f"Vektorok indexelése befejezve. Feldolgozott vektorok: {index.ntotal if vector_count > 0 else 0}, idő: {time.time() - start_time:.2f} mp")
    
    return index

def load_chunked_embeddings() -> Tuple[bool, pd.DataFrame]:
    """
    Chunked parquet fájlokból embeddings betöltése memory-safe módon.
    
    Returns:
        Tuple[bool, pd.DataFrame]: (success, combined_dataframe)
    """
    # Chunked parquet fájlok keresése
    chunked_pattern = str(OUT_DIR / "*_with_embeddings.parquet")
    chunk_files = glob.glob(chunked_pattern)
    
    if not chunk_files:
        logging.info("Nincs chunked parquet fájl találva")
        return False, pd.DataFrame()
    
    logging.info(f"🎯 CHUNKED PARQUET BETÖLTÉS: {len(chunk_files)} chunk fájl található")
    
    # Chunk fájlok rendezett betöltése (konzisztens sorrend)
    chunk_files.sort()
    all_chunks = []
    total_docs = 0
    
    for i, chunk_file in enumerate(chunk_files):
        try:
            logging.info(f"Chunk betöltése ({i+1}/{len(chunk_files)}): {os.path.basename(chunk_file)}")
            
            # Csak szükséges oszlopok betöltése (memória optimalizálás)
            chunk_df = pd.read_parquet(chunk_file, columns=['doc_id', 'embedding'])
            
            # Alapvető validáció
            if chunk_df.empty:
                logging.warning(f"Üres chunk: {chunk_file}")
                continue
            
            # Hiányzó embeddings eltávolítása
            missing_before = chunk_df['embedding'].isna().sum()
            if missing_before > 0:
                logging.warning(f"Chunk {i+1}: {missing_before} hiányzó embedding eltávolítva")
                chunk_df = chunk_df.dropna(subset=['embedding'])
            
            if not chunk_df.empty:
                all_chunks.append(chunk_df)
                total_docs += len(chunk_df)
                logging.info(f"Chunk {i+1} betöltve: {len(chunk_df):,} érvényes rekord")
            
            # Rendszeres memória tisztítás
            if i % 5 == 0:
                gc.collect()
                
        except Exception as e:
            logging.error(f"Hiba chunk betöltésében ({chunk_file}): {e}")
            continue
    
    if not all_chunks:
        logging.error("Nincs érvényes chunk adat")
        return False, pd.DataFrame()
    
    # Chunk-ok egyesítése
    logging.info("Chunk-ok egyesítése...")
    combined_df = pd.concat(all_chunks, ignore_index=True)
    
    # Memória felszabadítás
    del all_chunks
    gc.collect()
    
    logging.info(f"✅ Chunked betöltés sikeres:")
    logging.info(f"  📁 Chunk fájlok: {len(chunk_files)}")
    logging.info(f"  📄 Összesen dokumentumok: {len(combined_df):,}")
    logging.info(f"  🚀 Memory-optimalizált feldolgozás")
    
    return True, combined_df

def load_unified_embeddings() -> pd.DataFrame:
    """
    Unified parquet fájl betöltése (fallback mode).
    """
    if not PROCESSED_PARQUET_DATA_PATH.exists():
        raise FileNotFoundError(f"Unified parquet nem található: {PROCESSED_PARQUET_DATA_PATH}")
    
    logging.info("📄 UNIFIED PARQUET BETÖLTÉS (fallback mode)")
    logging.info(f"Embeddings betöltése: {PROCESSED_PARQUET_DATA_PATH}")
    
    df = pd.read_parquet(PROCESSED_PARQUET_DATA_PATH, columns=['doc_id', 'embedding'])
    
    logging.info(f"Unified parquet betöltve: {len(df):,} dokumentum")
    return df

def test_search(index: Any, vectors: np.ndarray, id_mapping: Dict[int, Any], k: int = 5) -> None:
    """
    Leteszteli a FAISS indexet egy egyszerű kereséssel az első vektor alapján.
    
    Args:
        index: A FAISS index.
        vectors: A vektorok, amelyekből az index készült (csak az elsőt használja a teszthez).
        id_mapping: Az FAISS index ID-k és az eredeti dokumentum ID-k közötti leképezés.
        k: A keresendő legközelebbi szomszédok száma.
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
            doc_ids = [id_mapping[idx] for idx in indices[0].tolist() if idx in id_mapping]
            logging.info(f"Találatok eredeti dokumentum ID-jai: {doc_ids}")
        except KeyError as e:
            logging.error(f"Hiba a dokumentum ID-k visszakeresésekor a leképezésből: hiányzó FAISS index {e}")
    else:
        logging.info("A tesztkeresés nem adott vissza találatot.")

def main():
    """
    Fő függvény a FAISS index létrehozásához.
    
    ÚJDONSÁG: Chunked parquet támogatás memory-safe FAISS index építéshez.
    Először chunked parquet fájlokat keres, fallback az egyesített parquet-re.
    """
    logging.info("🚀 CHUNKED-KOMPATIBILIS FAISS INDEX ÉPÍTÉS")
    
    # ===== 1. CHUNKED PARQUET BETÖLTÉS (PRIORITÁS) =====
    success, df = load_chunked_embeddings()
    
    # ===== 2. UNIFIED PARQUET FALLBACK =====
    if not success:
        logging.info("Chunked parquet nem elérhető, fallback unified parquet-re...")
        
        if not PROCESSED_PARQUET_DATA_PATH.exists():
            logging.error(f"Nincs elérhető embedding adat!")
            logging.error(f"Sem chunked parquet ({OUT_DIR}/*_with_embeddings.parquet)")
            logging.error(f"Sem unified parquet ({PROCESSED_PARQUET_DATA_PATH})")
            raise SystemExit("Először futtasd az embedding generálást!")
        
        df = load_unified_embeddings()

    try:
        # ===== 3. ADATOK VALIDÁLÁSA ÉS TISZTÍTÁSA =====
        logging.info(f"Embedding adatok validálása: {len(df):,} dokumentum")
        
        # Hiányzó embeddinges sorok kezelése (ha még vannak)
        missing_count = df['embedding'].isna().sum()
        if missing_count:
            logging.warning(f"{missing_count} sorban nincs embedding, ezek kiszűrése...")
            df = df.dropna(subset=['embedding']).reset_index(drop=True)
            logging.info(f"Szűrés után maradt {len(df):,} dokumentum")
        
        # Dimenzió ellenőrzés és szűrés a stackelés előtt
        if not df.empty:
            original_count = len(df)
            # Biztonságos hossz ellenőrzés: csak listákra/tömbökre alkalmazzuk
            df['embedding_len'] = df['embedding'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
            
            mismatched_dims = df[df['embedding_len'] != EMBEDDING_DIMENSION]
            if not mismatched_dims.empty:
                logging.warning(f"{len(mismatched_dims)} sor eltávolítva a nem megfelelő embedding dimenzió miatt.")
                logging.debug(f"Példák a hibás dimenziójú sorokból (doc_id, embedding hossza): {mismatched_dims[['doc_id', 'embedding_len']].head().to_dict('records')}")
                df = df[df['embedding_len'] == EMBEDDING_DIMENSION]

            df = df.drop(columns=['embedding_len'])

        if df.empty:
            logging.error("Nincsenek érvényes embeddingek a szűrés után. Leállás.")
            raise SystemExit("Nincs feldolgozható adat a dimenzióellenőrzés után.")

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
        
        # ===== VÉGSŐ ÖSSZEFOGLALÓ =====
        input_mode = "CHUNKED" if success else "UNIFIED"
        
        print(f"\n✅ CHUNKED-KOMPATIBILIS FAISS INDEX LÉTREHOZVA!")
        print(f"📊 Feldolgozott dokumentumok: {len(id_mapping):,}")
        print(f"📁 Input mód: {input_mode}")
        print(f"🗂️  FAISS index: {FAISS_INDEX_PATH}")
        print(f"🔗 ID mapping: {FAISS_MAPPING_PATH}")
        print(f"📏 Embedding dimenzió: {vectors.shape[1]}")
        print(f"🔍 Szemantikai kereshetőség: MEGŐRIZVE")
        if success:
            print(f"🚀 Memory-optimalizált chunked feldolgozás használva!")
        
        logging.info(f"FAISS index létrehozva ({input_mode} mode): {len(id_mapping):,} dokumentum")
        print(f"✅ ID-leképezés mentve: {FAISS_MAPPING_PATH}")
        
    except Exception as e:
        logging.error(f"Hiba történt: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise SystemExit("Hiba a FAISS index építése során!")

if __name__ == '__main__':
    main()