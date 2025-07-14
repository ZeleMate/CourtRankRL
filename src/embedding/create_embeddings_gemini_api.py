# src/embedding/create_embeddings_gemini_api.py

# ==============================================================================
# === 1. IMPORTÁLÁSOK ÉS ALAPVETŐ BEÁLLÍTÁSOK ===
# ==============================================================================
import pandas as pd
import numpy as np
import gc
import time
import os
import sys
import io
import logging
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from tqdm.auto import tqdm
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile

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

# ==============================================================================
# === 2. KONFIGURÁCIÓ ÉS NAPLÓZÁS ===
# ==============================================================================

# --- Naplózás beállítása ---
logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)

# --- API újrapróbálkozási beállítások ---
API_RETRY_ATTEMPTS = 5
API_RETRY_DELAY_SECONDS = 5
# A feldolgozás során egyszerre ennyi sort dolgoz fel és ír ki a Parquet-be.
WRITE_BATCH_SIZE = 50000

# ==============================================================================
# === 3. FELDOLGOZÓ OSZTÁLYOK ÉS FÜGGVÉNYEK ===
# ==============================================================================

class DocumentSplitter:
    """Felelős egy dokumentum szövegének darabolásáért (chunking)."""
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    
    def split(self, doc_id: str, text: str) -> list[dict]:
        """Szöveg darabolása, chunk-azonosítóval ellátva."""
        if not isinstance(text, str) or not text.strip():
            return []
        
        chunks_text = self.text_splitter.split_text(text)
        return [
            {'doc_id': doc_id, 'chunk_id': f"{doc_id}-{i}", 'text_chunk': chunk}
            for i, chunk in enumerate(chunks_text)
        ]

def get_gemini_embeddings_with_retry(texts: list[str], model_name: str, task_type: str) -> list[list[float]]:
    """
    Gemini embeddingek lekérése újrapróbálkozási és exponenciális visszalépési logikával.
    Sikertelen próbálkozás esetén np.nan-t ad vissza az adott elemre.
    """
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            result = genai.embed_content(
                model=model_name,
                content=texts,
                task_type=task_type
            )
            return result['embedding']
        except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable) as e:
            logging.warning(f"API hiba (kísérlet: {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}. Várakozás...")
            time.sleep(API_RETRY_DELAY_SECONDS * (2 ** attempt))
        except Exception as e:
            logging.error(f"Váratlan API hiba: {e}", exc_info=True)
            break
    
    nan_embedding = [np.nan] * config.EMBEDDING_DIMENSION
    return [nan_embedding] * len(texts)

def download_input_data(blob_storage: AzureBlobStorage) -> pd.DataFrame:
    """Letölti a bemeneti Parquet fájlt az Azure Blob Storage-ból."""
    logging.info(f"Bemeneti adatok letöltése innen: {config.BLOB_CLEANED_DOCUMENTS_PARQUET}")
    try:
        data = blob_storage.download_data(config.BLOB_CLEANED_DOCUMENTS_PARQUET)
        df = pd.read_parquet(io.BytesIO(data))
        logging.info(f"Sikeresen letöltve {len(df):,} sor.")
        return df
    except Exception as e:
        logging.error(f"Hiba a bemeneti fájl letöltése vagy olvasása közben: {e}", exc_info=True)
        return pd.DataFrame()

def create_text_chunks(df: pd.DataFrame) -> list[dict]:
    """DataFrame-ből létrehozza a szövegdarabok (chunks) listáját."""
    if df.empty:
        return []
        
    splitter = DocumentSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    all_chunk_records = []
    logging.info("Szövegek darabolásának megkezdése...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Dokumentumok darabolása"):
        chunks = splitter.split(row['doc_id'], row['text'])
        if not chunks:
            continue
            
        meta_cols = {col: row.get(col) for col in df.columns if col != 'text'}
        for chunk_info in chunks:
            record = meta_cols.copy()
            record.update(chunk_info)
            all_chunk_records.append(record)
            
    logging.info(f"Összesen {len(all_chunk_records):,} darab (chunk) készült.")
    return all_chunk_records


def generate_and_upload_embeddings(blob_storage: AzureBlobStorage, records: list[dict]):
    """
    Legenerálja az embeddingeket a szövegdarabokhoz és kötegelten feltölti 
    az eredményt egy Parquet fájlba az Azure Blob Storage-ba.
    """
    if not records:
        logging.warning("Nincsenek feldolgozandó rekordok, az embedding generálás átugorva.")
        return

    output_blob_path = config.BLOB_DOCUMENTS_WITH_EMBEDDINGS_PARQUET
    all_processed_dfs = []
    
    try:
        logging.info(f"Embeddingek generálása és mentése ide: {output_blob_path}")

        record_generator = (records[i:i + WRITE_BATCH_SIZE] for i in range(0, len(records), WRITE_BATCH_SIZE))

        for batch_records in tqdm(record_generator, total=(len(records) - 1) // WRITE_BATCH_SIZE + 1, desc="Fő feldolgozási ciklus"):
            if not batch_records:
                continue

            batch_df = pd.DataFrame(batch_records)
            text_chunks_list = batch_df['text_chunk'].tolist()
            
            all_embeddings = []
            for j in tqdm(range(0, len(text_chunks_list), config.BATCH_SIZE), desc="Gemini API hívások", leave=False):
                api_batch_texts = text_chunks_list[j:j + config.BATCH_SIZE]
                api_batch_embeddings = get_gemini_embeddings_with_retry(
                    texts=api_batch_texts,
                    model_name=config.MODEL_NAME,
                    task_type="RETRIEVAL_DOCUMENT"
                )
                all_embeddings.extend(api_batch_embeddings)
            
            batch_df['embedding'] = all_embeddings
            
            failed_mask = batch_df['embedding'].apply(lambda x: not isinstance(x, list) or (isinstance(x, list) and np.isnan(x).any()))
            if failed_mask.any():
                logging.warning(f"Egy darabon belül {failed_mask.sum()} embedding generálása sikertelen volt, ezek kihagyásra kerülnek.")
                batch_df = batch_df[~failed_mask]

            batch_df = batch_df.drop(columns=['text', 'text_chunk'], errors='ignore')

            if not batch_df.empty:
                all_processed_dfs.append(batch_df)

        if not all_processed_dfs:
            logging.warning("Nem sikerült egyetlen érvényes embeddinget sem létrehozni.")
            return

        final_df = pd.concat(all_processed_dfs, ignore_index=True)
        
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            final_df.to_parquet(tmp.name, engine='pyarrow', compression='snappy')
            tmp_path = tmp.name
        
        logging.info(f"Ideiglenes fájl létrehozva: {tmp_path}. Feltöltés...")
        blob_storage.upload_file(local_path=tmp_path, blob_path=output_blob_path)
        
        os.remove(tmp_path)
        logging.info(f"Sikeresen feltöltve {len(final_df):,} embedding.")

    except Exception as e:
        logging.error(f"Hiba az embedding generálás vagy feltöltés során: {e}", exc_info=True)
    finally:
        gc.collect()

def main():
    """Fő vezérlő függvény: letölti, darabolja, generálja az embeddingeket és feltölti az eredményt."""
    start_time = time.time()
    logging.info("Embedding generálási folyamat elindítva...")

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        logging.error("A GOOGLE_API_KEY környezeti változó nincs beállítva. A folyamat leáll.")
        sys.exit(1)
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    try:
        blob_storage = AzureBlobStorage(container_name=config.AZURE_CONTAINER_NAME)
    except ValueError as e:
        logging.error(e)
        sys.exit(1)

    df_cleaned = download_input_data(blob_storage)
    if df_cleaned.empty:
        logging.error("A bemeneti adatok üresek vagy nem sikerült letölteni. A folyamat leáll.")
        return

    chunk_records = create_text_chunks(df_cleaned)
    del df_cleaned
    gc.collect()

    generate_and_upload_embeddings(blob_storage, chunk_records)

    total_time = time.time() - start_time
    logging.info(f"--- Embedding generálás befejezve {total_time:.2f} másodperc alatt ---")


if __name__ == '__main__':
    main() 