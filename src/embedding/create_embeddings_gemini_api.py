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
except ImportError as e:
    print(f"HIBA: Modul importálása sikertelen: {e}")
    sys.exit(1)

# ==============================================================================
# === 2. KONFIGURÁCIÓ ÉS NAPLÓZÁS ===
# ==============================================================================

# --- Naplózás beállítása ---
logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)
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

def main():
    """Fő függvény, amely betölti az adatokat, legenerálja az embeddingeket és elmenti az eredményt."""
    logging.info("===== EMBEDDING GENERÁLÁS INDÍTÁSA (GEMINI API) =====")
    
    # Bemeneti adatok betöltése
    input_path = config.CLEANED_DOCUMENTS_PARQUET
    if not input_path.exists():
        logging.error(f"A bemeneti fájl nem található: {input_path}")
        sys.exit(1)
        
    logging.info(f"Adatok betöltése innen: {input_path}")
    df = pd.read_parquet(input_path)
    df = df.head(100) # TESZTELÉSHEZ
    
    # API kliens és modell inicializálása
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(config.MODEL_NAME)
    except Exception as e:
        logging.error(f"Hiba a Gemini API kliens inicializálásakor: {e}")
        sys.exit(1)
        
    # Feldolgozás és mentés
    output_path = config.DOCUMENTS_WITH_EMBEDDINGS_PARQUET
    logging.info(f"Embeddingek mentése ide: {output_path}")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        temp_output_path = tmp_file.name

    writer = None
    total_chunks = 0
    try:
        # A DF-ből chunkok készítése
        chunk_records = create_text_chunks(df)
        
        # Embedding generálás és írás kötegenként
        for i in tqdm(range(0, len(chunk_records), config.BATCH_SIZE), desc="Embedding kötegek feldolgozása"):
            batch_records = chunk_records[i:i+config.BATCH_SIZE]
            batch_df = pd.DataFrame(batch_records)

            texts_to_embed = batch_df['text_chunk'].tolist()
            embeddings = get_gemini_embeddings_with_retry(texts_to_embed, config.MODEL_NAME, "RETRIEVAL_DOCUMENT")

            batch_df['embedding'] = embeddings
            
            # Hibás sorok eltávolítása
            batch_df.dropna(subset=['embedding'], inplace=True)

            # Eredmény hozzáfűzése a Parquet fájlhoz
            table = pa.Table.from_pandas(batch_df)
            if writer is None:
                writer = pq.ParquetWriter(temp_output_path, table.schema)
            writer.write_table(table)
            total_chunks += 1
    finally:
        if writer:
            writer.close()

    if total_chunks > 0:
        os.replace(temp_output_path, output_path)
        logging.info(f"Sikeresen kiírva {total_chunks} köteg a(z) '{output_path}' fájlba.")
    else:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        logging.warning("Nem történt adatfeldolgozás, a kimeneti fájl nem jött létre.")
    
    logging.info("===== EMBEDDING GENERÁLÁS BEFEJEZVE =====")


if __name__ == '__main__':
    main() 