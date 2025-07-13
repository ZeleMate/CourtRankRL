# src/embedding/create_embeddings_gemini_api.py

# ==============================================================================
# === 1. CSOMAGOK TELEPÍTÉSE ===
# ==============================================================================
# Futtassa ezt a cellát a szükséges csomagok telepítéséhez a Jupyter/Colab környezetben.
import sys
import subprocess

# A linter-barát megoldás a csomagok telepítésére
packages = [
    "pandas", "numpy", "tqdm", "langchain", "pyarrow", 
    "python-dotenv", "google-generativeai"
]
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])
except subprocess.CalledProcessError as e:
    print(f"Hiba a csomagok telepítése közben: {e}")

# ==============================================================================
# === 2. IMPORTÁLÁSOK ÉS ALAPVETŐ BEÁLLÍTÁSOK ===
# ==============================================================================
import pandas as pd
import numpy as np
import gc
import time
import os
import sys
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from tqdm.auto import tqdm
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pyarrow as pa
import pyarrow.parquet as pq

# ==============================================================================
# === 3. KONFIGURÁCIÓS VÁLTOZÓK ===
# ==============================================================================

# --- Környezeti változók betöltése ---
# FONTOS: Győződjön meg róla, hogy a GOOGLE_API_KEY változó be van állítva.
load_dotenv()

# --- Helyi fájlbeállítások ---
PROCESSED_DATA_DIR = Path("processed")
INPUT_PARQUET_PATH = PROCESSED_DATA_DIR / "cleaned_documents.parquet"
OUTPUT_PARQUET_PATH = PROCESSED_DATA_DIR / "documents_with_gemini_embeddings.parquet"

# --- Gemini API beállítások ---
GEMINI_MODEL_NAME = "text-embedding-004"
GEMINI_TASK_TYPE = "RETRIEVAL_DOCUMENT"
GEMINI_BATCH_SIZE = 100
API_RETRY_ATTEMPTS = 5
API_RETRY_DELAY_SECONDS = 5

# --- Feldolgozási beállítások ---
WRITE_BATCH_SIZE = 50000 # Egyszerre ennyi sort dolgoz fel és ír ki

# --- Darabolás beállítások ---
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 200

# ==============================================================================
# === 4. FELDOLGOZÓ OSZTÁLYOK ===
# ==============================================================================

class DocumentProcessor:
    """Felelős egy dokumentum szövegének darabolásáért (chunking)."""
    def __init__(self, chunk_size, chunk_overlap):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    def split_document(self, doc_id, text):
        if not isinstance(text, str) or not text.strip():
            return []
        chunks_text = self.text_splitter.split_text(text)
        return [{'doc_id': doc_id, 'text_chunk': chunk} for chunk in chunks_text]

# ==============================================================================
# === 5. FŐ VEZÉRLŐ LOGIKA ===
# ==============================================================================

def get_gemini_embeddings_with_retry(texts, model_name, task_type):
    """Gemini embeddingek lekérése újrapróbálkozási logikával."""
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            result = genai.embed_content(
                model=model_name,
                content=texts,
                task_type=task_type
            )
            return result['embedding']
        except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable) as e:
            print(f"API hiba (kísérlet: {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}. Várakozás és újrapróbálás...")
            time.sleep(API_RETRY_DELAY_SECONDS * (2 ** attempt)) # Exponenciális backoff
        except Exception as e:
            print(f"Váratlan hiba az API hívás során: {e}")
            break
    return [np.nan] * len(texts)

def main():
    """Fő vezérlő függvény az embedding generáláshoz a Gemini API segítségével."""
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ HIBA: A GOOGLE_API_KEY környezeti változó nincs beállítva.")
        return
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    writer = None
    try:
        PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        print(f"Adatok olvasása innen: {INPUT_PARQUET_PATH}")
        if not INPUT_PARQUET_PATH.exists():
            print(f"❌ HIBA: A bemeneti fájl nem található: {INPUT_PARQUET_PATH}")
            return
            
        main_start_time = time.time()
        df_full = pd.read_parquet(INPUT_PARQUET_PATH)
        
        if df_full.empty:
            print("⚠️ A bemeneti DataFrame üres.")
            return

        processor = DocumentProcessor(CHUNK_SIZE, CHUNK_OVERLAP)
        
        print("Szövegek darabolása (chunking)...")
        all_chunk_records = []
        for _, row in tqdm(df_full.iterrows(), total=len(df_full), desc="Dokumentumok feldolgozása"):
            chunks = processor.split_document(row['doc_id'], row['text'])
            if chunks:
                meta_cols = {col: row.get(col) for col in df_full.columns if col != 'text'}
                for i, chunk_info in enumerate(chunks):
                    record = meta_cols.copy()
                    record['chunk_id'] = f"{chunk_info['doc_id']}-{i}"
                    record['text_chunk'] = chunk_info['text_chunk']
                    all_chunk_records.append(record)
        
        del df_full # Memória felszabadítása
        gc.collect()

        if not all_chunk_records:
            print("⚠️ A dokumentumokból nem sikerült darabokat készíteni.")
            return

        print(f"Összesen {len(all_chunk_records):,} darab (chunk) készült.")
        
        print(f"\nEmbeddingek generálása és mentés {WRITE_BATCH_SIZE} soros darabokban...")

        for i in tqdm(range(0, len(all_chunk_records), WRITE_BATCH_SIZE), desc="Fő feldolgozási ciklus"):
            batch_records = all_chunk_records[i:i + WRITE_BATCH_SIZE]
            
            if not batch_records:
                continue

            batch_df = pd.DataFrame(batch_records)

            text_chunks_list = batch_df['text_chunk'].tolist()
            all_embeddings = []
            for j in tqdm(range(0, len(text_chunks_list), GEMINI_BATCH_SIZE), desc="Gemini API hívások", leave=False):
                api_batch_texts = text_chunks_list[j:j + GEMINI_BATCH_SIZE]
                api_batch_embeddings = get_gemini_embeddings_with_retry(
                    texts=api_batch_texts,
                    model_name=GEMINI_MODEL_NAME,
                    task_type=GEMINI_TASK_TYPE
                )
                all_embeddings.extend(api_batch_embeddings)
            
            batch_df['embedding'] = all_embeddings
            
            failed_count = batch_df['embedding'].isna().sum()
            if failed_count > 0:
                print(f"⚠️ Darabon belül {failed_count} embedding generálása sikertelen volt.")
                batch_df.dropna(subset=['embedding'], inplace=True)

            if 'text' in batch_df.columns:
                batch_df = batch_df.drop(columns=['text'])

            # Adatok hozzáfűzése a Parquet fájlhoz
            table = pa.Table.from_pandas(batch_df)
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PARQUET_PATH, table.schema, compression='snappy')
            writer.write_table(table)
        
        print("\n✅ Mentés sikeres.")

        total_time = time.time() - main_start_time
        print(f"\n--- Feldolgozás befejezve {total_time:.2f} másodperc alatt ---")

    finally:
        if writer:
            writer.close()
        print("🧹 Memória felszabadítása...")
        gc.collect()

# ==============================================================================
# === 6. SCRIPT FUTTATÁSA ===
# ==============================================================================
if __name__ == '__main__':
    main() 