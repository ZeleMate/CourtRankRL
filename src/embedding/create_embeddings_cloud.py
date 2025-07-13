# src/embedding/create_embeddings_cloud.py

# ==============================================================================
# === 1. CSOMAGOK TELEPÍTÉSE ===
# ==============================================================================
# Futtassa ezt a cellát a szükséges csomagok telepítéséhez a Jupyter/Colab környezetben.
import sys
import subprocess

# A linter-barát megoldás a csomagok telepítésére
packages = [
    "pandas", "numpy", "torch", "tqdm", "sentence-transformers",
    "langchain", "pyarrow", "azure-storage-blob", "python-dotenv"
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
import torch
import time
import os
import sys
import tempfile
import shutil
import io
from tqdm.auto import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# ==============================================================================
# === 3. KONFIGURÁCIÓS VÁLTOZÓK ===
# ==============================================================================
# Helyi importok helyett itt definiáljuk a szükséges beállításokat.

# --- Azure Blob Storage beállítások ---
# FONTOS: Futtatás előtt győződjön meg róla, hogy az `AZURE_CONNECTION_STRING`
# környezeti változó be van állítva a notebook környezetében (pl. secrets)!
load_dotenv()
AZURE_CONTAINER_NAME = "courtrankrl"

# --- Blob Storage elérési utak ---
BLOB_PROCESSED_DATA_DIR = "processed"
BLOB_EMBEDDING_DIR = "embeddings"
BLOB_CLEANED_DOCUMENTS_PARQUET = f"{BLOB_PROCESSED_DATA_DIR}/cleaned_documents.parquet"
BLOB_DOCUMENTS_WITH_EMBEDDINGS_PARQUET = f"{BLOB_EMBEDDING_DIR}/documents_with_embeddings.parquet"

# --- Modell és darabolás beállítások ---
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 512
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 200

# --- Ideiglenes fájlok ---
TEMP_DATA_DIR = Path("temp_data")

# ==============================================================================
# === 4. AZURE BLOB STORAGE SEGÉDOSZTÁLY ===
# ==============================================================================
# A korábbi src.utils.azure_blob_storage.py tartalma beágyazva.

class AzureBlobStorage:
    def __init__(self, container_name: str):
        self.connection_string = os.getenv("AZURE_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError("AZURE_CONNECTION_STRING környezeti változó nincs beállítva.")
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        if not self.container_client.exists():
            self.container_client.create_container()

    def upload_data(self, data: bytes, blob_path: str):
        """Uploads in-memory data (bytes) to Azure Blob Storage."""
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_path)
        blob_client.upload_blob(data, overwrite=True)
        print(f"Uploaded data to {self.container_name}/{blob_path}")

    def download_data(self, blob_path: str) -> bytes:
        """Downloads a file from Azure Blob Storage as bytes."""
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=blob_path)
        return blob_client.download_blob().readall()

# ==============================================================================
# === 5. FELDOLGOZÓ OSZTÁLYOK ===
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
# === 6. FŐ VEZÉRLŐ LOGIKA ===
# ==============================================================================

def main():
    """Fő vezérlő függvény az embedding generáláshoz, több GPU támogatásával."""
    if not torch.cuda.is_available():
        print("❌ HIBA: CUDA nem elérhető. A szkript csak GPU-s környezetben futtatható.")
        return
        
    num_gpus = torch.cuda.device_count()
    print(f"🔥 {num_gpus} db GPU elérhető.")
    for i in range(num_gpus):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

    try:
        blob_storage = AzureBlobStorage(container_name=AZURE_CONTAINER_NAME)
    except ValueError as e:
        print(f"❌ Hiba az Azure kliens inicializálása közben: {e}")
        return

    # Helyi gyorsítótár könyvtár kezelése
    TEMP_DATA_DIR.mkdir(exist_ok=True)
    print(f"📁 Ideiglenes adatok helye: {TEMP_DATA_DIR.resolve()}")
    
    model = None
    pool = None
    try:
        input_blob_path = BLOB_CLEANED_DOCUMENTS_PARQUET
        local_input_file = TEMP_DATA_DIR / Path(input_blob_path).name

        # Bemeneti fájl ellenőrzése és feltételes letöltése
        if local_input_file.exists():
            print(f"✅ Bemeneti fájl már létezik a helyi gyorsítótárban: {local_input_file}")
        else:
            print(f"⬇️ Bemeneti adatok letöltése: {input_blob_path} -> {local_input_file}")
            try:
                input_data = blob_storage.download_data(input_blob_path)
                with open(local_input_file, "wb") as f:
                    f.write(input_data)
                del input_data
                gc.collect()
                print("✅ Letöltés sikeres.")
            except Exception as e:
                print(f"❌ Hiba a letöltés során: {e}")
                return
            
        print(f"\n--- Feldolgozás indítása {num_gpus} GPU-n ---")
        main_start_time = time.time()
        
        df_full = pd.read_parquet(local_input_file)
        
        if df_full.empty:
            print("⚠️ A bemeneti DataFrame üres, nincs mit feldolgozni.")
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
        
        if not all_chunk_records:
            print("⚠️ A dokumentumokból nem sikerült darabokat készíteni.")
            return

        print(f"Összesen {len(all_chunk_records):,} darab (chunk) készült.")
        result_df = pd.DataFrame(all_chunk_records)
        del df_full, all_chunk_records
        gc.collect()

        print(f"Modell betöltése: {MODEL_NAME}...")
        model = SentenceTransformer(
            MODEL_NAME, 
            trust_remote_code=True,
            cache_folder=os.path.join(tempfile.gettempdir(), 'sentence_transformers_cache')
        )

        print("Embeddingek generálása...")
        text_chunks_list = result_df['text_chunk'].tolist()
        
        # Több GPU-s feldolgozás indítása
        target_devices = [f'cuda:{i}' for i in range(num_gpus)]
        pool = model.start_multi_process_pool(target_devices=target_devices)
        
        # Az encode_multi_process automatikusan mutatja a progress bart
        embeddings = model.encode_multi_process(
            text_chunks_list,
            pool=pool,
            batch_size=BATCH_SIZE
        )
        
        model.stop_multi_process_pool(pool)
        pool = None # Biztosítjuk, hogy a finally blokk ne próbálja újra leállítani

        result_df['embedding'] = list(embeddings.astype(np.float32))

        if 'text' in result_df.columns:
            result_df = result_df.drop(columns=['text'])

        print(f"⬆️ Feldolgozott adatok feltöltése ({len(result_df):,} sor)...")
        output_blob_path = BLOB_DOCUMENTS_WITH_EMBEDDINGS_PARQUET
        
        buffer = io.BytesIO()
        result_df.to_parquet(buffer, index=False, engine='pyarrow', compression='snappy')
        buffer.seek(0)
        
        blob_storage.upload_data(data=buffer.getvalue(), blob_path=output_blob_path)
        
        print(f"✅ Feltöltés sikeres ide: {AZURE_CONTAINER_NAME}/{output_blob_path}")

        total_time = time.time() - main_start_time
        print(f"\n--- Feldolgozás befejezve {total_time:.2f} másodperc alatt ---")

    finally:
        print("🧹 Memória felszabadítása...")
        if pool:
            model.stop_multi_process_pool(pool)
        if model:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ==============================================================================
# === 7. SCRIPT FUTTATÁSA ===
# ==============================================================================
# A szkript futtatásához hívja meg a main() függvényt a notebook egy cellájában.
# A szkript végén lévő `main()` hívás automatikusan lefut, amikor a teljes
# scriptet végrehajtja. Ha cellánként futtatja, ezt a hívást hagyja a legvégére.
if __name__ == '__main__':
    main() 