# src/embedding/create_embeddings_cloud.py

import pandas as pd
import numpy as np
import gc
import torch
import time
import os
import sys
import glob
from tqdm.auto import tqdm
from pathlib import Path
import torch.multiprocessing as mp
from azure.storage.blob import BlobServiceClient
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- PATH KONFIGURÁCIÓ ---
# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from configs import config
except ImportError as e:
    print(f"HIBA: configs modul import sikertelen: {e}")
    sys.exit(1)

# ==============================================================================
# === WORKER LOGIKA (A FŐ SZKRIPTBE INTEGRÁLVA) ===
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

class EmbeddingGenerator:
    """Felelős az embedding modell betöltéséért és a szövegek embeddingjéért."""
    def __init__(self, model_name, batch_size, device):
        self.model = None
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
    
    def load_model(self):
        if self.model is None:
            print(f"[{self.device}] Modell betöltése: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
            print(f"[{self.device}] Modell betöltve.")
    
    def generate_embeddings(self, texts):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True
            ).astype(np.float32)

def process_data_chunk_on_worker(args):
    """
    Ez a fő worker függvény. Beolvas egy adatdarabot tartalmazó fájlt,
    legenerálja az embeddingeket, és az eredményt egy kimeneti fájlba menti.
    """
    worker_id, device, model_name, batch_size, chunk_size, chunk_overlap, input_file_path, output_file_path = args
    
    try:
        print(f"[Worker {worker_id}]: Indul, feldolgozza: {input_file_path}")
        processor = DocumentProcessor(chunk_size, chunk_overlap)
        generator = EmbeddingGenerator(model_name, batch_size, device)
        generator.load_model()

        df_input = pd.read_parquet(input_file_path)
        
        all_chunks = []; original_docs_info = {}
        for _, row in df_input.iterrows():
            chunks = processor.split_document(row['doc_id'], row['text'])
            if chunks:
                all_chunks.extend(chunks)
                meta_cols = {col: row.get(col) for col in df_input.columns if col != 'text'}
                original_docs_info[row['doc_id']] = meta_cols

        if not all_chunks: return None

        df_chunks = pd.DataFrame(all_chunks)
        df_chunks['embedding'] = list(generator.generate_embeddings(df_chunks['text_chunk'].tolist()))
        agg_embeddings = df_chunks.groupby('doc_id')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0))
        
        final_data = []
        for doc_id, doc_embedding in agg_embeddings.items():
            doc_info = original_docs_info.get(doc_id, {})
            doc_info['embedding'] = doc_embedding
            final_data.append(doc_info)
        
        result_df = pd.DataFrame(final_data)
        result_df.to_parquet(output_file_path)
        
        print(f"[Worker {worker_id}]: ✅ Befejezte, eredmény mentve: {output_file_path}")
        return output_file_path

    except Exception as e:
        import traceback
        print(f"\n\nCRITICAL ERROR IN WORKER {worker_id}: {e}")
        traceback.print_exc()
        return None
    finally:
        del processor, generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ==============================================================================
# === FŐ VEZÉRLŐ LOGIKA ===
# ==============================================================================

# === 1. FELHŐS KONFIGURÁCIÓ ===
# --- Azure ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = "bhgy"  # CSERÉLD LE a saját konténered nevére!
INPUT_BLOB_NAME = config.CLEANED_PARQUET_DATA_PATH.name
OUTPUT_BLOB_NAME = "documents_with_embeddings_final.parquet"

# --- Modell és Feldolgozás ---
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 96
CHUNK_SIZE_CHARS = 5000
CHUNK_OVERLAP_CHARS = 500

# --- Lokális útvonalak a RunPodon belül ---
LOCAL_TEMP_DIR = Path("/workspace/embedding_job_temp")
LOCAL_INPUT_FILE = LOCAL_TEMP_DIR / INPUT_BLOB_NAME

# === 2. FŐ VEZÉRLŐ FÜGGVÉNYEK ===

def download_from_azure(conn_str, container, blob_name, local_path):
    """Letölt egy fájlt az Azure Blob Storage-ból."""
    if not conn_str:
        raise ValueError("Azure connection string nincs beállítva (AZURE_CONNECTION_STRING)!")
    
    if local_path.exists():
        print(f"✅ A bemeneti fájl már létezik lokálisan: {local_path}")
        return
    
    print(f"⬇️ Fájl letöltése: az://{container}/{blob_name} -> {local_path}")
    local_path.parent.mkdir(exist_ok=True, parents=True)
    
    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
    
    with open(local_path, "wb") as download_file:
        downloader = blob_client.download_blob(max_concurrency=4)
        download_file.write(downloader.readall())
    print("✅ Letöltés sikeres.")

def upload_to_azure(conn_str, container, blob_name, local_path):
    """Feltölt egy fájlt az Azure Blob Storage-ba."""
    if not conn_str:
        raise ValueError("Azure connection string nincs beállítva (AZURE_CONNECTION_STRING)!")
        
    print(f"⬆️ Fájl feltöltése: {local_path} -> az://{container}/{blob_name}")
    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
    
    with open(local_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True, max_concurrency=4)
    print("✅ Feltöltés sikeres.")

# === 3. FŐ FELDOLGOZÁSI LOGIKA ===
def main():
    if not torch.cuda.is_available():
        print("❌ HIBA: CUDA nem elérhető. A szkript csak GPU-s környezetben futtatható.")
        return
        
    NUM_GPUS = torch.cuda.device_count()
    print(f"🔥 Talált GPU-k száma: {NUM_GPUS}")
    
    # 1. Bemeneti fájl letöltése
    try:
        download_from_azure(AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, INPUT_BLOB_NAME, LOCAL_INPUT_FILE)
    except Exception as e:
        print(f"❌ Hiba a letöltés során: {e}")
        return
        
    # 2. Fő feldolgozási ciklus
    print("\n--- Feldolgozás indítása több GPU-n (Fájl-alapú IPC) ---")
    main_start_time = time.time()
    
    mp.set_start_method('spawn', force=True)
    
    df_full = pd.read_parquet(LOCAL_INPUT_FILE)
    total_rows = len(df_full)
    
    # Az adatok egyenlő darabokra osztása a GPU-k között
    # Ez a teljes adathalmazra vonatkozik, nem daraboljuk tovább.
    # Minden worker egy nagy szeletet kap.
    df_chunks_for_gpus = np.array_split(df_full, NUM_GPUS)
    del df_full; gc.collect()

    worker_args = []
    temp_files_to_clean = []
    
    print("Ideiglenes fájlok létrehozása a workereknek...")
    for i, df_worker_chunk in enumerate(df_chunks_for_gpus):
        if not df_worker_chunk.empty:
            input_path = LOCAL_TEMP_DIR / f"input_worker_{i}.parquet"
            output_path = LOCAL_TEMP_DIR / f"output_worker_{i}.parquet"
            
            df_worker_chunk.to_parquet(input_path)
            temp_files_to_clean.append(input_path)
            
            worker_args.append((
                i, f'cuda:{i}', MODEL_NAME, BATCH_SIZE, 
                CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, 
                input_path, output_path
            ))

    # 3. Párhuzamos feldolgozás
    print("Worker processzek indítása...")
    with mp.Pool(processes=NUM_GPUS) as pool:
        result_paths = pool.map(process_data_chunk_on_worker, worker_args)
    
    # 4. Eredmények összefűzése
    print("\nAdatok összefűzése az ideiglenes fájlokból...")
    valid_result_paths = [p for p in result_paths if p is not None]
    
    if valid_result_paths:
        final_df = pd.concat(
            [pd.read_parquet(f) for f in valid_result_paths], 
            ignore_index=True
        )
        
        # 5. Végső feltöltés
        final_output_path = LOCAL_TEMP_DIR / "final_embeddings.parquet"
        final_df.to_parquet(final_output_path)
        temp_files_to_clean.extend(valid_result_paths)
        temp_files_to_clean.append(final_output_path)
        
        try:
            upload_to_azure(AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, OUTPUT_BLOB_NAME, final_output_path)
        except Exception as e:
            print(f"❌ Hiba a végső fájl feltöltése során: {e}")
        
        print(f"\n📊 Összesen {len(final_df):,} / {total_rows} dokumentum embeddingje jött létre.")
    else:
        print("⚠️ Nem lett feldolgozva adat, egyetlen worker sem tért vissza eredménnyel.")

    # 6. Takarítás
    print("Ideiglenes fájlok törlése...")
    for f in temp_files_to_clean:
        try:
            os.remove(f)
        except OSError:
            pass
    if LOCAL_TEMP_DIR.exists(): LOCAL_TEMP_DIR.rmdir()

    total_time_minutes = (time.time() - main_start_time) / 60
    print(f"\n🎉 Feldolgozás befejezve {total_time_minutes:.2f} perc alatt.")

if __name__ == "__main__":
    main() 