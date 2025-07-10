# src/embedding/create_embeddings_cloud.py

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
import torch.multiprocessing as mp
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- PATH KONFIGURÁCIÓ ---
# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from configs import config
    from src.utils.azure_blob_storage import AzureBlobStorage
except ImportError as e:
    print(f"HIBA: Modul importálása sikertelen: {e}")
    sys.exit(1)

# ==============================================================================
# === WORKER LOGIKA ===
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
            # A modell cache-elését egy ideiglenes könyvtárba irányítjuk, hogy ne szemelje tele a RunPod tárhelyet
            self.model = SentenceTransformer(
                self.model_name, 
                device=self.device, 
                trust_remote_code=True,
                cache_folder=os.path.join(tempfile.gettempdir(), 'sentence_transformers_cache')
            )
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

def main():
    """Fő vezérlő függvény az embedding generáláshoz."""
    if not torch.cuda.is_available():
        print("❌ HIBA: CUDA nem elérhető. A szkript csak GPU-s környezetben futtatható.")
        return
        
    NUM_GPUS = torch.cuda.device_count()
    print(f"🔥 Talált GPU-k száma: {NUM_GPUS}")

    # Azure Blob Storage kliens
    try:
        blob_storage = AzureBlobStorage(container_name=config.AZURE_CONTAINER_NAME)
    except ValueError as e:
        print(f"❌ Hiba az Azure kliens inicializálása közben: {e}")
        return

    # Ideiglenes könyvtár létrehozása a feladat idejére
    local_temp_dir = Path(tempfile.mkdtemp(prefix="embedding_job_"))
    print(f"📁 Ideiglenes könyvtár létrehozva: {local_temp_dir}")
    
    try:
        # 1. Bemeneti fájl letöltése
        input_blob_path = config.BLOB_CLEANED_DOCUMENTS_PARQUET
        local_input_file = local_temp_dir / Path(input_blob_path).name

        print(f"⬇️ Bemeneti adatok letöltése: {input_blob_path}")
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
            
        # 2. Fő feldolgozási ciklus
        print("\n--- Feldolgozás indítása több GPU-n ---")
        main_start_time = time.time()
        
        mp.set_start_method('spawn', force=True)
        
        df_full = pd.read_parquet(local_input_file)
        
        if df_full.empty:
            print("⚠️ A bemeneti DataFrame üres, nincs mit feldolgozni.")
            return

        df_chunks_for_gpus = np.array_split(df_full, NUM_GPUS)
        del df_full; gc.collect()

        worker_args = []
        
        print("Ideiglenes fájlok létrehozása a workereknek...")
        for i, df_worker_chunk in enumerate(df_chunks_for_gpus):
            if not df_worker_chunk.empty:
                input_path = local_temp_dir / f"input_worker_{i}.parquet"
                output_path = local_temp_dir / f"output_worker_{i}.parquet"
                
                df_worker_chunk.to_parquet(input_path)
                
                worker_args.append((
                    i, f'cuda:{i}', config.MODEL_NAME, config.BATCH_SIZE, 
                    config.CHUNK_SIZE, config.CHUNK_OVERLAP, 
                    str(input_path), str(output_path)
                ))

        # 3. Párhuzamos feldolgozás
        print(f"Worker processzek indítása ({len(worker_args)} db)...")
        with mp.Pool(processes=NUM_GPUS) as pool:
            result_paths = pool.map(process_data_chunk_on_worker, worker_args)
        
        # 4. Eredmények összefűzése
        print("\nAdatok összefűzése az ideiglenes fájlokból...")
        valid_result_paths = [p for p in result_paths if p is not None and Path(p).exists()]
        
        if valid_result_paths:
            final_df = pd.concat(
                [pd.read_parquet(f) for f in valid_result_paths], 
                ignore_index=True
            )
            
            # 5. Végső feltöltés
            print(f"⬆️ Feldolgozott adatok feltöltése ({len(final_df):,} sor)...")
            output_blob_path = config.BLOB_DOCUMENTS_WITH_EMBEDDINGS_PARQUET
            
            buffer = io.BytesIO()
            final_df.to_parquet(buffer, index=False, engine='pyarrow', compression='snappy')
            buffer.seek(0)
            
            blob_storage.upload_data(data=buffer.getvalue(), blob_path=output_blob_path)
            
            print(f"✅ Feltöltés sikeres ide: {config.AZURE_CONTAINER_NAME}/{output_blob_path}")
        else:
            print("⚠️ Nem keletkezett feldolgozott adat.")

        total_time = time.time() - main_start_time
        print(f"\n--- Feldolgozás befejezve {total_time:.2f} másodperc alatt ---")

    finally:
        # 6. Ideiglenes könyvtár törlése
        print(f"🗑️ Ideiglenes könyvtár törlése: {local_temp_dir}")
        shutil.rmtree(local_temp_dir, ignore_errors=True)

if __name__ == '__main__':
    main() 