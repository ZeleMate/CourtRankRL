# ==============================================================================
# === 🚀 RUNPOD A100 EMBEDDING GENERÁLÁS (AZURE INTEGRÁCIÓVAL) TELJES KÓD ===
# ==============================================================================

# === 1. KÖRNYEZET BEÁLLÍTÁSA ÉS TELEPÍTÉS ===
print("--- [1/6] Környezet beállítása és könyvtárak telepítése ---")
# A -q (quiet) kapcsoló csökkenti a kimenet zaját
!pip install -q -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q -U sentence-transformers accelerate pyarrow pandas tqdm transformers psutil bitsandbytes
!pip install -q -U adlfs azure-storage-blob fsspec
print("✅ Könyvtárak telepítve.\n")

# === IMPORTÁLÁSOK ===
import pandas as pd
import numpy as np
import gc
import json
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
import torch
import time
from tqdm.auto import tqdm
from typing import List, Dict
from pathlib import Path
import os
import psutil
from transformers import AutoTokenizer

# === GPU OPTIMALIZÁCIÓK ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# === 2. KONFIGURÁCIÓ ===
print("--- [2/6] Konfiguráció betöltése ---")

# --- Azure Blob Storage beállítások ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
if not AZURE_CONNECTION_STRING:
    raise ValueError("Az AZURE_CONNECTION_STRING környezeti változó nincs beállítva! Add hozzá a RunPod Secrets-hez.")
storage_options = {'connection_string': AZURE_CONNECTION_STRING}

AZURE_CONTAINER_NAME = "bhgy"  # <<-- CSERÉLD LE a saját konténered nevére!
INPUT_BLOB_NAME = "cleaned_data_for_embedding.csv"
OUTPUT_BLOB_NAME = "documents_with_embeddings.parquet"

INPUT_AZURE_PATH = f"az://{AZURE_CONTAINER_NAME}/{INPUT_BLOB_NAME}"
OUTPUT_AZURE_PATH = f"az://{AZURE_CONTAINER_NAME}/{OUTPUT_BLOB_NAME}"

# --- Lokális átmeneti fájlok ---
TEMP_DIR = Path("/tmp/embedding_job")
TEMP_DIR.mkdir(exist_ok=True, parents=True)
CHUNKS_PARQUET_PATH = TEMP_DIR / "temp_chunks.parquet"
CHUNK_EMBEDDINGS_PATH = TEMP_DIR / "temp_chunks_with_embeddings.parquet"

# --- Modell és darabolás beállítások ---
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024
BATCH_SIZE = 512
MAX_SEQUENCE_LENGTH = 8192
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 200

# Nyers, feldolgozott adatok CSV-ben
RAW_CSV_DATA_PATH = PROCESSED_DATA_DIR / 'raw_documents.csv'
CLEANED_CSV_DATA_PATH = PROCESSED_DATA_DIR / 'cleaned_data_for_embedding.csv'

# Tisztított, végleges adatok Parquet formátumban
CLEANED_PARQUET_DATA_PATH = PROCESSED_DATA_DIR / 'cleaned_documents.parquet'

# === SZÖVEGTISZTÍTÁSI BEÁLLÍTÁSOK ===
CLEANING_MIN_TEXT_LENGTH = 150 # Minimum karakterhossz, ami alatt a szöveget zajnak tekintjük

print(f"📦 Azure Konténer: {AZURE_CONTAINER_NAME}")
print(f"📄 Bemeneti Blob: {INPUT_AZURE_PATH}")
print(f"💾 Kimeneti Blob: {OUTPUT_AZURE_PATH}")
print(f"🤖 Modell: {MODEL_NAME}")
print(f"🔪 Chunking: Méret={CHUNK_SIZE}, Átfedés={CHUNK_OVERLAP}\n")

# === 3. OSZTÁLYOK ÉS FÜGGVÉNYEK DEFINÍCIÓJA ===
print("--- [3/6] Osztályok és függvények definiálása ---")

class A100EmbeddingGenerator:
    def __init__(self, model_name: str, batch_size: int, dimension: int, max_seq_length: int):
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = dimension
        self.max_seq_length = max_seq_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        print(f"🚀 A100 Embedding Generátor inicializálva: {self.device}")

    def load_model(self):
        if self.model: return
        print(f"🔄 Modell betöltése: '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)
        if hasattr(self.model, 'max_seq_length'): self.model.max_seq_length = self.max_seq_length
        self._warmup_model()
        print("✅ Modell betöltve és bemelegítve.")

    def _warmup_model(self):
        print("🔥 Modell bemelegítése...")
        self.generate_embeddings(["melegítés"])
        self._cleanup_memory()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model: raise RuntimeError("Modell nincs betöltve!")
        embeddings = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True, device=self.device)
        if embeddings.shape[1] != self.dimension:
            embeddings = embeddings[:, :self.dimension]
        return embeddings.astype(np.float32)

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def chunk_document(doc_id: str, text: str, tokenizer, chunk_size: int, overlap: int) -> List[Dict]:
    if not isinstance(text, str) or text.isspace(): return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= chunk_size:
        return [{'doc_id': doc_id, 'chunk_text': text, 'chunk_num': 1, 'total_chunks': 1}]
    
    chunks, start_index, chunk_num, stride = [], 0, 1, chunk_size - overlap
    while start_index < len(tokens):
        chunk_tokens = tokens[start_index : start_index + chunk_size]
        chunks.append({'doc_id': doc_id, 'chunk_text': tokenizer.decode(chunk_tokens), 'chunk_num': chunk_num})
        chunk_num += 1
        start_index += stride
    
    for chunk in chunks: chunk['total_chunks'] = len(chunks)
    return chunks

print("✅ Osztályok és függvények definiálva.\n")

# === 4. FŐ FELDOLGOZÁSI FOLYAMAT ===
print("--- [4/6] Fő feldolgozási folyamat indítása ---")
overall_start = time.time()

# =======================================================================
# === JAVÍTOTT KÓD: 1. LÉPÉS: DOKUMENTUMOK DARABOLÁSA (STREAMINGGEL) ===
# =======================================================================

print("\n[1/3] Dokumentumok darabolása...")
if CHUNKS_PARQUET_PATH.exists():
    print(f"🔄 Chunk fájl már létezik, betöltés: {CHUNKS_PARQUET_PATH}")
    chunks_df = pd.read_parquet(CHUNKS_PARQUET_PATH)
else:
    print(f"📖 CSV olvasás indítása Azure-ból (darabokban): {INPUT_AZURE_PATH}")
    
    # Létrehozzuk az iterátort, ami 50,000 soronként adagolja az adatot
    # Ez a memóriaigényt drasztikusan csökkenti.
    csv_iterator = pd.read_csv(
        INPUT_AZURE_PATH, 
        storage_options=storage_options, 
        dtype={'text': 'string'},
        chunksize=50000  # 50,000 sort olvasunk be egy ciklusban
    )
    
    print("Tokenizer betöltése...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    all_chunks = []
    
    # A tqdm segítségével látni fogod a haladást, ahogy dolgozza fel a nagy fájl darabjait
    print("Fájl feldolgozása darabokban...")
    for df_chunk in tqdm(csv_iterator, desc="CSV darabok feldolgozása"):
        # Most az 50,000 soros DataFrame-et daraboljuk tovább dokumentumonként
        for _, row in df_chunk.iterrows():
            # A fillna itt is fontos, ha egy chunkban üres sorok lennének
            text_content = row.get('text', '') or ''
            doc_chunks = chunk_document(row['doc_id'], text_content, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
            all_chunks.extend(doc_chunks)

    if not all_chunks: 
        raise ValueError("Nincs feldolgozható szöveg a bemeneti fájlban!")
    
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_parquet(CHUNKS_PARQUET_PATH, index=False)
    print(f"✅ Lokális chunk fájl mentve: {len(chunks_df):,} darab.")
    del csv_iterator, df_chunk, all_chunks, tokenizer; gc.collect()

# 2. LÉPÉS: CHUNK EMBEDDING GENERÁLÁS
print("\n[2/3] Chunk embedding generálás...")
if CHUNK_EMBEDDINGS_PATH.exists():
    print(f"🔄 Chunk embedding fájl létezik, betöltés: {CHUNK_EMBEDDINGS_PATH}")
    chunks_with_embeddings_df = pd.read_parquet(CHUNK_EMBEDDINGS_PATH)
else:
    generator = A100EmbeddingGenerator(MODEL_NAME, BATCH_SIZE, EMBEDDING_DIMENSION, MAX_SEQUENCE_LENGTH)
    generator.load_model()
    
    texts_to_process = chunks_df['chunk_text'].tolist()
    all_embeddings = []
    
    print(f"🔥 Embedding generálás {len(texts_to_process):,} chunk-ra...")
    for i in tqdm(range(0, len(texts_to_process), BATCH_SIZE), desc="Embedding batch-ek"):
        all_embeddings.extend(generator.generate_embeddings(texts_to_process[i:i + BATCH_SIZE]))
        if i > 0 and i % (BATCH_SIZE * 10) == 0: generator._cleanup_memory()

    chunks_df['embedding'] = all_embeddings
    chunks_df.to_parquet(CHUNK_EMBEDDINGS_PATH, index=False)
    chunks_with_embeddings_df = chunks_df
    print(f"✅ Lokális chunk embedding fájl mentve.")
    del generator, texts_to_process, all_embeddings; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# 3. LÉPÉS: AGGREGÁLÁS ÉS FELTÖLTÉS
print("\n[3/3] Aggregálás és feltöltés Azure-ba...")
aggregated_embeddings = chunks_with_embeddings_df.groupby('doc_id')['embedding'].apply(lambda s: np.mean(np.array(s.tolist()), axis=0)).reset_index()

df_orig_meta = pd.read_csv(INPUT_AZURE_PATH, storage_options=storage_options, usecols=lambda c: c != 'text')
df_text = pd.read_csv(INPUT_AZURE_PATH, storage_options=storage_options, usecols=['doc_id', 'text']).fillna({'text': ''})

final_df = pd.merge(df_orig_meta, aggregated_embeddings, on='doc_id')
def create_metadata_json(row: pd.Series) -> str:
    return json.dumps({k: str(v) for k, v in row.drop('embedding').dropna().to_dict().items()}, ensure_ascii=False)
final_df['metadata_json'] = final_df.apply(create_metadata_json, axis=1)

final_df_to_save = pd.merge(final_df[['doc_id', 'embedding', 'metadata_json']], df_text, on='doc_id')
final_df_to_save = final_df_to_save[['doc_id', 'text', 'embedding', 'metadata_json']]

print(f"💾 Végső Parquet feltöltése Azure-ba: {OUTPUT_AZURE_PATH}")
final_df_to_save.to_parquet(OUTPUT_AZURE_PATH, storage_options=storage_options, index=False, compression='snappy')
print("✅ Feltöltés sikeres!")

# TAKARÍTÁS ÉS ÖSSZEGZÉS
print("\n🗑️ Átmeneti lokális fájlok törlése...")
if CHUNKS_PARQUET_PATH.exists(): CHUNKS_PARQUET_PATH.unlink()
if CHUNK_EMBEDDINGS_PATH.exists(): CHUNK_EMBEDDINGS_PATH.unlink()

total_time = time.time() - overall_start
print("\n" + "=" * 70)
print("🎉 FELDOLGOZÁS SIKERESEN BEFEJEZVE! (AZURE)")
print(f"📄 Kimeneti Blob: {OUTPUT_AZURE_PATH}")
print(f"⏱️ Teljes idő: {total_time:.2f}s ({total_time/60:.2f} perc)")
print(f"📊 Feldolgozott dokumentumok: {len(final_df_to_save):,}")
print("=" * 70 + "\n")

# === 5. VALIDÁCIÓ ===
print("--- [5/6] Kimenet validálása Azure-ból ---")
try:
    df_sample = pd.read_parquet(OUTPUT_AZURE_PATH, storage_options=storage_options).head(5)
    sample_embedding = df_sample['embedding'].iloc[0]
    
    print("✅ VALIDÁCIÓ SIKERES!")
    print(f"🏷️ Oszlopok: {df_sample.columns.tolist()}")
    print(f"🔢 Aggregált embedding dimenziója: {len(sample_embedding)}")
    norm = np.linalg.norm(sample_embedding)
    print(f"📏 Első embedding normája: {norm:.4f} (normalizált, ~1.0 körül kell lennie)")
    print("\n📋 Minta (első sor):")
    print(df_sample.iloc[[0]].to_string())
except Exception as e:
    print(f"❌ Validációs hiba: {e}")

# === 6. BEFEJEZÉS ===
print("\n--- [6/6] Minden folyamat befejeződött. ---")