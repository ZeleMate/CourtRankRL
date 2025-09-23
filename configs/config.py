# configs/config.py - CourtRankRL Configuration
import os
from pathlib import Path
import logging

# --- Project Structure ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = DATA_DIR / "models"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# --- Input Files ---
# Raw court documents (DOCX)
RAW_DOCUMENTS_DIR = RAW_DATA_DIR

# Development qrels file (JSONL format)
DEV_QRELS_FILE = DATA_DIR / "dev_qrels.jsonl"

# --- Output Files ---
# Processed chunks with metadata (JSONL)
CHUNKS_JSONL = PROCESSED_DATA_DIR / "chunks.jsonl"

# Processed documents list (JSONL, egy sor = {"doc_id": ...})
PROCESSED_DOCS_LIST = PROCESSED_DATA_DIR / "processed_docs.jsonl"

# Preprocess checkpoint (állapot mentése nagy futásokhoz)
CHECKPOINT_PATH = PROCESSED_DATA_DIR / "preprocess_checkpoint.json"

# Indexes
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNK_ID_MAP_PATH = INDEX_DIR / "chunk_id_map.json"

# RL Policy
RL_POLICY_PATH = MODELS_DIR / "rl_policy.pth"

# Query Results
BASELINE_RESULTS_DIR = DATA_DIR / "baseline_results"
RERANKED_RESULTS_DIR = DATA_DIR / "reranked_results"

# --- Model Configuration ---
# Embedding model: Qwen3-Embedding-0.6B (Hugging Face)
# Agents.md spec: Qwen3-Embedding-0.6B
EMBEDDING_MODEL_TYPE = "qwen3"

# Qwen3 konfiguráció - RunPod 5090 GPU optimalizált
QWEN3_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_DIMENSION = 1024  # Qwen3-0.6B kimeneti dimenzió

# Aktív embedding dimenzió
EMBEDDING_DIMENSION = QWEN3_DIMENSION

# --- Retrieval Configuration ---
TOP_K_BASELINE = 100
TOP_K_RERANKED = 10
RRF_K = 60  # Reciprocal Rank Fusion parameter

# --- RL Configuration ---
RL_LEARNING_RATE = 1e-4
RL_BATCH_SIZE = 32
RL_EPOCHS = 10
RL_HIDDEN_DIM = 64

# --- BM25 Configuration ---
BM25_K1 = 1.5
BM25_B = 0.75

# --- Preprocess / IO batching ---
# Hány sort írjunk ki egyben a JSONL-be, mielőtt ürítjük a memóriapuffert
CHUNK_WRITE_BATCH_SIZE = 200

# --- Memory guard (soft limit, bytes) ---
# RunPod 5090 GPU optimalizált (~20 GiB GPU memory)
MEMORY_SOFT_LIMIT_BYTES = 20 * 1024 * 1024 * 1024

# --- Embedding Configuration ---
# RunPod 5090 GPU optimalizált (24GB VRAM)
EMBEDDING_BATCH_SIZE = 128  # Teljes batch méret a 5090 GPU-hoz
EMBEDDING_MAX_LENGTH = 1024  # Teljes token hossz

# --- FAISS IVF konfiguráció ---
# Célszerű klaszterszám: nlist ~ sqrt(N). Határok és tréning mintakövetelmény.
FAISS_NLIST_MIN = 64
FAISS_NLIST_MAX = 2048
FAISS_TRAIN_POINTS_PER_CENTROID = 39  # FAISS heurisztika (~40 pont/centroid)
FAISS_TRAIN_MAX_POINTS = 100_000  # tréning minták felső korlátja (RAM kontroll)

# --- File Extensions (DOCX only) ---
SUPPORTED_TEXT_EXTENSIONS = ['.docx', '.DOCX']

# --- Text Cleaning Configuration ---
CLEANING_MIN_TEXT_LENGTH = 150  # Minimum karakterhossz, ami alatt a szöveget zajnak tekintjük

# --- Minimal Logging ---
LOGGING_LEVEL = logging.WARNING  # No logging as per spec
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
