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
BM25_INDEX_DIR = INDEX_DIR / "bm25"
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.json"
BM25_STATS_PATH = BM25_INDEX_DIR / "bm25_stats.json"
BM25_TOKEN_CACHE_DIR = BM25_INDEX_DIR / "token_cache"
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNK_ID_MAP_PATH = INDEX_DIR / "chunk_id_map.json"

# RL Policy
RL_POLICY_PATH = MODELS_DIR / "rl_policy.pth"

# Query Results
BASELINE_RESULTS_DIR = DATA_DIR / "baseline_results"
RERANKED_RESULTS_DIR = DATA_DIR / "reranked_results"

# --- Model Configuration ---
# Embedding model: google/embeddinggemma-300m (Hugging Face)
# Agents.md spec: google/embeddinggemma-300m
EMBEDDING_MODEL_TYPE = "embeddinggemma"

# EmbeddingGemma konfiguráció – lokális M3 optimalizált setup
EMBEDDING_GEMMA_MODEL_NAME = "google/embeddinggemma-300m"
EMBEDDING_GEMMA_DIMENSION = 768  # EmbeddingGemma-300m kimeneti dimenzió

# Aktív embedding dimenzió
EMBEDDING_DIMENSION = EMBEDDING_GEMMA_DIMENSION

# --- Retrieval Configuration ---
# Lokális erőforrásokhoz optimalizált értékek
TOP_K_BASELINE = 100
TOP_K_RERANKED = 20
RRF_K = 60

# --- RL Configuration ---
# Lokális training paraméterek
RL_LEARNING_RATE = 1e-4
RL_BATCH_SIZE = 32
RL_EPOCHS = 15
RL_HIDDEN_DIM = 96

# --- BM25 Configuration ---
BM25_K1 = 1.5
BM25_B = 0.75

# --- Preprocess / IO batching ---
# Hány sort írjunk ki egyben a JSONL-be, mielőtt ürítjük a memóriapuffert
# Lokális disk I/O-hoz igazított batch méret
CHUNK_WRITE_BATCH_SIZE = 200

# --- Memory guard (soft limit, bytes) ---
# M3 MacBook Air soft limit (RAM)
MEMORY_SOFT_LIMIT_BYTES = 12 * 1024 * 1024 * 1024

# --- Embedding Configuration ---
# Lokális embedding paraméterek (MPS)
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_MAX_LENGTH = 1024

# --- FAISS IVF konfiguráció ---
# FAISS IVF beállítások – kis korpusz
FAISS_NLIST_MIN = 64
FAISS_NLIST_MAX = 1024
FAISS_TRAIN_POINTS_PER_CENTROID = 40
FAISS_TRAIN_MAX_POINTS = 200_000

# --- File Extensions (DOCX only) ---
SUPPORTED_TEXT_EXTENSIONS = ['.docx', '.DOCX']

# --- Text Cleaning Configuration ---
CLEANING_MIN_TEXT_LENGTH = 150  # Minimum karakterhossz, ami alatt a szöveget zajnak tekintjük

# --- Minimal Logging ---
LOGGING_LEVEL = logging.WARNING  # No logging as per spec
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# --- BM25S specifikus beállítások ---
BM25_USE_CACHE = True
BM25_STOPWORDS = None
BM25_USE_NUMBA = False
BM25_THREADS = 0  # 0 → autonóm; -1 → összes core, >0 → fix érték
BM25_TOP_K_CACHE = 200
