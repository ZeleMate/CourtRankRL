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
# Raw court documents (RTF/DOCX)
RAW_DOCUMENTS_DIR = RAW_DATA_DIR

# Development qrels file (JSONL format)
DEV_QRELS_FILE = DATA_DIR / "dev_qrels.jsonl"

# --- Output Files ---
# Processed chunks with metadata (JSONL)
CHUNKS_JSONL = PROCESSED_DATA_DIR / "chunks.jsonl"

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
# Qwen3-Embedding-0.6B modell (helyi, API-kulcs nélküli)
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
# Qwen3 embedding alapértelmezett dimenzió: 1024 (felhasználó által definiálható 32-1024 között)
EMBEDDING_DIMENSION = 1024

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

# --- File Extensions ---
SUPPORTED_TEXT_EXTENSIONS = ['.rtf', '.docx', '.RTF', '.DOCX']

# --- Text Cleaning Configuration ---
CLEANING_MIN_TEXT_LENGTH = 150  # Minimum karakterhossz, ami alatt a szöveget zajnak tekintjük

# --- Minimal Logging ---
LOGGING_LEVEL = logging.WARNING  # No logging as per spec
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'