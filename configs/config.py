# configs/config.py
import os
from pathlib import Path
import logging

# --- Alapvető beállítások ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# --- Adatkönyvtárak létrehozása ---
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDING_DIR = DATA_DIR / "embeddings"
INDEX_DIR = DATA_DIR / "index"
GRAPH_DIR = DATA_DIR / "graph"
MODELS_DIR = DATA_DIR / "models"
EVAL_DIR = DATA_DIR / "evaluations"

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDING_DIR, INDEX_DIR, GRAPH_DIR, MODELS_DIR, EVAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# --- Lokális fájl elérési utak ---
RL_AGENT_PATH = MODELS_DIR / "rl_agent.pth"

# Feldolgozás előtti és utáni adatok
RAW_DOCUMENTS_CSV = RAW_DATA_DIR / "raw_documents.csv"
CLEANED_DATA_FOR_EMBEDDING_CSV = PROCESSED_DATA_DIR / "cleaned_data_for_embedding.csv"
CLEANED_DOCUMENTS_PARQUET = PROCESSED_DATA_DIR / "cleaned_documents.parquet"

# Embeddingek
DOCUMENTS_WITH_EMBEDDINGS_PARQUET = EMBEDDING_DIR / "documents_with_embeddings.parquet"

# FAISS Index és kapcsolódó fájlok
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
FAISS_DOC_ID_MAP_PATH = INDEX_DIR / "doc_id_map.json"

# Gráf
GRAPH_PATH = GRAPH_DIR / "document_graph.gpickle"

# Kiértékelések
EXPERT_EVALUATIONS_CSV = EVAL_DIR / "expert_evaluations.csv"


# --- Modell és darabolás beállítások ---
MODEL_NAME = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768
BATCH_SIZE = 100
MAX_SEQUENCE_LENGTH = 8192
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 200


# --- Szövegtisztítási beállítások ---
CLEANING_MIN_TEXT_LENGTH = 150 # Minimum karakterhossz, ami alatt a szöveget zajnak tekintjük

# --- Loggolási beállítások ---
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# --- Támogatott fájltípusok ---
SUPPORTED_TEXT_EXTENSIONS = ['.rtf', '.docx', '.txt']