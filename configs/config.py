# configs/config.py
import os
from pathlib import Path
import logging

# --- Alapvető beállítások ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Azure Blob Storage beállítások ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
if not AZURE_CONNECTION_STRING:
    print("Figyelem: Az AZURE_CONNECTION_STRING környezeti változó nincs beállítva.")

AZURE_CONTAINER_NAME = "courtrankrl" 

# --- Blob Storage elérési utak ---
BLOB_RAW_DATA_DIR = "raw"
BLOB_PROCESSED_DATA_DIR = "processed"
BLOB_EMBEDDING_DIR = "embeddings"
BLOB_INDEX_DIR = "index"
BLOB_GRAPH_DIR = "graph"
BLOB_MODELS_DIR = "models"
BLOB_EVAL_DIR = "evaluations"

BLOB_RL_AGENT_PATH = f"{BLOB_MODELS_DIR}/rl_agent.pth"

# Feldolgozás előtti és utáni adatok
BLOB_RAW_DOCUMENTS_CSV = f"{BLOB_RAW_DATA_DIR}/raw_documents.csv"
BLOB_CLEANED_DATA_FOR_EMBEDDING_CSV = f"{BLOB_PROCESSED_DATA_DIR}/cleaned_data_for_embedding.csv"
BLOB_CLEANED_DOCUMENTS_PARQUET = f"{BLOB_PROCESSED_DATA_DIR}/cleaned_documents.parquet"

# Embeddingek
BLOB_DOCUMENTS_WITH_EMBEDDINGS_PARQUET = f"{BLOB_EMBEDDING_DIR}/documents_with_embeddings.parquet"

# FAISS Index és kapcsolódó fájlok
BLOB_FAISS_INDEX = f"{BLOB_INDEX_DIR}/faiss_index.bin"
BLOB_FAISS_DOC_ID_MAP = f"{BLOB_INDEX_DIR}/doc_id_map.json"

# Gráf
BLOB_GRAPH = f"{BLOB_GRAPH_DIR}/document_graph.gpickle"

# Kiértékelések
BLOB_EXPERT_EVALUATIONS_CSV = f"{BLOB_EVAL_DIR}/expert_evaluations.csv"


# --- Modell és darabolás beállítások ---
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024
BATCH_SIZE = 512
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