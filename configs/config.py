# configs/config.py
import os
from pathlib import Path
import logging

# --- Alapvető könyvtárak ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'processed_data'
TEMP_DIR = Path("/tmp/embedding_job")
TEMP_DIR.mkdir(exist_ok=True, parents=True)

# --- Azure Blob Storage beállítások ---
# Az AZURE_CONNECTION_STRING-et a környezeti változókból olvassuk (pl. RunPod Secrets)
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")
if not AZURE_CONNECTION_STRING:
    print("Figyelem: Az AZURE_CONNECTION_STRING környezeti változó nincs beállítva.")
    # Lehet, hogy itt hibát kellene dobni, de egyelőre csak figyelmeztetünk
    # raise ValueError("Az AZURE_CONNECTION_STRING környezeti változó nincs beállítva!")
storage_options = {'connection_string': AZURE_CONNECTION_STRING} if AZURE_CONNECTION_STRING else {}

AZURE_CONTAINER_NAME = "bhgy"
INPUT_BLOB_NAME_CSV = "cleaned_data_for_embedding.csv"
OUTPUT_BLOB_NAME_PARQUET = "documents_with_embeddings.parquet"

INPUT_AZURE_PATH = f"az://{AZURE_CONTAINER_NAME}/{INPUT_BLOB_NAME_CSV}"
OUTPUT_AZURE_PATH = f"az://{AZURE_CONTAINER_NAME}/{OUTPUT_BLOB_NAME_PARQUET}"


# --- Lokális adatfájlok útvonalai ---
# Nyers, feldolgozott adatok CSV-ben
RAW_CSV_DATA_PATH = PROCESSED_DATA_DIR / 'raw_documents.csv'
CLEANED_CSV_DATA_PATH = PROCESSED_DATA_DIR / 'cleaned_data_for_embedding.csv'

# Tisztított, végleges adatok Parquet formátumban
CLEANED_PARQUET_DATA_PATH = PROCESSED_DATA_DIR / 'cleaned_documents.parquet'

# Lokális átmeneti fájlok az embedding generáláshoz
CHUNKS_PARQUET_PATH = TEMP_DIR / "temp_chunks.parquet"
CHUNK_EMBEDDINGS_PATH = TEMP_DIR / "temp_chunks_with_embeddings.parquet"


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