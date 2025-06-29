"""
Configuration settings for the project.
"""
import os
from pathlib import Path
import logging # Added for logging level
from dotenv import load_dotenv # Import load_dotenv

load_dotenv() # Load environment variables from .env file

# ------------------------------------------------------------------
# Alapvető elérési utak
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" # Corrected path from "BHGY-k"
OUT_DIR = PROJECT_ROOT / "processed_data"
RAW_DATA_CSV_FILENAME = "raw_data_for_eda.csv"
PROCESSED_DATA_PARQUET_FILENAME = "processed_documents_with_embeddings.parquet"
CLEANED_DATA_CSV_FILENAME = "cleaned_data_for_embedding.csv" # Új fájlnév a tisztított adatoknak

# ------------------------------------------------------------------
# Gráf kimeneti fájlok elérési útjai
# ------------------------------------------------------------------
GRAPH_OUTPUT_GML_PATH = "processed_data/graph_data/graph.gml"
GRAPH_OUTPUT_JSON_PATH = "processed_data/graph_data/graph.json"
GRAPH_OUTPUT_GRAPHML_PATH = "processed_data/graph_data/graph.graphml" # Új GraphML formátum
# Relatív útvonal a metaadat fájlhoz, a többi gráf fájlhoz hasonlóan
GRAPH_METADATA_PATH = "processed_data/graph_data/graph_metadata.json"

# Ensure processed_data directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Teljes útvonalak
RAW_CSV_DATA_PATH = OUT_DIR / RAW_DATA_CSV_FILENAME
PROCESSED_PARQUET_DATA_PATH = OUT_DIR / PROCESSED_DATA_PARQUET_FILENAME
CLEANED_CSV_DATA_PATH = OUT_DIR / CLEANED_DATA_CSV_FILENAME # Új elérési út a tisztított CSV-hez

# ------------------------------------------------------------------
# Adatfeldolgozási beállítások (preprocess_documents.py)
# ------------------------------------------------------------------
DATA_PROCESSING_CHUNK_SIZE = 500 # Feldolgozási egység mérete a fájlbejárásnál (opcionális)
# Támogatott szövegfájl kiterjesztések (kisbetűsen)
SUPPORTED_TEXT_EXTENSIONS = ['.docx', '.rtf']

# ------------------------------------------------------------------
# OpenAI Embedding beállítások (generate_embeddings.py)
# ------------------------------------------------------------------
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIMENSION = 1536
OPENAI_EMBEDDING_BATCH_SIZE = 100
OPENAI_EMBEDDING_MAX_TOKENS = 8191 # Maximális token szám a beágyazási modellhez
OPENAI_API_REQUEST_MAX_TOKENS = 300000 # Maximális token szám egy API kérésben
# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------------------
# FAISS Index Beállítások (build_faiss_index.py)
# ------------------------------------------------------------------
FAISS_INDEX_NLIST = 100  # Inverz fájl celláinak száma
FAISS_INDEX_NPROBE = 10  # Kereséskor vizsgált cellák száma

# Kimeneti oszlopok a Parquet fájlhoz (generate_embeddings.py)
EMBEDDING_OUTPUT_COLUMNS = [
    'doc_id',
    'birosag',
    'jogterulet',
    'hatarozat_id_mappa',
    'text',
    'embedding', # OpenAI embedding vektor
    'metadata_json',
    # Ide jöhetnek további oszlopok a CSV-ből, ha szükségesek
]

# ------------------------------------------------------------------
# Kimeneti fájl írási beállítások
# ------------------------------------------------------------------
# CSV írás (preprocess_documents.py)
CSV_ENCODING = 'utf-8'
CSV_INDEX = False # Írjuk-e a DataFrame indexet a CSV-be

# Parquet írás (generate_embeddings.py)
PARQUET_ENGINE = 'pyarrow'
PARQUET_INDEX = False # Írjuk-e a DataFrame indexet a Parquet-be
# WRITE_CHUNK_SIZE = 10000 # Parquet írás chunk mérete (ha szükséges lenne)

# ------------------------------------------------------------------
# Loggolási beállítások
# ------------------------------------------------------------------
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ------------------------------------------------------------------
# Adattisztítási beállítások (eda_clean_for_embedding.py)
# ------------------------------------------------------------------
CLEANING_MIN_TEXT_LENGTH = 50 # Minimális szöveghossz a tisztításhoz

# ------------------------------------------------------------------
# Egyéb beállítások (pl. modell, keresés, gráf DB, RL - ezek már itt voltak)
# ------------------------------------------------------------------
EMBEDDING_MODEL_NAME = OPENAI_EMBEDDING_MODEL
EMBEDDING_DIMENSION = OPENAI_EMBEDDING_DIMENSION
INITIAL_TOP_K = 20
FINAL_TOP_K = 5
GRAPH_DB_URI = "bolt://localhost:7687"
GRAPH_DB_USER = os.getenv("NEO4J_USER", "neo4j")
GRAPH_DB_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
RL_ALGORITHM = "GRPO"
POLICY_NETWORK_PARAMS = {
    "input_dim": EMBEDDING_DIMENSION * (INITIAL_TOP_K + 1),
    "hidden_dim": 256,
    "output_dim": INITIAL_TOP_K
}
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.99
EPOCHS_PER_UPDATE = 5
TRAINING_BATCH_SIZE = 32
MAX_TRAINING_ITERATIONS = 1000
MODEL_SAVE_PATH = "models"
GRAPH_SCHEMA_PATH = "configs/graph_schema.json"
EXPERT_EVAL_PATH = "data/expert_evaluations.csv"
RL_AGENT_SAVE_PATH = "models/rl_agent.pt"

# Figyelmeztetés, ha az API kulcs nincs beállítva
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is not set as an environment variable. Please ensure it is defined in your .env file or environment.")
else:
    # Optional: Print a confirmation that the key is loaded, but mask the actual key
    print("OpenAI API Key loaded successfully from environment.")

print(f"Konfiguráció betöltve. Projekt gyökér: {PROJECT_ROOT}")
print(f"Adat könyvtár: {DATA_DIR}")
print(f"Kimeneti könyvtár: {OUT_DIR}")
print(f"Nyers adat CSV: {RAW_CSV_DATA_PATH}")
print(f"Tisztított adat CSV: {CLEANED_CSV_DATA_PATH}") # Kiíratás hozzáadása
print(f"Feldolgozott Parquet: {PROCESSED_PARQUET_DATA_PATH}")
print(f"OpenAI Embedding modell: {OPENAI_EMBEDDING_MODEL}")