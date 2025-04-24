"""
Configuration settings for the project.
"""
import os # Import os to potentially read API key from environment
from pathlib import Path

# ------------------------------------------------------------------
# Alapvető elérési utak
# ------------------------------------------------------------------
# Projekt gyökérkönyvtár (feltételezve, hogy a config.py a 'configs' mappában van)
PROJECT_ROOT = Path(__file__).parent.parent

# Adatkönyvtár (ahol a .json, .docx, .rtf fájlok vannak)
# Módosítsd ezt a saját adatkönyvtárad elérési útjára!
# Példa: DATA_DIR = Path("/path/to/your/data")
DATA_DIR = PROJECT_ROOT / "BHGY-k" # Vagy adj meg abszolút útvonalat

# Kimeneti könyvtár (ahová a feldolgozott CSV kerül)
OUT_DIR = PROJECT_ROOT / "processed_data"
OUT_FILENAME = "processed_documents.csv" # Kimeneti fájlnév (tömörítés nélkül)

# ------------------------------------------------------------------
# NLP Feldolgozási beállítások
# ------------------------------------------------------------------
# Használt spaCy modell neve
# Győződj meg róla, hogy telepítve van: python -m spacy download hu_core_news_lg
NLP_MODEL = "hu_core_news_md" # Nagyobb, de potenciálisan pontosabb magyar modell

# spaCy feldolgozás batch mérete (nlp.pipe)
# Nagyobb érték gyorsabb lehet, de több memóriát használ.
# Kísérletezz ezzel az értékkel a rendszeredhez igazítva.
BATCH_SIZE = 256 # Default batch size

# spaCy párhuzamos feldolgozási szálak száma (n_process az nlp.pipe-ban)
# -1: Az összes elérhető CPU mag használata (nagyon memóriaigényes lehet!)
# 1: Nincs párhuzamosítás (legkisebb memóriahasználat)
# 2, 3, 4, ...: Megadott számú processz használata (kompromisszum sebesség és memória között)
# Kezdd 1-gyel vagy 2-vel 16GB RAM esetén, és figyeld a memóriahasználatot.
SPACY_N_PROCESS = 2 # Ajánlott kezdőérték 16GB RAM-hoz

# Tiltsa le a szükségtelen spaCy komponenseket a memória csökkentése érdekében?
# Ha csak lemmatizálásra és/vagy entitásfelismerésre van szükség, a parser letiltható.
# Ha az entitásfelismerés (NER) sem kell, azt is tiltsd le: ["parser", "ner"]
DISABLE_SPACY_COMPONENTS = True # True esetén letiltja a 'parser'-t

# Egyedi jogi stop szavak (opcionális, bővíthető)
# Ezeket a szavakat a spaCy alapértelmezett stop szavaihoz adjuk hozzá.
LEGAL_STOP_WORDS = {
    "alperes", "felperes", "peres", "felek", "ítélet", "végzés", "határozat",
    "bíróság", "törvényszék", "ítélőtábla", "kúria", "ügyészség", "ügyvéd",
    "eljárás", "kereset", "fellebbezés", "felülvizsgálat", "indítvány",
    "bizonyíték", "tanú", "szakértő", "tárgyalás", "jegyzőkönyv",
    "beadvány", "indokolás", "rendelkező", "rész",
    # Ide további releváns szavakat lehet felvenni
}

# ------------------------------------------------------------------
# Kimeneti CSV beállítások
# ------------------------------------------------------------------
# Oszlopok, amelyeket a kimeneti CSV fájlba írunk
# Válaszd ki a szükséges oszlopokat a memóriahasználat és a fájlméret csökkentése érdekében.
OUTPUT_COLUMNS = [
    'doc_id',
    'birosag',
    'jogterulet',
    'hatarozat_id_mappa',
    'text', # Eredeti szöveg (lehet, hogy nagy)
    'lemmatized_text', # Lemmatizált szöveg
    'plain_lemmas', # Ékezetmentesített lemmák
    'entities', # Kinyert entitások (JSON string)
    'metadata_json', # Eredeti metaadatok (JSON string)
    # További metaadat oszlopok a JSON-ból, ha szükségesek voltak a DataFrame-ben
    # pl. 'ev', 'hatarozat_szama', stb. (ezeket a 'records' összeállításánál kellene hozzáadni)
]

# Memória limit (GB) a chunk-olt CSV íráshoz
# Ha a DataFrame becsült memóriaigénye meghaladja ezt az értéket,
# a CSV fájl chunk-okban (részekben) lesz kiírva.
# Állítsd be a rendelkezésre álló RAM egy részére (pl. 50-70%).
MEMORY_LIMIT_GB = 12 # 8 GB limit 16 GB RAM esetén (hagy némi puffert)

# CSV írás chunk mérete (sorok száma)
# Csak akkor használatos, ha a MEMORY_LIMIT_GB-t túllépi a DataFrame.
# Nagyobb érték gyorsabb lehet, de több memóriát használ az írási művelet alatt.
# 0 vagy negatív érték esetén nem használ chunk-olt írást, még ha a limitet átlépi is.
CSV_WRITE_CHUNK_SIZE = 10000 # 10,000 soros darabokban írás

# ------------------------------------------------------------------
# Egyéb beállítások (ha szükséges)
# ------------------------------------------------------------------
# Pl. loggolási szint, stb.

# Ellenőrzés: DATA_DIR létezik-e (opcionális, a fő szkript is ellenőrzi)
# if not DATA_DIR.exists() or not DATA_DIR.is_dir():
#     print(f"Figyelmeztetés: A konfigurált DATA_DIR nem létezik vagy nem könyvtár: {DATA_DIR}")

print(f"Konfiguráció betöltve. Projekt gyökér: {PROJECT_ROOT}")
print(f"Adat könyvtár: {DATA_DIR}")
print(f"Kimeneti könyvtár: {OUT_DIR}")
print(f"spaCy modell: {NLP_MODEL}, Párhuzamos processzek: {SPACY_N_PROCESS}")

# Model configuration
# Choose appropriate model like 'SZTAKI-HLT/hubert-base-cc', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', or a specific LegalBERT
# Using OpenAI's API
EMBEDDING_MODEL_NAME = "openai/text-embedding-3-large"
# Update dimension based on the chosen model
EMBEDDING_DIMENSION = 3072  # Dimension for text-embedding-3-large

# OpenAI API Key - IMPORTANT: Set this as an environment variable 'OPENAI_API_KEY'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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") # Example, better to just read from env

# Search configuration
INITIAL_TOP_K = 20 # Number of candidates retrieved by semantic search for RL re-ranking
FINAL_TOP_K = 5    # Number of results shown to the user after re-ranking
# SIMILARITY_THRESHOLD = 0.7 # May not be needed if using vector search index directly

# Graph Database configuration
GRAPH_DB_URI = "bolt://localhost:7687" # Example for Neo4j
GRAPH_DB_USER = "neo4j"
GRAPH_DB_PASSWORD = "password" # Use environment variables in production

# RL configuration
RL_ALGORITHM = "GRPO" # or "PolicyGradient", "PPO", etc.
POLICY_NETWORK_PARAMS = {
    "input_dim": EMBEDDING_DIMENSION * (INITIAL_TOP_K + 1), # Updated dimension used here
    "hidden_dim": 256,
    "output_dim": INITIAL_TOP_K # Example: scores for each doc
}
LEARNING_RATE = 0.0001 # Adjusted learning rate for policy networks
DISCOUNT_FACTOR = 0.99
EPOCHS_PER_UPDATE = 5
TRAINING_BATCH_SIZE = 32 # Number of query-ranking pairs per update
MAX_TRAINING_ITERATIONS = 1000 # Total training iterations

# Data paths
RAW_DATA_PATH = "/Users/zelenyianszkimate/Downloads/BHGY-k" # Updated path
PROCESSED_DATA_PATH = "/Users/zelenyianszkimate/Downloads/Feldolgozott BHGY-k/feldolgozott_hatarozatok.csv.gz" # Path to the output of preprocess_documents.py
MODEL_SAVE_PATH = "models"
GRAPH_SCHEMA_PATH = "configs/graph_schema.json" # Optional: if schema is defined in a file
EXPERT_EVAL_PATH = "data/expert_evaluations.csv" # Path to store/load expert feedback
RL_AGENT_SAVE_PATH = "models/rl_agent.pt"