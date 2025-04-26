# src/data_loader/generate_embeddings.py
import pandas as pd
import logging, os, sys, gc, time
from tqdm import tqdm
import pyarrow
import pyarrow.parquet as pq
from openai import OpenAI, RateLimitError, APIError
import tiktoken  # Import tiktoken

# Calculate the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Konfiguráció importálása
from configs import config

# ------------------------------------------------------------------
# Konfiguráció betöltése
# ------------------------------------------------------------------
OUT_DIR = config.OUT_DIR
IN_CSV_PATH = config.RAW_CSV_DATA_PATH
OUT_PARQUET_PATH = config.PROCESSED_PARQUET_DATA_PATH
OPENAI_API_KEY = config.OPENAI_API_KEY
OPENAI_EMBEDDING_MODEL = config.OPENAI_EMBEDDING_MODEL
OPENAI_EMBEDDING_BATCH_SIZE = config.OPENAI_EMBEDDING_BATCH_SIZE
OPENAI_EMBEDDING_DIMENSION = config.OPENAI_EMBEDDING_DIMENSION  # Get dimension from config
OUTPUT_COLUMNS = config.EMBEDDING_OUTPUT_COLUMNS
PARQUET_ENGINE = config.PARQUET_ENGINE
PARQUET_INDEX = config.PARQUET_INDEX
LOGGING_LEVEL = config.LOGGING_LEVEL
LOGGING_FORMAT = config.LOGGING_FORMAT
CSV_ENCODING = config.CSV_ENCODING

# Ellenőrizzük a kimeneti könyvtárat
os.makedirs(OUT_DIR, exist_ok=True)

# Logging beállítása a konfigurációból
logging.basicConfig(
    level=LOGGING_LEVEL,
    format=LOGGING_FORMAT
)

# Initialize OpenAI Client
if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
    logging.error("OpenAI API kulcs nincs megfelelően beállítva. Ellenőrizd a config.py-t vagy a környezeti változókat.")
    raise SystemExit("Hiba: Hiányzó OpenAI API kulcs.")

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info("OpenAI kliens sikeresen inicializálva.")
except Exception as e:
    logging.error(f"Hiba az OpenAI kliens inicializálásakor: {e}")
    raise SystemExit("Hiba az OpenAI kliens inicializálásakor.")

# Initialize tiktoken encoder for the specified model
try:
    encoding = tiktoken.encoding_for_model(OPENAI_EMBEDDING_MODEL)
    MAX_TOKENS = 8191  # Use 8191 to be safe
    logging.info(f"Tiktoken encoder for model '{OPENAI_EMBEDDING_MODEL}' loaded. Max tokens: {MAX_TOKENS}")
except Exception as e:
    logging.error(f"Could not load tiktoken encoding for model {OPENAI_EMBEDDING_MODEL}: {e}")
    encoding = tiktoken.get_encoding("cl100k_base")
    MAX_TOKENS = 8191
    logging.warning(f"Using default tiktoken encoder 'cl100k_base'. Max tokens set to {MAX_TOKENS}.")

# ------------------------------------------------------------------
# Segédfüggvények
# ------------------------------------------------------------------
def truncate_text(text: str, max_tokens: int) -> str:
    """Truncates text to a maximum number of tokens."""
    if not isinstance(text, str):  # Handle potential non-string inputs
        text = str(text)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return text

def get_openai_embeddings(texts: list[str], model: str, batch_size: int) -> list[list[float] | None]:
    """Fetches embeddings for a list of texts using OpenAI API with batching, truncation, and retries."""
    embeddings: list[list[float] | None] = [None] * len(texts)
    for i in tqdm(range(0, len(texts), batch_size), desc="OpenAI Embeddings"):
        batch_texts_original = texts[i:i + batch_size]
        processed_batch = []
        original_indices_in_batch = []  # Keep track of original indices for error reporting

        # Preprocess and truncate texts in the batch
        for idx, text in enumerate(batch_texts_original):
            original_index = i + idx
            original_indices_in_batch.append(original_index)
            if pd.notna(text) and str(text).strip():
                try:
                    truncated_text = truncate_text(str(text), MAX_TOKENS)
                    if len(truncated_text) < len(str(text)):
                        logging.debug(f"Truncated text at original index {original_index} due to token limit.")
                    processed_batch.append(truncated_text)
                except Exception as e:
                    logging.error(f"Error truncating text at original index {original_index}: {e}. Skipping.")
                    processed_batch.append(" ")  # Add placeholder for skipped text
            else:
                processed_batch.append(" ")  # Use a space for empty/NaN text

        if not processed_batch:  # Skip if batch is empty after processing
            continue

        retries = 3
        delay = 5
        while retries > 0:
            try:
                response = client.embeddings.create(
                    input=processed_batch,
                    model=model,
                )
                batch_embeddings = [item.embedding for item in response.data]

                # Assign embeddings back, handling potential length mismatch if truncation failed
                for j, emb in enumerate(batch_embeddings):
                    if j < len(original_indices_in_batch):
                        original_idx = original_indices_in_batch[j]
                        if pd.notna(texts[original_idx]) and str(texts[original_idx]).strip() and processed_batch[j] != " ":
                            embeddings[original_idx] = emb
                        else:
                            embeddings[original_idx] = None
                break  # Success
            except RateLimitError as e:
                logging.warning(f"Rate limit error encountered processing batch starting at original index {original_indices_in_batch[0]}: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
                retries -= 1
                delay *= 2  # Exponential backoff
            except APIError as e:
                if "context_length_exceeded" in str(e) or "maximum context length" in str(e):
                    logging.error(f"OpenAI API context length error processing batch starting at original index {original_indices_in_batch[0]} DESPITE TRUNCATION: {e}.")
                else:
                    logging.error(f"OpenAI API error processing batch starting at original index {original_indices_in_batch[0]}: {e}")
                for original_idx in original_indices_in_batch:
                    if original_idx < len(embeddings):
                        embeddings[original_idx] = None
                retries = 0
            except Exception as e:
                logging.error(f"Unexpected error fetching embeddings for batch starting at original index {original_indices_in_batch[0]}: {e}")
                for original_idx in original_indices_in_batch:
                    if original_idx < len(embeddings):
                        embeddings[original_idx] = None
                retries = 0

        if retries == 0 and any(embeddings[idx] is None for idx in original_indices_in_batch if idx < len(embeddings)):
            logging.error(f"Failed to get some/all embeddings for batch starting at original index {original_indices_in_batch[0]} after multiple retries or due to errors.")

    return embeddings

# ------------------------------------------------------------------
# Fő végrehajtási blokk
# ------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Bemeneti CSV betöltése
    # ------------------------------------------------------------------
    if not IN_CSV_PATH.exists():
        logging.error(f"A bemeneti CSV fájl nem található: {IN_CSV_PATH}")
        logging.error("Kérlek, először futtasd a preprocess_documents.py szkriptet.")
        raise SystemExit(f"Hiba: Bemeneti fájl hiányzik: {IN_CSV_PATH}")

    logging.info(f"Bemeneti CSV betöltése: {IN_CSV_PATH}")
    try:
        df = pd.read_csv(IN_CSV_PATH, encoding=CSV_ENCODING)
        logging.info(f"CSV sikeresen betöltve, {len(df)} sor.")
    except Exception as e:
        logging.error(f"Hiba a CSV fájl betöltésekor: {e}")
        raise SystemExit("Hiba a CSV betöltésekor.")

    if 'text' not in df.columns:
        logging.error("A bemeneti CSV fájl nem tartalmaz 'text' oszlopot.")
        raise SystemExit("Hiba: Hiányzó 'text' oszlop a CSV-ben.")

    # ------------------------------------------------------------------
    # Embeddingek generálása
    # ------------------------------------------------------------------
    logging.info("OpenAI embeddingek generálása (with truncation)...")
    texts_to_embed = df['text'].tolist()
    embeddings = get_openai_embeddings(
        texts_to_embed,
        model=OPENAI_EMBEDDING_MODEL,
        batch_size=OPENAI_EMBEDDING_BATCH_SIZE
    )
    df['embedding'] = embeddings
    logging.info("Embeddingek generálása befejezve.")

    failed_embeddings = df['embedding'].isna().sum()
    if failed_embeddings > 0:
        logging.warning(f"{failed_embeddings} sornál nem sikerült embeddinget generálni.")

    # ------------------------------------------------------------------
    # Kimeneti Parquet fájl írása
    # ------------------------------------------------------------------
    final_output_columns = [col for col in OUTPUT_COLUMNS if col in df.columns]
    missing_cols = set(OUTPUT_COLUMNS) - set(final_output_columns)
    if missing_cols:
        logging.warning(f"A következő definiált kimeneti oszlopok hiányoznak a DataFrame-ből és kihagyásra kerülnek: {missing_cols}")

    if not final_output_columns:
        logging.error("Nincsenek érvényes kimeneti oszlopok a Parquet fájl írásához.")
        raise SystemExit("Hiba: Nincsenek kimeneti oszlopok.")

    df_output = df[final_output_columns]

    logging.info(f"Adatok mentése Parquet fájlba: {OUT_PARQUET_PATH}")
    try:
        df_output.to_parquet(OUT_PARQUET_PATH, index=PARQUET_INDEX, engine=PARQUET_ENGINE)
        logging.info("Parquet fájl sikeresen elmentve.")
    except Exception as e:
        logging.error(f"Hiba történt a Parquet fájl mentésekor: {e}")
        raise SystemExit("Hiba a Parquet mentésekor.")

    del df, df_output, embeddings
    gc.collect()

    print("Az embedding generáló szkript futása befejeződött.")
    print(f"A kimeneti Parquet fájl itt található: {OUT_PARQUET_PATH}")

if __name__ == '__main__':
    main()