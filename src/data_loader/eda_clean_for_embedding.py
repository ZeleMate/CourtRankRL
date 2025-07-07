# Ez a szkript felelős a `preprocess_documents.py` által generált nyers CSV
# fájl(ok) beolvasásáért, a szöveges adatok tisztításáért, szűréséért,
# és a végső, embedding készítésre előkészített CSV fájl mentéséért.
import pandas as pd
import sys
from pathlib import Path
import re
from tqdm import tqdm
import csv
import logging
import os

# --- PATH KONFIGURÁCIÓ ---
# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from configs import config
except ImportError as e:
    print(f"HIBA: configs modul import sikertelen: {e}")
    sys.exit(1)

# --- LOGGOLÁS ---
logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

# --- CSV FELDOLGOZÁSI LIMIT NÖVELÉSE ---
# Szükséges a nagyon hosszú szövegmezőket tartalmazó sorok kezeléséhez.
try:
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
    logging.info(f"CSV field size limit beállítva: {max_int}")
except (ValueError, TypeError) as e:
    logging.warning(f"Nem sikerült beállítani a CSV field size limitet: {e}")
    csv.field_size_limit(131072) # Default fallback

# --- GLOBÁLIS KONFIGURÁCIÓ ---
MIN_TEXT_LENGTH = config.CLEANING_MIN_TEXT_LENGTH

def clean_text(text):
    """
    Szöveg tisztítása: speciális karakterek, URL-ek, email címek és extra szóközök eltávolítása.
    """
    if not isinstance(text, str):
        return ""
    
    # \x00 null byte karakter eltávolítása
    text = text.replace('\x00', '')
    
    # URL-ek eltávolítása
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Email címek eltávolítása
    text = re.sub(r'\S+@\S+', '', text, flags=re.MULTILINE)
    
    # Nem-alfanumerikus karakterek eltávolítása (kivéve pont, vessző, magyar ékezetes karakterek)
    text = re.sub(r'[^\w\s.,-]', '', text)
    
    # Többszörös szóközök, tabulátorok, új sorok cseréje egyetlen szóközre
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    A teljes DataFrame tisztítása és szűrése.
    """
    initial_rows = len(df)
    logging.info(f"DataFrame tisztításának kezdete {initial_rows:,} sorral.")

    # tqdm integráció a pandas apply művelethez
    tqdm.pandas(desc="Szövegtisztítás")

    # Szöveg tisztítása a 'text' oszlopon
    if 'text' in df.columns:
        df['cleaned_text'] = df['text'].astype(str).progress_apply(clean_text)
        logging.info("A 'text' oszlop tisztítása befejeződött.")
    else:
        logging.warning("A 'text' oszlop nem található a DataFrame-ben. A szövegtisztítás kimarad.")
        df['cleaned_text'] = ''

    # Üres szövegek és túl rövid szövegek szűrése
    df = df[df['cleaned_text'].str.len() >= MIN_TEXT_LENGTH]
    
    final_rows = len(df)
    removed_rows = initial_rows - final_rows
    logging.info(f"Szűrés után {final_rows:,} sor maradt (eltávolítva: {removed_rows:,}).")
    
    return df

def main():
    """
    Fő feldolgozási logika.
    """
    logging.info("===== TISZTÍTÁSI ÉS ELŐKÉSZÍTÉSI FOLYAMAT INDUL =====")
    
    in_path = config.RAW_CSV_DATA_PATH
    out_path = config.CLEANED_CSV_DATA_PATH

    if not in_path.exists():
        logging.error(f"A bemeneti fájl nem található: {in_path}")
        print(f"❌ HIBA: A bemeneti fájl nem található: {in_path}")
        print("Kérlek, először futtasd a `preprocess_documents.py` szkriptet.")
        return

    try:
        logging.info(f"Bemeneti CSV beolvasása: {in_path}")
        df = pd.read_csv(
            in_path, 
            encoding=config.CSV_ENCODING, 
            quoting=csv.QUOTE_ALL,
            on_bad_lines='warn', # Hibás sorok jelzése
            engine='python' # Szükséges az on_bad_lines és a quoting beállításokhoz
        )
        logging.info(f"Sikeresen beolvasva {len(df):,} sor.")
    except Exception as e:
        logging.error(f"Hiba a CSV beolvasása közben: {e}")
        print(f"❌ HIBA a CSV beolvasása közben. A részletekért lásd a log fájlt.")
        return

    # DataFrame tisztítása
    cleaned_df = clean_dataframe(df)

    # Kimeneti mappa létrehozása, ha nem létezik
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logging.info(f"Tisztított DataFrame mentése: {out_path}")
        cleaned_df.to_csv(
            out_path, 
            index=False, 
            encoding=config.CSV_ENCODING, 
            quoting=csv.QUOTE_ALL
        )
        logging.info(f"Sikeresen mentve {len(cleaned_df):,} sor.")
    except Exception as e:
        logging.error(f"Hiba a tisztított CSV mentése közben: {e}")
        print(f"❌ HIBA a tisztított CSV mentése közben. A részletekért lásd a log fájlt.")
        return

    # Összegzés
    print("\n✅ TISZTÍTÁS ÉS ELŐKÉSZÍTÉS BEFEJEZVE!")
    print(f"📄 Bemeneti fájl: {in_path}")
    print(f"📄 Kimeneti fájl: {out_path}")
    print(f"📊 Eredeti sorok száma: {len(df):,}")
    print(f"📊 Tisztított sorok száma (min. {MIN_TEXT_LENGTH} karakter): {len(cleaned_df):,}")

if __name__ == "__main__":
    main()