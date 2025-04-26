import pandas as pd
import logging
import os
import sys

# Calculate the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Correct: go up one level from data_loader
# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Debugging ---
print("--- Debugging sys.path ---")
print(f"Calculated project_root: {project_root}")
print("Current sys.path:")
for p in sys.path:
    print(f"- {p}")
print("--- End Debugging ---")
# --- End Debugging ---

# Konfiguráció importálása
try:
    from configs import config
    print("Successfully imported 'configs.config'") # Add success message
except ModuleNotFoundError as e:
    print(f"Failed to import 'configs.config'. Error: {e}")
    print("Please ensure the 'configs' directory exists at the project root and contains '__init__.py' and 'config.py'.")
    sys.exit(1) # Exit if import fails

# ------------------------------------------------------------------
# Konfiguráció betöltése
# ------------------------------------------------------------------
IN_CSV_PATH = config.RAW_CSV_DATA_PATH
OUT_CSV_PATH = config.CLEANED_CSV_DATA_PATH
CSV_ENCODING = config.CSV_ENCODING
CSV_INDEX = config.CSV_INDEX # Use the same index setting for output
LOGGING_LEVEL = config.LOGGING_LEVEL
LOGGING_FORMAT = config.LOGGING_FORMAT
MIN_TEXT_LENGTH = 50 # Minimum character length for text

# Logging beállítása a konfigurációból
logging.basicConfig(
    level=LOGGING_LEVEL,
    format=LOGGING_FORMAT
)

# ------------------------------------------------------------------
# Fő végrehajtási blokk
# ------------------------------------------------------------------
def main():
    logging.info("Adattisztító szkript indítása...")

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
        initial_rows = len(df)
        logging.info(f"CSV sikeresen betöltve, eredeti sorok száma: {initial_rows}")
    except Exception as e:
        logging.error(f"Hiba a CSV fájl betöltésekor: {e}")
        raise SystemExit("Hiba a CSV betöltésekor.")

    if 'text' not in df.columns:
        logging.error("A bemeneti CSV fájl nem tartalmaz 'text' oszlopot.")
        raise SystemExit("Hiba: Hiányzó 'text' oszlop a CSV-ben.")

    # ------------------------------------------------------------------
    # Tisztítási lépések
    # ------------------------------------------------------------------
    logging.info("Tisztítás megkezdése...")

    # 1. Üres vagy csak whitespace szövegek eltávolítása
    df_cleaned = df.dropna(subset=['text']) # Remove NaN in 'text'
    rows_after_nan = len(df_cleaned)
    nan_removed = initial_rows - rows_after_nan
    if nan_removed > 0:
        logging.info(f"Eltávolítva {nan_removed} sor, ahol a 'text' hiányzott (NaN).")

    df_cleaned = df_cleaned[df_cleaned['text'].str.strip().astype(bool)] # Remove empty/whitespace strings
    rows_after_empty = len(df_cleaned)
    empty_removed = rows_after_nan - rows_after_empty
    if empty_removed > 0:
        logging.info(f"Eltávolítva {empty_removed} sor üres vagy csak szóközöket tartalmazó 'text' miatt.")

    # 2. Túl rövid szövegek eltávolítása
    df_cleaned = df_cleaned[df_cleaned['text'].str.len() >= MIN_TEXT_LENGTH]
    rows_after_short = len(df_cleaned)
    short_removed = rows_after_empty - rows_after_short
    if short_removed > 0:
        logging.info(f"Eltávolítva {short_removed} sor, ahol a 'text' rövidebb volt, mint {MIN_TEXT_LENGTH} karakter.")

    # 3. Duplikált doc_id eltávolítása (ha létezik az oszlop)
    duplicate_removed = 0
    if 'doc_id' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop_duplicates(subset=['doc_id'], keep='first')
        rows_after_duplicates = len(df_cleaned)
        duplicate_removed = rows_after_short - rows_after_duplicates
        if duplicate_removed > 0:
            logging.info(f"Eltávolítva {duplicate_removed} sor duplikált 'doc_id' miatt (az első előfordulás megtartva).")
    else:
        logging.warning("A 'doc_id' oszlop nem található, a duplikátumok ellenőrzése kihagyva.")

    final_rows = len(df_cleaned)
    total_removed = initial_rows - final_rows
    logging.info(f"Tisztítás befejezve. Összesen eltávolítva: {total_removed} sor.")
    logging.info(f"Tisztított adatsorok száma: {final_rows}")

    # ------------------------------------------------------------------
    # Kimeneti tisztított CSV fájl írása
    # ------------------------------------------------------------------
    if final_rows > 0:
        logging.info(f"Tisztított adatok mentése CSV fájlba: {OUT_CSV_PATH}")
        try:
            df_cleaned.to_csv(OUT_CSV_PATH, encoding=CSV_ENCODING, index=CSV_INDEX)
            logging.info("Tisztított CSV fájl sikeresen elmentve.")
        except Exception as e:
            logging.error(f"Hiba történt a tisztított CSV fájl mentésekor: {e}")
            raise SystemExit("Hiba a tisztított CSV mentésekor.")
    else:
        logging.warning("Nincs tisztított adat a mentéshez.")

    print("Az adattisztító szkript futása befejeződött.")
    if final_rows > 0:
        print(f"A tisztított kimeneti CSV fájl itt található: {OUT_CSV_PATH}")
    else:
        print("Nem maradt adat a tisztítás után.")

if __name__ == '__main__':
    main()