import pandas as pd
import logging
import os
import sys

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Debugging ---
# print("--- Debugging sys.path ---") # Eltávolítva
# print(f"Calculated project_root: {project_root}") # Eltávolítva
# print("Current sys.path:") # Eltávolítva
# for p in sys.path: # Eltávolítva
#     print(f"- {p}") # Eltávolítva
# print("--- End Debugging ---") # Eltávolítva
# --- End Debugging ---

# Konfiguráció importálása
try:
    from configs import config
    # logging.info("A 'configs.config' sikeresen importálva.") # Ez inkább debug print volt
except ModuleNotFoundError as e:
    # Használjunk logging-ot itt is, ha már beállítottuk, bár itt még nincs konfigból
    print(f"HIBA: Nem sikerült importálni a 'configs.config'-ot. Hiba: {e}")
    print("Győződj meg róla, hogy a 'configs' könyvtár létezik a projekt gyökerében és tartalmazza a '__init__.py' és 'config.py' fájlokat.")
    sys.exit(1)

# ------------------------------------------------------------------
# Konfiguráció betöltése
# ------------------------------------------------------------------
IN_CSV_PATH = config.RAW_CSV_DATA_PATH # Bemeneti "nyers" CSV
OUT_CSV_PATH = config.CLEANED_CSV_DATA_PATH # Kimeneti "tisztított" CSV
CSV_ENCODING = config.CSV_ENCODING # CSV kódolás
CSV_INDEX = config.CSV_INDEX # Írjuk-e az indexet a CSV-be
LOGGING_LEVEL = config.LOGGING_LEVEL # Loggolási szint
LOGGING_FORMAT = config.LOGGING_FORMAT # Loggolási formátum
MIN_TEXT_LENGTH = config.CLEANING_MIN_TEXT_LENGTH # Minimális szöveghossz a tisztításhoz

# Loggolás beállítása a központi konfigurációból
logging.basicConfig(
    level=LOGGING_LEVEL,
    format=LOGGING_FORMAT
)

# ------------------------------------------------------------------
# Fő végrehajtási blokk
# ------------------------------------------------------------------
def main():
    """
    Fő függvény az adatok tisztításához embedding generálás előtt.

    Beolvassa a nyers CSV adatokat, elvégzi a tisztítási lépéseket
    (hiányzó értékek, rövid szövegek, duplikátumok kezelése),
    és elmenti a tisztított adatokat egy új CSV fájlba.
    """
    logging.info("Adattisztító szkript indítása az embeddinghez való előkészítéshez...")

    if not IN_CSV_PATH.exists():
        logging.error(f"A bemeneti CSV fájl nem található: {IN_CSV_PATH}")
        logging.error("Kérlek, először futtasd a `preprocess_documents.py` szkriptet a nyers adatok előállításához.")
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

    logging.info("Adattisztítás megkezdése...")

    df_cleaned = df.dropna(subset=['text']) # NaN értékek eltávolítása a 'text' oszlopból
    rows_after_nan = len(df_cleaned)
    nan_removed = initial_rows - rows_after_nan
    if nan_removed > 0:
        logging.info(f"Eltávolítva {nan_removed} sor, ahol a 'text' oszlop hiányzott (NaN).")

    # Üres vagy csak whitespace-t tartalmazó stringek eltávolítása
    df_cleaned = df_cleaned[df_cleaned['text'].astype(str).str.strip().astype(bool)] 
    rows_after_empty = len(df_cleaned)
    empty_removed = rows_after_nan - rows_after_empty
    if empty_removed > 0:
        logging.info(f"Eltávolítva {empty_removed} sor üres vagy csak szóközöket tartalmazó 'text' miatt.")

    # Túl rövid szövegek eltávolítása
    df_cleaned = df_cleaned[df_cleaned['text'].str.len() >= MIN_TEXT_LENGTH]
    rows_after_short = len(df_cleaned)
    short_removed = rows_after_empty - rows_after_short
    if short_removed > 0:
        logging.info(f"Eltávolítva {short_removed} sor, ahol a 'text' rövidebb volt, mint {MIN_TEXT_LENGTH} karakter.")

    # Duplikált `doc_id`-k eltávolítása (az első előfordulás megtartása)
    duplicate_removed = 0
    if 'doc_id' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop_duplicates(subset=['doc_id'], keep='first')
        rows_after_duplicates = len(df_cleaned)
        duplicate_removed = rows_after_short - rows_after_duplicates
        if duplicate_removed > 0:
            logging.info(f"Eltávolítva {duplicate_removed} sor duplikált 'doc_id' miatt (az első előfordulás megtartva).")
    else:
        logging.warning("A 'doc_id' oszlop nem található, a duplikátumok ellenőrzése és eltávolítása kihagyva.")

    final_rows = len(df_cleaned)
    total_removed = initial_rows - final_rows
    logging.info(f"Adattisztítás befejezve. Összesen eltávolítva: {total_removed} sor.")
    logging.info(f"Tisztított adatsorok száma: {final_rows}")

    if final_rows > 0:
        logging.info(f"Tisztított adatok mentése CSV fájlba: {OUT_CSV_PATH}")
        try:
            OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True) # Győződjünk meg róla, hogy a mappa létezik
            df_cleaned.to_csv(OUT_CSV_PATH, encoding=CSV_ENCODING, index=CSV_INDEX)
            logging.info("Tisztított CSV fájl sikeresen elmentve.")
            print(f"A tisztított kimeneti CSV fájl itt található: {OUT_CSV_PATH}") # print itt maradhat a felhasználói visszajelzéshez
        except Exception as e:
            logging.error(f"Hiba történt a tisztított CSV fájl mentésekor: {e}")
            raise SystemExit("Hiba a tisztított CSV mentésekor.")
    else:
        logging.warning("Nincs tisztított adat a mentéshez.")
        print("Nem maradt adat a tisztítás után.") # print itt maradhat

    logging.info("Az adattisztító szkript futása befejeződött.")

if __name__ == '__main__':
    main()