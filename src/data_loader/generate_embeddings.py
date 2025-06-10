# src/data_loader/generate_embeddings.py
import pandas as pd
import logging, os, sys, gc, time
from tqdm import tqdm
import numpy as np

# Calculate the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs import config
from models.embedding import OpenAIEmbeddingModel

# ------------------------------------------------------------------
# Konfiguráció betöltése
# ------------------------------------------------------------------
OUT_DIR = config.OUT_DIR
IN_CSV_PATH = config.CLEANED_CSV_DATA_PATH
OUT_PARQUET_PATH = config.PROCESSED_PARQUET_DATA_PATH
CSV_ENCODING = config.CSV_ENCODING
PARQUET_INDEX = config.PARQUET_INDEX
PARQUET_ENGINE = config.PARQUET_ENGINE

# --- ÚJ: Oszlop definíciók (Frissítve) ---
# A végső Parquet fájlhoz szükséges oszlopnevek
FINAL_OUTPUT_COLUMNS = [
    'doc_id', # Hozzáadva, ez lesz az azonosító
    'MeghozoBirosag',
    'JogTerulet',
    'Jogszabalyhelyek',
    'HatarozatEve',
    'AllKapcsolodoUgyszam',
    'AllKapcsolodoBirosag',
    'KapcsolodoHatarozatok',
    'text',
    'embedding'
]
# Azonosító oszlop az egyesítéshez és ellenőrzéshez
ID_COLUMN = 'doc_id'
# Szöveg oszlop az embeddinghez
TEXT_COLUMN = 'text'
# Nincs szükség átnevezésre, a nevek megegyeznek
# --- Oszlop definíciók VÉGE ---

# Ellenőrizzük a kimeneti könyvtárat
os.makedirs(OUT_DIR, exist_ok=True)

# Logging beállítása a konfigurációból
logging.basicConfig(
    level=config.LOGGING_LEVEL,
    format=config.LOGGING_FORMAT
)

# Initialize OpenAIEmbeddingModel
try:
    embedding_model_instance = OpenAIEmbeddingModel()
    # Az OpenAIEmbeddingModel __init__-je már ellenőrzi az API kulcsot
    logging.info(f"OpenAIEmbeddingModel sikeresen inicializálva a '{embedding_model_instance.model_name}' modellel.")
    # Az OPENAI_EMBEDDING_BATCH_SIZE-t az encode hívásakor adjuk át, ha az OpenAIEmbeddingModel támogatja.
    # Jelenleg az OpenAIEmbeddingModel encode metódusa fix batch_size=32-t használ.
    # Ezt lehetne paraméterezhetővé tenni az OpenAIEmbeddingModel-ben, és a configból átadni.
    # Most a config.OPENAI_EMBEDDING_BATCH_SIZE-t használom itt, de ezt az OpenAIEmbeddingModel-nek is tudnia kellene.
    # Frissítem az OpenAIEmbeddingModel-t, hogy fogadja a batch_size-t.
    EMBEDDING_BATCH_SIZE_FROM_CONFIG = config.OPENAI_EMBEDDING_BATCH_SIZE
except ValueError as e: # Ha az API kulcs hiányzik
    logging.error(f"Hiba az OpenAIEmbeddingModel inicializálásakor: {e}")
    raise SystemExit(f"Hiba: {e}")
except Exception as e:
    logging.error(f"Váratlan hiba az OpenAIEmbeddingModel inicializálásakor: {e}")
    raise SystemExit(f"Váratlan hiba az OpenAIEmbeddingModel inicializálásakor: {e}")

# ------------------------------------------------------------------
# Fő végrehajtási blokk
# ------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Bemeneti CSV és kimeneti Parquet ellenőrzése
    # ------------------------------------------------------------------
    if not IN_CSV_PATH.exists():
        logging.error(f"A bemeneti CSV fájl nem található: {IN_CSV_PATH}")
        logging.error("Kérlek, győződj meg róla, hogy a CSV fájl létezik a megadott helyen.")
        raise SystemExit(f"Hiba: Bemeneti CSV fájl hiányzik: {IN_CSV_PATH}")

    # --- MÓDOSÍTOTT RÉSZ: Ellenőrzés és Oszlopok Egyesítése/Szűrése ---
    if OUT_PARQUET_PATH.exists():
        logging.info(f"A kimeneti Parquet fájl már létezik: {OUT_PARQUET_PATH}")
        logging.info("Ellenőrzöm és frissítem az oszlopokat a szükségesekre.")

        try:
            logging.info("Meglévő Parquet fájl betöltése...")
            df_parquet = pd.read_parquet(OUT_PARQUET_PATH)
            logging.info(f"Parquet betöltve, {len(df_parquet)} sor, oszlopok: {df_parquet.columns.tolist()}")

            # Ellenőrizzük, hogy az ID oszlop ('doc_id') létezik-e a Parquet-ban
            if ID_COLUMN not in df_parquet.columns:
                 logging.error(f"A szükséges ID oszlop ('{ID_COLUMN}') nem található a meglévő Parquet fájlban. Az egyesítés/frissítés nem lehetséges.")
                 print(f"Hiba: Az '{ID_COLUMN}' oszlop hiányzik a Parquet fájlból.")
                 return # Kilépés

            # Ellenőrizzük, hogy az embedding oszlop létezik-e
            if 'embedding' not in df_parquet.columns:
                logging.error("Az 'embedding' oszlop hiányzik a meglévő Parquet fájlból. Újra kell generálni az embeddingeket.")
                print("Hiba: Az 'embedding' oszlop hiányzik a Parquet fájlból. Futtasd a szkriptet a Parquet fájl törlése után az újrageneráláshoz.")
                return # Kilépés

            # Mely oszlopok hiányoznak a FINAL_OUTPUT_COLUMNS listából (az ID-n és embeddingen kívül)?
            required_data_cols = [col for col in FINAL_OUTPUT_COLUMNS if col not in [ID_COLUMN, 'embedding']]
            cols_missing_in_parquet = [col for col in required_data_cols if col not in df_parquet.columns]

            df_merged = df_parquet # Kiindulunk a meglévőből

            if cols_missing_in_parquet:
                logging.info(f"A következő szükséges oszlopok hiányoznak a Parquet-ból: {cols_missing_in_parquet}")

                # Határozzuk meg a CSV-ből betöltendő oszlopokat (ID + hiányzók)
                csv_cols_to_load = [ID_COLUMN] + cols_missing_in_parquet

                logging.info(f"Eredeti CSV betöltése (csak a szükséges oszlopok): {IN_CSV_PATH} -> {csv_cols_to_load}")
                try:
                    df_csv = pd.read_csv(IN_CSV_PATH, encoding=CSV_ENCODING, usecols=lambda c: c in csv_cols_to_load)
                    loaded_csv_cols = df_csv.columns.tolist()
                    logging.info(f"CSV részlet betöltve, oszlopok: {loaded_csv_cols}")
                    # --- ÚJ: Ellenőrzés CSV betöltés után (merge ág) ---
                    check_missing_in_csv = set(csv_cols_to_load) - set(loaded_csv_cols)
                    if check_missing_in_csv:
                        logging.error(f"HIBA: A CSV betöltése után hiányoznak a következő várt oszlopok: {check_missing_in_csv}")
                        print(f"HIBA: Nem sikerült betölteni a CSV-ből: {check_missing_in_csv}")
                        return # Kilépés
                    else:
                        logging.info("CSV betöltés ellenőrzése sikeres: Minden kért oszlop betöltve.")
                    # --- Ellenőrzés VÉGE ---
                except ValueError as e:
                     logging.error(f"Hiba a CSV olvasásakor (valószínűleg hiányzó oszlop): {e}. Szükséges oszlopok: {csv_cols_to_load}")
                     print(f"Hiba: Nem találhatók a szükséges oszlopok a CSV-ben: {csv_cols_to_load}")
                     return

                # Ellenőrizzük, hogy a CSV tartalmazza-e a szükséges oszlopokat
                missing_in_csv = [col for col in csv_cols_to_load if col not in df_csv.columns]
                if missing_in_csv:
                    logging.error(f"A következő szükséges oszlopok hiányoznak a CSV fájlból: {missing_in_csv}. Az egyesítés nem lehetséges.")
                    print(f"Hiba: A következő oszlopok hiányoznak a CSV fájlból: {missing_in_csv}")
                    return # Kilépés

                # Egyesítés
                logging.info(f"DataFrame-ek egyesítése a '{ID_COLUMN}' oszlop alapján.")
                # Itt a df_merged felülíródik az egyesített verzióval
                df_merged = pd.merge(
                    df_parquet, # A már meglévő Parquet adatok
                    df_csv,     # A CSV-ből betöltött hiányzó oszlopok
                    on=ID_COLUMN,
                    how='left'  # Megtartjuk a Parquet összes sorát
                )
            else:
                logging.info("Nincs hiányzó oszlop a Parquet fájlban a CSV-ből való pótlásra.")

            # --- Közös rész a Parquet létezik ág végén ---
            # Kiválasztjuk CSAK a végső oszlopokat (ezzel töröljük a feleslegeseket)
            final_cols_present = [col for col in FINAL_OUTPUT_COLUMNS if col in df_merged.columns]
            missing_final_cols = set(FINAL_OUTPUT_COLUMNS) - set(final_cols_present)
            if missing_final_cols:
                logging.warning(f"A következő szükséges oszlopok hiányozni fognak a végső fájlból (nem voltak elérhetők): {missing_final_cols}")

            if not final_cols_present:
                 logging.error("Nem maradt egyetlen szükséges oszlop sem az egyesítés/szűrés után.")
                 print("Hiba: Nem sikerült előállítani a szükséges oszlopokat.")
                 return

            df_final = df_merged[final_cols_present] # Csak a szükséges oszlopok kiválasztása

            # --- ÚJ: Ellenőrzés mentés előtt (merge ág) ---
            final_df_cols = df_final.columns.tolist()
            logging.info(f"Mentés előtti ellenőrzés (merge ág): DataFrame oszlopok: {final_df_cols}")
            check_missing_before_save = set(FINAL_OUTPUT_COLUMNS) - set(final_df_cols)
            # Csak azokat a hiányzókat jelezzük hibaként, amik nem voltak már eleve hiányzók (missing_final_cols)
            critical_missing = check_missing_before_save - missing_final_cols
            if critical_missing:
                logging.error(f"HIBA mentés előtt: A DataFrame-ből váratlanul hiányoznak oszlopok: {critical_missing}")
                print(f"HIBA: Kritikus hiányzó oszlopok mentés előtt: {critical_missing}")
                # Dönthetünk úgy, hogy itt megállunk, vagy csak figyelmeztetünk
                # return
            elif check_missing_before_save:
                 logging.warning(f"Mentés előtti ellenőrzés: A DataFrame-ből hiányoznak oszlopok (de ezek várhatóan hiányoztak): {check_missing_before_save}")
            else:
                logging.info("Mentés előtti ellenőrzés sikeres: Minden szükséges oszlop megvan a DataFrame-ben.")
            # --- Ellenőrzés VÉGE ---

            # Mentés felülírással
            logging.info(f"Frissített adatok mentése (csak a szükséges oszlopokkal): {OUT_PARQUET_PATH} -> {df_final.columns.tolist()}")
            df_final.to_parquet(OUT_PARQUET_PATH, index=PARQUET_INDEX, engine=PARQUET_ENGINE)
            logging.info("Parquet fájl sikeresen frissítve.")

            # Ellenőrzés (opcionális, de hasznos)
            logging.info(f"Ellenőrzés: Újraolvasom a mentett Parquet fájlt ({OUT_PARQUET_PATH})")
            try:
                df_check = pd.read_parquet(OUT_PARQUET_PATH)
                final_columns_check = df_check.columns.tolist()
                logging.info(f"Ellenőrzött Parquet oszlopok: {final_columns_check}")
                # Ellenőrizzük, hogy a mentett oszlopok megegyeznek-e a df_final oszlopaival
                if set(df_final.columns) == set(final_columns_check):
                    logging.info("Ellenőrzés sikeres: A mentett Parquet fájl tartalmazza a várt oszlopokat.")
                    print("Ellenőrzés sikeres: A Parquet fájl frissítése a szükséges oszlopokra megtörtént.")
                else:
                    logging.error(f"Ellenőrzés sikertelen! Várt oszlopok: {list(df_final.columns)}, Mentett oszlopok: {final_columns_check}")
                    print(f"HIBA az ellenőrzés során: Az oszlopok nem egyeznek a mentés után!")
            except Exception as e:
                logging.error(f"Hiba történt a mentett Parquet fájl ellenőrzése közben: {e}")
                print(f"Hiba történt a frissített Parquet fájl ellenőrzése közben: {e}")

            print(f"A frissített Parquet fájl itt található: {OUT_PARQUET_PATH}")
            return # Sikeres frissítés és ellenőrzés után kilépünk

        except FileNotFoundError:
            logging.error(f"Hiba: Valamelyik fájl nem található az egyesítés során (CSV: {IN_CSV_PATH} vagy Parquet: {OUT_PARQUET_PATH})")
            print("Hiba: Fájl nem található az egyesítés során.")
            return
        except KeyError as e:
             logging.error(f"Hiba: Oszlop hiba az egyesítés során (valószínűleg '{ID_COLUMN}' vagy más oszlop hiányzik valamelyik fájlból). Részletek: {e}")
             print(f"Hiba: Oszlop hiba az egyesítés során. Ellenőrizd az '{ID_COLUMN}' oszlopnevet és a fájlok tartalmát.")
             return
        except Exception as e:
            logging.error(f"Hiba történt a hiányzó oszlopok egyesítése közben: {e}")
            print(f"Hiba történt az egyesítés során: {e}")
            return # Hiba esetén kilépünk

    # --- MÓDOSÍTOTT RÉSZ VÉGE ---

    # --- EREDETI KÓD KEZDETE (csak akkor fut le, ha a Parquet nem létezik) ---
    logging.info("A kimeneti Parquet fájl nem létezik. Új fájl generálása embeddingekkel.")
    # ------------------------------------------------------------------
    # Bemeneti CSV betöltése (csak a szükséges oszlopok)
    # ------------------------------------------------------------------
    # Oszlopok, amik kellenek a CSV-ből az új fájlhoz (ID, text, és a többi adat)
    csv_cols_needed_for_new = [col for col in FINAL_OUTPUT_COLUMNS if col != 'embedding']
    logging.info(f"Bemeneti CSV betöltése (csak a szükséges oszlopok): {IN_CSV_PATH} -> {csv_cols_needed_for_new}")
    try:
        df = pd.read_csv(IN_CSV_PATH, encoding=CSV_ENCODING, usecols=lambda c: c in csv_cols_needed_for_new)
        loaded_csv_cols_new = df.columns.tolist()
        logging.info(f"CSV sikeresen betöltve, {len(df)} sor, oszlopok: {loaded_csv_cols_new}")
        # --- ÚJ: Ellenőrzés CSV betöltés után (új fájl ág) ---
        check_missing_in_csv_new = set(csv_cols_needed_for_new) - set(loaded_csv_cols_new)
        if check_missing_in_csv_new:
            logging.error(f"HIBA: A CSV betöltése után hiányoznak a következő várt oszlopok: {check_missing_in_csv_new}")
            raise SystemExit(f"Hiba: Nem sikerült betölteni a CSV-ből: {check_missing_in_csv_new}")
        else:
            logging.info("CSV betöltés ellenőrzése sikeres: Minden kért oszlop betöltve.")
        # --- Ellenőrzés VÉGE ---
    except ValueError as e:
        logging.error(f"Hiba a CSV olvasásakor (valószínűleg hiányzó oszlop): {e}. Szükséges oszlopok: {csv_cols_needed_for_new}")
        raise SystemExit(f"Hiba: Nem találhatók a szükséges oszlopok a CSV-ben: {csv_cols_needed_for_new}")
    except Exception as e:
        logging.error(f"Hiba a CSV fájl betöltésekor: {e}")
        raise SystemExit("Hiba a CSV betöltésekor.")

    # Ellenőrizzük a szükséges oszlopokat a betöltött DataFrame-ben
    missing_in_df = [col for col in csv_cols_needed_for_new if col not in df.columns]
    if missing_in_df:
        # Ez elvileg a usecols miatt nem fordulhat elő, de dupla ellenőrzés
        logging.error(f"A betöltött DataFrame nem tartalmazza a következő szükséges oszlopokat: {missing_in_df}")
        raise SystemExit(f"Hiba: Hiányzó oszlopok a DataFrame-ben: {missing_in_df}")

    # Külön ellenőrizzük a text oszlopot az embeddinghez
    if TEXT_COLUMN not in df.columns:
         logging.error(f"A betöltött DataFrame nem tartalmaz '{TEXT_COLUMN}' oszlopot az embeddinghez.")
         raise SystemExit(f"Hiba: Hiányzó '{TEXT_COLUMN}' oszlop a DataFrame-ben.")

    # ------------------------------------------------------------------
    # Embeddingek generálása
    # ------------------------------------------------------------------
    logging.info(f"OpenAI embeddingek generálása a '{TEXT_COLUMN}' oszlop alapján az OpenAIEmbeddingModel segítségével...")
    
    texts_for_embedding_series = df[TEXT_COLUMN].astype(str)
    all_texts_list = texts_for_embedding_series.tolist()

    embeddings_array = np.array([]) # Kezdeti értékadás az unbound hiba elkerülésére

    try:
        embeddings_array = embedding_model_instance.encode(
            all_texts_list,
            batch_size=EMBEDDING_BATCH_SIZE_FROM_CONFIG
        )
        logging.info(f"Sikeresen lekérdezve {len(embeddings_array)} embedding eredmény (beleértve a lehetséges NaN vektorokat).")

        # Az embeddingek hozzárendelése a DataFrame-hez.
        # Az embeddings_array sorai vagy érvényes embeddingek vagy NaN vektorok.
        # A DataFrame-be listaként vagy None-ként szeretnénk őket tárolni.
        
        processed_embeddings_for_df = []
        failed_count = 0
        for i in range(embeddings_array.shape[0]):
            # Ellenőrizzük, hogy az egész sor (vektor) NaN-e
            if np.all(np.isnan(embeddings_array[i])):
                processed_embeddings_for_df.append(None) # Pandas jobban kezeli a None-t az object oszlopokban
                failed_count += 1
            else:
                processed_embeddings_for_df.append(embeddings_array[i].tolist()) # Lista formában
        
        df['embedding'] = processed_embeddings_for_df
        
        if failed_count > 0:
            logging.warning(f"{failed_count} sornál nem sikerült embeddinget generálni (None értékkel helyettesítve).")
        
    except Exception as e:
        logging.error(f"Kritikus hiba az embeddingek generálása és feldolgozása közben: {e}")
        # Hiba esetén az embedding oszlopot None értékekkel töltjük fel
        df['embedding'] = pd.Series([None] * len(df), dtype=object)
        logging.warning(f"Minden embedding None értékre lett állítva a hiba miatt.")

    logging.info("Embeddingek generálása és DataFrame-hez adása befejezve.")
    # A failed_embeddings számlálót már fentebb kezeltük (failed_count)

    # ------------------------------------------------------------------
    # Kimeneti Parquet fájl írása
    # ------------------------------------------------------------------
    # Nincs szükség átnevezésre

    # Kiválasztjuk a végső oszlopokat (most már az embeddinggel együtt)
    final_cols_present_new = [col for col in FINAL_OUTPUT_COLUMNS if col in df.columns]
    missing_final_cols_new = set(FINAL_OUTPUT_COLUMNS) - set(final_cols_present_new)
    if missing_final_cols_new:
        logging.warning(f"A következő szükséges oszlopok hiányoznak a DataFrame-ből a mentés előtt: {missing_final_cols_new}")

    if not final_cols_present_new:
        logging.error("Nincsenek érvényes kimeneti oszlopok a Parquet fájl írásához.")
        raise SystemExit("Hiba: Nincsenek kimeneti oszlopok.")

    df_output = df[final_cols_present_new] # Csak a szükséges oszlopok kiválasztása

    # --- ÚJ: Ellenőrzés mentés előtt (új fájl ág) ---
    final_df_cols_new = df_output.columns.tolist()
    logging.info(f"Mentés előtti ellenőrzés (új fájl ág): DataFrame oszlopok: {final_df_cols_new}")
    check_missing_before_save_new = set(FINAL_OUTPUT_COLUMNS) - set(final_df_cols_new)
    # Csak azokat a hiányzókat jelezzük hibaként, amik nem voltak már eleve hiányzók (missing_final_cols_new)
    critical_missing_new = check_missing_before_save_new - missing_final_cols_new
    if critical_missing_new:
        logging.error(f"HIBA mentés előtt: A DataFrame-ből váratlanul hiányoznak oszlopok: {critical_missing_new}")
        print(f"HIBA: Kritikus hiányzó oszlopok mentés előtt: {critical_missing_new}")
        # Itt is dönthetünk a megállásról
        # raise SystemExit("Hiba: Kritikus hiányzó oszlopok mentés előtt.")
    elif check_missing_before_save_new:
        logging.warning(f"Mentés előtti ellenőrzés: A DataFrame-ből hiányoznak oszlopok (de ezek várhatóan hiányoztak): {check_missing_before_save_new}")
    else:
        logging.info("Mentés előtti ellenőrzés sikeres: Minden szükséges oszlop megvan a DataFrame-ben.")
    # --- Ellenőrzés VÉGE ---

    logging.info(f"Adatok mentése ÚJ Parquet fájlba (csak a szükséges oszlopokkal): {OUT_PARQUET_PATH} -> {df_output.columns.tolist()}")
    try:
        df_output.to_parquet(OUT_PARQUET_PATH, index=PARQUET_INDEX, engine=PARQUET_ENGINE)
        logging.info("Új Parquet fájl sikeresen elmentve.")
    except Exception as e:
        logging.error(f"Hiba történt a Parquet fájl mentésekor: {e}")
        raise SystemExit("Hiba a Parquet mentésekor.")

    del df, df_output, embeddings_array
    gc.collect()

    print("Az embedding generáló szkript futása befejeződött.")
    print(f"A kimeneti Parquet fájl itt található: {OUT_PARQUET_PATH}")
    # --- EREDETI KÓD VÉGE ---

if __name__ == '__main__':
    main()