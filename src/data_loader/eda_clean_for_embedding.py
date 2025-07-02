import pandas as pd
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Debug: config import ellenőrzése
configs_path = Path(project_root) / "configs"
if not configs_path.exists():
    print(f"HIBA: configs mappa nem található: {configs_path}")
    print(f"Project root: {project_root}")
    print(f"Working directory: {os.getcwd()}")
    sys.exit(1)

# ===== CHUNKED CLEANING KONFIGURÁCIÓJA =====
ENABLE_UNIFIED_CSV = True  # Egyesített cleaned CSV létrehozása backwards compatibility-ért
USE_CHUNKED_INPUT = True   # Chunked input használata (ha elérhető)

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

def clean_single_chunk(chunk_df):
    """
    Egyetlen chunk adattisztítása (az eredeti tisztítási logika alapján).
    
    Returns:
        tuple: (cleaned_df, stats_dict)
    """
    initial_rows = len(chunk_df)
    
    if 'text' not in chunk_df.columns:
        logging.error("A chunk nem tartalmaz 'text' oszlopot!")
        return pd.DataFrame(), {'initial': initial_rows, 'final': 0, 'removed': initial_rows}
    
    # 1. NaN értékek eltávolítása a 'text' oszlopból
    df_cleaned = chunk_df.dropna(subset=['text'])
    rows_after_nan = len(df_cleaned)
    nan_removed = initial_rows - rows_after_nan
    
    # 2. Üres vagy csak whitespace-t tartalmazó stringek eltávolítása
    df_cleaned = df_cleaned[df_cleaned['text'].astype(str).str.strip().astype(bool)]
    rows_after_empty = len(df_cleaned)
    empty_removed = rows_after_nan - rows_after_empty
    
    # 3. Túl rövid szövegek eltávolítása
    df_cleaned = df_cleaned[df_cleaned['text'].str.len() >= MIN_TEXT_LENGTH]
    rows_after_short = len(df_cleaned)
    short_removed = rows_after_empty - rows_after_short
    
    # 4. Duplikált doc_id-k eltávolítása (az első előfordulás megtartása)
    duplicate_removed = 0
    if 'doc_id' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop_duplicates(subset=['doc_id'], keep='first')
        rows_after_duplicates = len(df_cleaned)
        duplicate_removed = rows_after_short - rows_after_duplicates
    
    final_rows = len(df_cleaned)
    total_removed = initial_rows - final_rows
    
    stats = {
        'initial': initial_rows,
        'final': final_rows,
        'removed': total_removed,
        'nan_removed': nan_removed,
        'empty_removed': empty_removed,
        'short_removed': short_removed,
        'duplicate_removed': duplicate_removed
    }
    
    return df_cleaned.reset_index(drop=True), stats

def save_cleaned_chunk(cleaned_df, chunk_idx):
    """
    Cleaned chunk mentése CSV fájlba.
    """
    if cleaned_df.empty:
        return None
    
    chunk_filename = f"cleaned_chunk_{chunk_idx:04d}.csv"
    chunk_dir = config.OUT_DIR / "chunked_cleaned"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / chunk_filename
    
    cleaned_df.to_csv(chunk_path, index=CSV_INDEX, encoding=CSV_ENCODING, errors='replace')
    
    logging.info(f"Cleaned chunk mentve: {chunk_filename} ({len(cleaned_df):,} rekord)")
    return chunk_path

def process_chunked_input():
    """
    Chunked input fájlok feldolgozása (ha elérhetők).
    
    Returns:
        tuple: (success, cleaned_chunk_files, total_stats)
    """
    chunked_raw_dir = config.OUT_DIR / "chunked_raw"
    
    if not chunked_raw_dir.exists():
        return False, [], {}
    
    raw_chunk_files = sorted(list(chunked_raw_dir.glob("raw_chunk_*.csv")))
    
    if not raw_chunk_files:
        return False, [], {}
    
    logging.info(f"Chunked input feldolgozás: {len(raw_chunk_files)} raw chunk találva")
    
    cleaned_chunk_files = []
    total_stats = {
        'total_initial': 0,
        'total_final': 0, 
        'total_removed': 0,
        'chunks_processed': 0
    }
    
    for chunk_idx, raw_chunk_path in enumerate(tqdm(raw_chunk_files, desc="Raw chunk-ok tisztítása")):
        try:
            # Raw chunk betöltése
            raw_df = pd.read_csv(raw_chunk_path, encoding=CSV_ENCODING)
            
            # Tisztítás
            cleaned_df, chunk_stats = clean_single_chunk(raw_df)
            
            # Statisztikák frissítése
            total_stats['total_initial'] += chunk_stats['initial']
            total_stats['total_final'] += chunk_stats['final']
            total_stats['total_removed'] += chunk_stats['removed']
            total_stats['chunks_processed'] += 1
            
            # Cleaned chunk mentése
            if not cleaned_df.empty:
                cleaned_chunk_path = save_cleaned_chunk(cleaned_df, chunk_idx)
                if cleaned_chunk_path:
                    cleaned_chunk_files.append(cleaned_chunk_path)
            else:
                logging.warning(f"Chunk {chunk_idx} teljesen üres a tisztítás után")
                
        except Exception as e:
            logging.error(f"Hiba a raw chunk feldolgozásában ({raw_chunk_path}): {e}")
            continue
    
    return True, cleaned_chunk_files, total_stats

def process_unified_input():
    """
    Egyesített CSV input feldolgozása (fallback vagy ha nincs chunked input).
    
    Returns:
        tuple: (success, cleaned_data_df, total_stats)
    """
    if not IN_CSV_PATH.exists():
        return False, pd.DataFrame(), {}
    
    logging.info("Egyesített CSV feldolgozás (fallback mode)")
    
    try:
        df = pd.read_csv(IN_CSV_PATH, encoding=CSV_ENCODING)
        cleaned_df, stats = clean_single_chunk(df)
        
        total_stats = {
            'total_initial': stats['initial'],
            'total_final': stats['final'],
            'total_removed': stats['removed'],
            'chunks_processed': 1
        }
        
        return True, cleaned_df, total_stats
        
    except Exception as e:
        logging.error(f"Hiba az egyesített CSV betöltésében: {e}")
        return False, pd.DataFrame(), {}

# ------------------------------------------------------------------
# Fő végrehajtási blokk
# ------------------------------------------------------------------
def main():
    """
    Fő függvény az adatok tisztításához embedding generálás előtt.
    
    ÚJDONSÁG: Chunked input támogatással és memory-safe feldolgozással.
    """
    logging.info("Chunked adattisztító szkript indítása...")
    
    cleaned_chunk_files = []
    unified_cleaned_df = pd.DataFrame()
    total_stats = {}
    
    # ===== 1. CHUNKED INPUT FELDOLGOZÁS (PRIORITÁS) =====
    if USE_CHUNKED_INPUT:
        success, cleaned_chunk_files, total_stats = process_chunked_input()
        
        if success:
            logging.info("✅ Chunked input feldolgozás sikeres")
        else:
            logging.info("⚠️  Nincs chunked input - fallback egyesített CSV-re")
    
    # ===== 2. EGYESÍTETT INPUT FELDOLGOZÁS (FALLBACK) =====
    if not cleaned_chunk_files:  # Ha nincs chunked input vagy nem sikerült
        success, unified_cleaned_df, total_stats = process_unified_input()
        
        if not success:
            logging.error("Nincs elérhető input adat (sem chunked, sem egyesített)")
            logging.error("Kérlek, először futtasd a `preprocess_documents.py` szkriptet!")
            raise SystemExit("Hiba: Nincs bemeneti adat")
    
    # ===== 3. ÖSSZEGZŐ STATISZTIKÁK =====
    if total_stats:
        removal_percentage = (total_stats['total_removed'] / total_stats['total_initial'] * 100) if total_stats['total_initial'] > 0 else 0
        
        logging.info("📊 TISZTÍTÁSI ÖSSZEFOGLALÓ:")
        logging.info(f"  Input rekordok: {total_stats['total_initial']:,}")
        logging.info(f"  Output rekordok: {total_stats['total_final']:,}")
        logging.info(f"  Eltávolított rekordok: {total_stats['total_removed']:,} ({removal_percentage:.1f}%)")
        logging.info(f"  Feldolgozott chunk-ok: {total_stats['chunks_processed']}")
    
    # ===== 4. OPCIONÁLIS EGYESÍTETT CLEANED CSV =====
    unified_csv_created = False
    if ENABLE_UNIFIED_CSV:
        logging.info("Egyesített cleaned CSV létrehozása backwards compatibility-ért...")
        
        try:
            if cleaned_chunk_files:
                # Chunk-okból egyesítés
                all_cleaned_dfs = []
                for chunk_path in tqdm(cleaned_chunk_files, desc="Cleaned chunk-ok egyesítése"):
                    chunk_df = pd.read_csv(chunk_path, encoding=CSV_ENCODING)
                    if not chunk_df.empty:
                        all_cleaned_dfs.append(chunk_df)
                
                if all_cleaned_dfs:
                    final_cleaned_df = pd.concat(all_cleaned_dfs, ignore_index=True)
                else:
                    final_cleaned_df = pd.DataFrame()
                    
            elif not unified_cleaned_df.empty:
                # Már van egyesített DataFrame
                final_cleaned_df = unified_cleaned_df
            else:
                final_cleaned_df = pd.DataFrame()
            
            # Egyesített CSV mentése
            if not final_cleaned_df.empty:
                OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
                final_cleaned_df.to_csv(OUT_CSV_PATH, encoding=CSV_ENCODING, index=CSV_INDEX)
                unified_csv_created = True
                logging.info(f"Egyesített cleaned CSV mentve: {OUT_CSV_PATH} ({len(final_cleaned_df):,} sor)")
            else:
                logging.warning("Nincs tisztított adat az egyesített CSV-hez")
                
        except Exception as e:
            logging.error(f"Hiba az egyesített cleaned CSV létrehozásában: {e}")
    
    # ===== 5. VÉGSŐ ÜZENETEK =====
    print(f"\n✅ CHUNKED CLEANING BEFEJEZVE!")
    
    if cleaned_chunk_files:
        print(f"📁 Cleaned chunk fájlok ({len(cleaned_chunk_files)} db): {config.OUT_DIR / 'chunked_cleaned'}")
    
    if unified_csv_created:
        print(f"📄 Egyesített cleaned CSV: {OUT_CSV_PATH}")
        print(f"💡 Következő scriptek használhatják az egyesített CSV-t vagy a chunk-okat")
    else:
        print(f"⚠️  Nincs egyesített cleaned CSV - csak chunk-ok (memória kímélés)")
    
    if total_stats:
        print(f"📊 {total_stats['total_final']:,} tisztított rekord ({total_stats['total_removed']:,} eltávolítva)")
        print(f"🚀 Következő lépés: embedding generálás chunked módban")
    
    logging.info("Chunked adattisztító szkript befejezve.")

if __name__ == '__main__':
    main()