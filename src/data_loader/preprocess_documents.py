# Ez a szkript felelős a nyers dokumentumok (RTF, DOCX) és a hozzájuk tartozó
# JSON metaadatok feldolgozásáért, majd chunked CSV fájlokba történő mentéséért.
# MÓDOSÍTVA: Memory-safe chunked mentés az OOM problémák elkerülésére.
import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm
import sys
import os
import logging # logging importálása
from striprtf.striprtf import rtf_to_text

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = Path(__file__).resolve().parent.parent.parent  # data_loader -> src -> project_root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Debug: config import ellenőrzése
configs_path = project_root / "configs"
if not configs_path.exists():
    print(f"HIBA: configs mappa nem található: {configs_path}")
    print(f"Project root: {project_root}")
    print(f"Working directory: {os.getcwd()}")
    sys.exit(1)

try:
    from configs import config
except ImportError as e:
    print(f"HIBA: configs modul import sikertelen: {e}")
    print(f"Python path: {sys.path}")
    print(f"Configs path: {configs_path}")
    sys.exit(1)

# ===== CHUNKED MENTÉS KONFIGURÁCIÓJA =====
CHUNK_SIZE = 2000  # Rekordok száma chunk-onként (memória optimalizáláshoz)
ENABLE_UNIFIED_CSV = True  # Egyesített CSV létrehozása backwards compatibility-ért

# Loggolás beállítása a központi konfigurációból
# Ennek a config importálása UTÁN kell következnie
logging.basicConfig(
    level=config.LOGGING_LEVEL,
    format=config.LOGGING_FORMAT,
    # force=True # Szükséges lehet, ha a root logger már konfigurálva van máshol (pl. notebookban)
               # vagy ha a szkriptet többször importáljuk/futtatjuk ugyanabban a sessionben.
               # Óvatosan használandó, mivel felülírja a meglévő beállításokat.
)

def save_chunk_to_csv(chunk_records, chunk_idx, expected_cols):
    """
    Chunk mentése CSV fájlba a szemantikai kereshetőség megőrzésével.
    """
    if not chunk_records:
        return None
    
    # DataFrame létrehozása
    df_chunk = pd.DataFrame(chunk_records)
    
    # Hiányzó oszlopok hozzáadása (az eredeti logika alapján)
    for col in expected_cols:
        if col not in df_chunk.columns:
            df_chunk[col] = None
    
    # Oszlopok sorrendjének beállítása (az eredeti logika alapján)
    final_ordered_cols = [col for col in expected_cols if col in df_chunk.columns]
    other_cols = [col for col in df_chunk.columns if col not in final_ordered_cols]
    df_chunk = df_chunk[final_ordered_cols + other_cols]
    
    # Chunk fájl mentése
    chunk_filename = f"raw_chunk_{chunk_idx:04d}.csv"
    chunk_dir = config.OUT_DIR / "chunked_raw"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / chunk_filename
    
    df_chunk.to_csv(chunk_path, index=False, encoding=config.CSV_ENCODING, errors='replace')
    
    logging.info(f"Raw chunk mentve: {chunk_filename} ({len(df_chunk):,} rekord)")
    return chunk_path

# Adat könyvtár elérési útja a konfigurációból
root_dir_to_scan = project_root / 'data' # Ez a könyvtár lesz rekurzívan bejárva
paths = list(root_dir_to_scan.rglob('*')) # Az összes fájl és mappa lekérése

# ===== CHUNKED FELDOLGOZÁS VÁLTOZÓK =====
chunk_records = []  # Aktuális chunk rekordjai (korlátozott méret!)
chunk_idx = 0       # Chunk sorszáma
saved_chunks = []   # Mentett chunk fájlok listája
total_records = 0   # Statisztika

# Támogatott szövegfájl kiterjesztések
SUPPORTED_EXTENSIONS = tuple(ext.lower() for ext in config.SUPPORTED_TEXT_EXTENSIONS)

# Várt oszlopok (az eredeti logika alapján) 
expected_cols_for_raw_csv = [
    'doc_id', 'text', 'birosag', 'JogTerulet', 'Azonosito', 'MeghozoBirosag',
    'EgyediAzonosito', 'HatarozatEve', 'AllKapcsolodoUgyszam', 'AllKapcsolodoBirosag',
    'KapcsolodoHatarozatok', 'Jogszabalyhelyek'
]

logging.info(f"Chunked feldolgozás kezdése (chunk méret: {CHUNK_SIZE:,})")
logging.info(f"Találva {len(paths):,} potenciális fájl")

for path in tqdm(paths, desc="Dokumentumfájlok feldolgozása"): # tqdm progress bar
    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
        base_filename = path.stem # Fájlnév kiterjesztés nélkül
        text_path = path

        # A kapcsolódó JSON metaadat fájl nevének képzése
        # Pl. '123.docx' -> '123.RTF_OBH.JSON' (a logika alapján az RTF_OBH fix)
        json_filename = base_filename + '.RTF_OBH.JSON'
        json_path = path.with_name(json_filename)

        text_content = "" # Alapértelmezett üres szöveg
        if text_path.suffix.lower() == '.rtf':
            try:
                with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                    rtf_content = f.read()
                text_content = rtf_to_text(rtf_content, errors="ignore")
            except Exception as e:
                # Itt jobb lenne logging.warning vagy error
                print(f"Figyelmeztetés: Nem sikerült kinyerni a szöveget az RTF fájlból ({text_path}) a striprtf segítségével: {e}")
        elif text_path.suffix.lower() == '.docx':
            try:
                from docx import Document # Importálás csak itt, ha tényleg szükség van rá
                doc = Document(str(text_path))
                text_content = ' \n'.join(para.text for para in doc.paragraphs if para.text.strip()) # Üres paragrafusok kihagyása
            except Exception as e:
                print(f"Figyelmeztetés: Nem sikerült kinyerni a szöveget a DOCX fájlból ({text_path}): {e}")
        
        # Szöveg normalizálása: többszörös whitespace cseréje egy szóközre, felesleges szóközök eltávolítása az elejéről/végéről
        text_content = re.sub(r'\s+', ' ', text_content).strip()

        extracted_metadata = {} # Kinyert metaadatok tárolására
        all_related_ugyszam = [] # Kapcsolódó ügyszámok listája
        all_related_birosag = [] # Kapcsolódó bíróságok listája

        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    metadata_dict = json.load(jf)
                    # Feltételezzük, hogy a releváns adatok a 'List' kulcs alatt lévő lista első elemében vannak
                    if 'List' in metadata_dict and isinstance(metadata_dict['List'], list) and len(metadata_dict['List']) > 0:
                        extracted_metadata = metadata_dict['List'][0]
                        # Kapcsolódó határozatok adatainak kinyerése
                        if 'KapcsolodoHatarozatok' in extracted_metadata and isinstance(extracted_metadata['KapcsolodoHatarozatok'], list):
                            for related_case in extracted_metadata['KapcsolodoHatarozatok']:
                                if isinstance(related_case, dict):
                                    all_related_ugyszam.append(related_case.get('KapcsolodoUgyszam'))
                                    all_related_birosag.append(related_case.get('KapcsolodoBirosag'))
                                else:
                                    print(f"Figyelmeztetés: A KapcsolodoHatarozatok lista egyik eleme nem szótár a {json_path} fájlban.")
                                    all_related_ugyszam.append(None)
                                    all_related_birosag.append(None)
                        # Összetett 'Jogszabalyhelyek' és 'KapcsolodoHatarozatok' stringgé alakítása a CSV kompatibilitás érdekében
                        if 'Jogszabalyhelyek' in extracted_metadata and not isinstance(extracted_metadata['Jogszabalyhelyek'], (str, int, float, bool)):
                            extracted_metadata['Jogszabalyhelyek'] = json.dumps(extracted_metadata['Jogszabalyhelyek'], ensure_ascii=False)
                        if 'KapcsolodoHatarozatok' in extracted_metadata and not isinstance(extracted_metadata['KapcsolodoHatarozatok'], (str, int, float, bool)):
                            extracted_metadata['KapcsolodoHatarozatok'] = json.dumps(extracted_metadata['KapcsolodoHatarozatok'], ensure_ascii=False)
            except json.JSONDecodeError:
                print(f"Figyelmeztetés: Nem sikerült dekódolni a JSON fájlt: {json_path}")
            except Exception as e:
                print(f"Figyelmeztetés: Hiba a JSON fájl feldolgozása közben ({json_path}): {e}")
        # else: # Ha nincs JSON, a metaadatok üresek maradnak
            # Ide lehetne loggolást tenni, ha hiányzik a JSON, de a jelenlegi kód csendben továbbmegy.

        # Bíróság nevének kinyerése az elérési útból (fallback)
        birosag_from_path = None
        try:
            abs_root_dir = root_dir_to_scan.resolve()
            abs_path = path.resolve()
            if abs_path.is_relative_to(abs_root_dir):
                 rel_parts = abs_path.relative_to(abs_root_dir).parts
                 if len(rel_parts) > 1:
                    birosag_from_path = rel_parts[0]
            # else: # Ha nem relatív, nem tudjuk megállapítani
                 # print(f"Figyelmeztetés: Az útvonal ({path}) nem relatív a gyökérhez ({root_dir_to_scan}). A bíróság nem állapítható meg az útvonalból.")
        except Exception as e_path:
             print(f"Figyelmeztetés: Váratlan hiba a bíróság nevének útvonalból történő kinyerése közben ({path}): {e_path}")

        record = {
            'text': text_content,
            **extracted_metadata, # Kinyert metaadatok hozzáadása
            'AllKapcsolodoUgyszam': json.dumps(all_related_ugyszam, ensure_ascii=False) if all_related_ugyszam else None,
            'AllKapcsolodoBirosag': json.dumps(all_related_birosag, ensure_ascii=False) if all_related_birosag else None,
        }
        # doc_id beállítása: elsődlegesen a JSON-ból ('Azonosito'), másodlagosan a fájlnévből
        record['doc_id'] = extracted_metadata.get('Azonosito', base_filename)
        # Bíróság beállítása: elsődlegesen a JSON-ból ('MeghozoBirosag'), másodlagosan az útvonalból
        record['birosag'] = extracted_metadata.get('MeghozoBirosag', birosag_from_path)

        # Potenciálisan problémás vagy felesleges mezők eltávolítása
        record.pop('Szoveg', None) # Ha a JSON tartalmazta a teljes szöveget, itt eltávolítjuk
        record.pop('RezumeSzovegKornyezet', None)
        record.pop('DownloadLink', None)
        record.pop('metadata', None) # Ha a **extracted_metadata hozzáadta volna a teljes 'List' objektumot

        # ===== CHUNKED MENTÉS =====
        chunk_records.append(record)
        total_records += 1
        
        # Ha elérte a chunk méretet, mentés és reset
        if len(chunk_records) >= CHUNK_SIZE:
            chunk_path = save_chunk_to_csv(chunk_records, chunk_idx, expected_cols_for_raw_csv)
            if chunk_path:
                saved_chunks.append(chunk_path)
            chunk_records = []  # Reset a memória felszabadításához
            chunk_idx += 1

# ===== UTOLSÓ CHUNK MENTÉSE =====
if chunk_records:
    chunk_path = save_chunk_to_csv(chunk_records, chunk_idx, expected_cols_for_raw_csv)
    if chunk_path:
        saved_chunks.append(chunk_path)

# ===== ÖSSZEGZŐ STATISZTIKÁK =====
logging.info(f"Chunked feldolgozás befejezve:")
logging.info(f"  Feldolgozott rekordok: {total_records:,}")
logging.info(f"  Mentett chunk-ok: {len(saved_chunks)}")
logging.info(f"  Chunk-ok mappája: {config.OUT_DIR / 'chunked_raw'}")

# ===== OPCIONÁLIS EGYESÍTETT CSV (BACKWARDS COMPATIBILITY) =====
unified_csv_created = False
if ENABLE_UNIFIED_CSV and saved_chunks:
    logging.info("Egyesített CSV létrehozása backwards compatibility-ért...")
    logging.warning("FIGYELEM: Ez megnöveli a memóriahasználatot!")
    
    try:
        # Chunk-ok egyesítése
        all_chunk_dfs = []
        for chunk_path in saved_chunks:
            chunk_df = pd.read_csv(chunk_path, encoding=config.CSV_ENCODING)
            all_chunk_dfs.append(chunk_df)
        
        # Egyesített DataFrame
        unified_df = pd.concat(all_chunk_dfs, ignore_index=True)
        
        # Egyesített CSV mentése (az eredeti helyre)
        out_path = config.RAW_CSV_DATA_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        unified_df.to_csv(out_path, index=False, encoding=config.CSV_ENCODING, errors='replace')
        
        unified_csv_created = True
        logging.info(f"Egyesített CSV mentve: {out_path} ({len(unified_df):,} sor)")
        
        # Memória felszabadítás
        del all_chunk_dfs, unified_df
        
    except Exception as e:
        logging.error(f"Hiba az egyesített CSV létrehozásában: {e}")
        logging.info("A chunk-ok továbbra is elérhetők a chunked_raw mappában.")

# ===== VÉGSŐ ÜZENETEK =====
print(f"\n✅ CHUNKED PREPROCESSING BEFEJEZVE!")
print(f"📊 Feldolgozott rekordok: {total_records:,}")
print(f"📁 Chunk fájlok ({len(saved_chunks)} db): {config.OUT_DIR / 'chunked_raw'}")

if unified_csv_created:
    print(f"📄 Egyesített CSV (backwards compatibility): {config.RAW_CSV_DATA_PATH}")
    print(f"💡 Következő scriptek használhatják az egyesített CSV-t vagy a chunk-okat")
else:
    print(f"⚠️  Nincs egyesített CSV - csak chunk-ok (memória kímélés)")
    print(f"💡 Következő lépés: eda_clean_for_embedding.py módosítása chunked olvasásra")

print(f"🚀 Memory használat optimalizálva: max {CHUNK_SIZE:,} rekord memóriában egyszerre")