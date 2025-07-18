# Ez a szkript felelős a nyers dokumentumok (RTF, DOCX) és a hozzájuk tartozó
# JSON metaadatok feldolgozásáért. A szkript először letölti a nyers adatokat
# egy ideiglenes helyi könyvtárba, ott feldolgozza őket, majd az eredményül
# kapott egységes Parquet fájlt feltölti az Azure Blob Storage-ba.
import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm
import sys
import os
import logging
from striprtf.striprtf import rtf_to_text
import csv
from docx import Document
from bs4 import BeautifulSoup
import html
import io
import tempfile
import shutil

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Konfiguráció és segédprogramok importálása
try:
    from configs import config
except ImportError as e:
    print(f"HIBA: Modul importálása sikertelen: {e}")
    sys.exit(1)

# Loggolás beállítása
logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

# Támogatott szövegfájl kiterjesztések
# Már a config fájlban definiálva van, így itt nincs rá szükség.
# SUPPORTED_EXTENSIONS = tuple(ext.lower() for ext in config.SUPPORTED_TEXT_EXTENSIONS)

def clean_text_for_embedding(text: str) -> str:
    """
    Szöveg tisztítása embedding generálás előtt.
    Eltávolítja a HTML tageket, URL-eket és egyéb technikai zajt.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # HTML entitások dekódolása (pl. &amp; -> &)
    try:
        text = html.unescape(text)
    except Exception:
        pass # Ha hiba történik, a nyers szöveggel megyünk tovább

    # HTML tagek eltávolítása BeautifulSoup-pal
    try:
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=' ')
    except Exception:
        # Ha a BeautifulSoup hibát dob, egyszerű regex-szel próbáljuk
        text = re.sub(r'<[^>]+>', '', text)
    
    # URL-ek, email címek eltávolítása
    text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text, flags=re.MULTILINE)

    # Null byte és egyéb nem szöveges vezérlő karakterek eltávolítása
    text = text.replace('\x00', '')
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # RTF specifikus maradványok eltávolítása, amik a konverzió után maradhatnak
    text = re.sub(r'\\[a-zA-Z]+\d*\s?', '', text)
    text = re.sub(r'[{}]', '', text)
    
    # Többszörös szóközök, tabulátorok, új sorok cseréje egyetlen szóközre
    text = re.sub(r'\\s+', ' ', text).strip()
    
    return text

def clean_surrogates(text):
    """Eltávolítja az érvénytelen surrogate párokat a stringből."""
    if not isinstance(text, str):
        return text
    return re.sub(r'[\\ud800-\\udfff]', '', text)


def process_local_files(local_dir: Path) -> pd.DataFrame:
    """
    Feldolgozza a helyi könyvtárban lévő dokumentumokat és metaadatokat.
    """
    logging.info("Helyi fájlok feldolgozásának megkezdése...")
    
    document_files = {}
    json_files = {}
    supported_extensions = tuple(ext.lower() for ext in config.SUPPORTED_TEXT_EXTENSIONS)

    for file_path in local_dir.rglob('*'):
        if not file_path.is_file():
            continue
        
        if file_path.suffix.lower() in supported_extensions:
            base_name = file_path.stem
            document_files[base_name] = file_path
        elif file_path.suffix.lower() == '.json':
            # A JSON fájl nevéből eltávolítjuk a specifikus OBH postfixeket
            base_name = re.sub(r'\.(RTF|DOCX)_OBH$', '', file_path.stem, flags=re.IGNORECASE)
            json_files[base_name] = file_path

    all_records = []
    logging.info(f"Feldolgozásra váró dokumentum párok száma: {len(document_files):,}")

    for base_filename, text_filepath in tqdm(document_files.items(), desc="Dokumentumok feldolgozása"):
        record = process_single_document(base_filename, text_filepath, json_files)
        if record:
            all_records.append(record)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(clean_surrogates)

    df['birosag'] = df['birosag'].fillna('ISMERETLEN')
    
    # Oszlopok sorrendjének fixálása a konzisztencia érdekében
    standard_columns = [
        'doc_id', 'birosag', 'text', 'UgySzam', 'Ugyiratszam', 'MeghozoSzerv', 
        'MeghozoDatum', 'HatarozatKategoria', 'HatarozatJellege', 'Targyszavak', 'Jogszabalyhelyek',
        'AllKapcsolodoUgyszam', 'AllKapcsolodoBirosag'
    ]
    
    # Csak a létező oszlopokat tartjuk meg a standard listából
    final_columns = [col for col in standard_columns if col in df.columns]
    # Hozzáadjuk azokat az oszlopokat, amik a df-ben vannak, de a standard listában nem
    final_columns.extend([col for col in df.columns if col not in final_columns])
    
    return df[final_columns]


def process_single_document(base_filename: str, text_filepath: Path, json_files: dict) -> dict | None:
    """Egyetlen dokumentum és a hozzá tartozó JSON feldolgozása."""
    json_filepath = json_files.get(base_filename)
    
    text_content = extract_text_from_file(text_filepath)
    if not text_content:
        return None

    cleaned_text = clean_text_for_embedding(text_content)
    if len(cleaned_text) < config.CLEANING_MIN_TEXT_LENGTH:
        logging.debug(f"Dokumentum átugorva (túl rövid): {text_filepath.name}")
        return None

    metadata, related_cases = extract_metadata_from_json(json_filepath)
    
    # Bíróság nevének kinyerése a path-ból, ha máshogy nem elérhető
    birosag_from_path = None
    try:
        # feltételezzük, hogy a local_dir a 'raw_documents'
        relative_path_parts = text_filepath.relative_to(text_filepath.parent.parent).parts
        if len(relative_path_parts) > 1:
            birosag_from_path = relative_path_parts[0]
    except Exception:
        pass # Nem baj, ha ez nem sikerül, ez csak egy fallback

    record = {
        'doc_id': metadata.get('Azonosito', base_filename),
        'birosag': metadata.get('MeghozoBirosag', birosag_from_path),
        'text': cleaned_text,
        **metadata,
        **related_cases
    }

    # Felesleges vagy duplikált mezők eltávolítása
    for key in ['Szoveg', 'RezumeSzovegKornyezet', 'DownloadLink', 'metadata', 'KapcsolodoHatarozatok']:
        record.pop(key, None)
        
    return record


def extract_text_from_file(filepath: Path) -> str | None:
    """Szöveg kinyerése RTF vagy DOCX fájlból."""
    try:
        if filepath.suffix.lower() == '.rtf':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return rtf_to_text(f.read(), errors="ignore")
        elif filepath.suffix.lower() == '.docx':
            doc = Document(filepath)
            return ' \n'.join(para.text for para in doc.paragraphs if para.text.strip())
    except Exception as e:
        logging.warning(f"Hiba a szöveg kinyerésekor ({filepath.name}): {e}")
        return None


def extract_metadata_from_json(filepath: Path | None) -> tuple[dict, dict]:
    """Metaadatok kinyerése a JSON fájlból."""
    if not filepath or not filepath.exists():
        return {}, {}

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            metadata_list = json.load(f).get('List', [])
        
        if not metadata_list:
            return {}, {}
            
        metadata = metadata_list[0]
        related_ugyszam = []
        related_birosag = []

        # Kapcsolódó ügyek feldolgozása
        if 'KapcsolodoHatarozatok' in metadata and isinstance(metadata['KapcsolodoHatarozatok'], list):
            for case in metadata['KapcsolodoHatarozatok']:
                if isinstance(case, dict):
                    related_ugyszam.append(case.get('KapcsolodoUgyszam'))
                    related_birosag.append(case.get('KapcsolodoBirosag'))

        # A komplex listákat/dict-eket JSON stringgé alakítjuk a Parquet-kompatibilitásért
        if 'Jogszabalyhelyek' in metadata and not isinstance(metadata['Jogszabalyhelyek'], (str, int, float, bool, type(None))):
            metadata['Jogszabalyhelyek'] = json.dumps(metadata['Jogszabalyhelyek'], ensure_ascii=False)

        related_cases_data = {
            'AllKapcsolodoUgyszam': json.dumps(related_ugyszam, ensure_ascii=False) if related_ugyszam else None,
            'AllKapcsolodoBirosag': json.dumps(related_birosag, ensure_ascii=False) if related_birosag else None
        }

        return metadata, related_cases_data

    except (json.JSONDecodeError, IndexError) as e:
        logging.warning(f"Hiba a JSON metaadatok feldolgozásakor ({filepath.name}): {e}")
        return {}, {}


def main():
    """
    Fő függvény, amely beolvassa a nyers adatokat a lokális `raw` könyvtárból,
    feldolgozza őket, és az eredményt elmenti a `processed` könyvtárba.
    """
    logging.info("===== DOKUMENTUM ELŐFELDOLGOZÁS INDÍTÁSA =====")
    
    raw_dir = config.RAW_DATA_DIR
    output_parquet_path = config.CLEANED_DOCUMENTS_PARQUET
    
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        logging.error(f"A bemeneti '{raw_dir}' könyvtár nem létezik vagy üres.")
        logging.info("Kérlek, helyezd a feldolgozandó JSON és a hozzá tartozó RTF/DOCX fájlokat ebbe a könyvtárba.")
        sys.exit(1)

    processed_df = process_local_files(raw_dir)

    if not processed_df.empty:
        logging.info(f"Sikeresen feldolgozva {len(processed_df):,} dokumentum.")
        try:
            logging.info(f"Feldolgozott adatok mentése ide: {output_parquet_path}")
            processed_df.to_parquet(output_parquet_path, index=False, engine='pyarrow')
            logging.info("Sikeres mentés.")
        except Exception as e:
            logging.error(f"Hiba a Parquet fájl mentésekor: {e}", exc_info=True)
    else:
        logging.warning("Nem lett egyetlen dokumentum sem feldolgozva.")

    logging.info("===== DOKUMENTUM ELŐFELDOLGOZÁS BEFEJEZVE =====")


if __name__ == '__main__':
    main()