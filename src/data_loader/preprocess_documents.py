# Ez a szkript felelős a nyers dokumentumok (RTF, DOCX) és a hozzájuk tartozó
# JSON metaadatok feldolgozásáért, majd egyetlen CSV fájlba történő mentéséért.
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

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Konfiguráció és segédprogramok importálása
try:
    from configs import config
    from src.utils.azure_blob_storage import AzureBlobStorage
except ImportError as e:
    print(f"HIBA: Modul importálása sikertelen: {e}")
    sys.exit(1)

# Loggolás beállítása
logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

# Azure Blob Storage kliens inicializálása
try:
    blob_storage = AzureBlobStorage(container_name=config.AZURE_CONTAINER_NAME)
except ValueError as e:
    logging.error(e)
    sys.exit(1)

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

# Támogatott szövegfájl kiterjesztések
SUPPORTED_EXTENSIONS = tuple(ext.lower() for ext in config.SUPPORTED_TEXT_EXTENSIONS)

logging.info("Feldolgozás kezdése Azure Blob Storage-ból.")

# Blobs listázása a 'raw' prefixszel
try:
    all_blob_paths = blob_storage.list_blobs(path_prefix=config.BLOB_RAW_DATA_DIR)
    logging.info(f"Találva {len(all_blob_paths):,} blob a '{config.BLOB_RAW_DATA_DIR}/' prefix alatt.")
except Exception as e:
    logging.error(f"Hiba a blobok listázása közben: {e}")
    sys.exit(1)

# Szövegfájlok és a hozzájuk tartozó JSON-ok összepárosítása
document_blobs = {}
json_blobs = {}

for blob_path in all_blob_paths:
    path = Path(blob_path)
    if path.suffix.lower() in SUPPORTED_EXTENSIONS:
        # 'raw/filename.rtf' -> 'filename'
        base_name = path.stem
        document_blobs[base_name] = blob_path
    elif path.suffix.lower() == '.json':
        # 'raw/filename.RTF_OBH.JSON' -> 'filename'
        base_name = path.stem.replace('.RTF_OBH', '')
        json_blobs[base_name] = blob_path

# A feldolgozott rekordokat egyetlen listában gyűjtjük
all_records = []
total_records = 0

logging.info(f"Feldolgozásra váró dokumentum párok száma: {len(document_blobs)}")

for base_filename, text_blob_path in tqdm(document_blobs.items(), desc="Dokumentum blobok feldolgozása"):
    text_path = Path(text_blob_path)
    json_blob_path = json_blobs.get(base_filename)

    text_content = ""
    try:
        text_data = blob_storage.download_data(text_blob_path)
        if text_path.suffix.lower() == '.rtf':
            try:
                # Az rtf_to_text stringet vár, ezért dekódoljuk
                rtf_content = text_data.decode('utf-8', errors='ignore')
                text_content = rtf_to_text(rtf_content, errors="ignore")
            except Exception as e:
                logging.warning(f"Nem sikerült kinyerni a szöveget az RTF blobból ({text_blob_path}): {e}")
        elif text_path.suffix.lower() == '.docx':
            try:
                # A Document BytesIO-t vár
                doc = Document(io.BytesIO(text_data))
                text_content = ' \\n'.join(para.text for para in doc.paragraphs if para.text.strip())
            except Exception as e:
                logging.warning(f"Nem sikerült kinyerni a szöveget a DOCX blobból ({text_blob_path}): {e}")
    except Exception as e:
        logging.error(f"Hiba a blob letöltése közben ({text_blob_path}): {e}")
        continue
    
    # A kinyert nyers szöveg azonnali tisztítása
    cleaned_text_content = clean_text_for_embedding(text_content)

    # Csak akkor dolgozzuk fel a rekordot, ha a tisztítás után is maradt értékelhető szöveg
    if len(cleaned_text_content) < config.CLEANING_MIN_TEXT_LENGTH:
        logging.debug(f"Dokumentum átugorva, mert a tisztított szöveg túl rövid: {text_path.name}")
        continue

    extracted_metadata = {}
    all_related_ugyszam = []
    all_related_birosag = []

    if json_blob_path:
        try:
            json_data = blob_storage.download_data(json_blob_path)
            # A json.loads bytes-ot vagy string-et vár, a dekódolás biztonságosabb
            metadata_dict = json.loads(json_data.decode('utf-8', errors='ignore'))
            if 'List' in metadata_dict and isinstance(metadata_dict['List'], list) and len(metadata_dict['List']) > 0:
                extracted_metadata = metadata_dict['List'][0]
                if 'KapcsolodoHatarozatok' in extracted_metadata and isinstance(extracted_metadata['KapcsolodoHatarozatok'], list):
                    for related_case in extracted_metadata['KapcsolodoHatarozatok']:
                        if isinstance(related_case, dict):
                            all_related_ugyszam.append(related_case.get('KapcsolodoUgyszam'))
                            all_related_birosag.append(related_case.get('KapcsolodoBirosag'))
                        else:
                            logging.warning(f"A KapcsolodoHatarozatok lista egyik eleme nem szótár a {json_blob_path} blobban.")
                            all_related_ugyszam.append(None)
                            all_related_birosag.append(None)
                if 'Jogszabalyhelyek' in extracted_metadata and not isinstance(extracted_metadata['Jogszabalyhelyek'], (str, int, float, bool)):
                    extracted_metadata['Jogszabalyhelyek'] = json.dumps(extracted_metadata['Jogszabalyhelyek'], ensure_ascii=False)
                if 'KapcsolodoHatarozatok' in extracted_metadata and not isinstance(extracted_metadata['KapcsolodoHatarozatok'], (str, int, float, bool)):
                    extracted_metadata['KapcsolodoHatarozatok'] = json.dumps(extracted_metadata['KapcsolodoHatarozatok'], ensure_ascii=False)
        except json.JSONDecodeError:
            logging.warning(f"Nem sikerült dekódolni a JSON blobot: {json_blob_path}")
        except Exception as e:
            logging.warning(f"Hiba a JSON blob feldolgozása közben ({json_blob_path}): {e}")

    birosag_from_path = None
    try:
        # A 'raw/birosag_neve/aktaszam.rtf' formátumot feltételezzük
        path_parts = text_path.parts
        if len(path_parts) > 2 and path_parts[0] == config.BLOB_RAW_DATA_DIR:
             birosag_from_path = path_parts[1]
    except Exception as e_path:
         logging.warning(f"Váratlan hiba a bíróság nevének útvonalból történő kinyerése közben ({text_path}): {e_path}")

    record = {
        'text': cleaned_text_content,
        **extracted_metadata,
        'AllKapcsolodoUgyszam': json.dumps(all_related_ugyszam, ensure_ascii=False) if all_related_ugyszam else None,
        'AllKapcsolodoBirosag': json.dumps(all_related_birosag, ensure_ascii=False) if all_related_birosag else None,
    }
    record['doc_id'] = extracted_metadata.get('Azonosito', base_filename)
    record['birosag'] = extracted_metadata.get('MeghozoBirosag', birosag_from_path)
    
    record.pop('Szoveg', None)
    record.pop('RezumeSzovegKornyezet', None)
    record.pop('DownloadLink', None)
    record.pop('metadata', None)

    all_records.append(record)
    total_records += 1

# ===== EGYESÍTETT, TISZTÍTOTT PARQUET LÉTREHOZÁSA ÉS FELTÖLTÉSE AZURE BLOB-BA =====
logging.info("Feldolgozás befejezve, egységes DataFrame létrehozása és feltöltése...")

if all_records:
    try:
        df = pd.DataFrame(all_records)
        
        # A 'birosag' oszlop feltöltése, ha hiányzik (fontos a konzisztenciához)
        df['birosag'] = df['birosag'].fillna('ISMERETLEN')

        # Oszlopok sorrendjének biztosítása a jobb átláthatóságért
        expected_cols = [
            'doc_id', 'text', 'birosag', 'JogTerulet', 'Azonosito', 'MeghozoBirosag',
            'EgyediAzonosito', 'HatarozatEve', 'AllKapcsolodoUgyszam', 'AllKapcsolodoBirosag',
            'KapcsolodoHatarozatok', 'Jogszabalyhelyek'
        ]
        
        final_cols = [col for col in expected_cols if col in df.columns]
        other_cols = [col for col in df.columns if col not in final_cols]
        df = df[final_cols + other_cols]

        # DataFrame konvertálása Parquet formátumú bytes-objektummá
        parquet_buffer = io.BytesIO()
        df.to_parquet(
            path=parquet_buffer,
            engine='pyarrow',
            compression='snappy',
            index=False,
        )
        parquet_buffer.seek(0)

        # Feltöltés az Azure Blob Storage-ba
        blob_path = config.BLOB_CLEANED_DOCUMENTS_PARQUET
        blob_storage.upload_data(data=parquet_buffer.getvalue(), blob_path=blob_path)
        
        logging.info(f"Tisztított Parquet fájl sikeresen feltöltve ide: {blob_path} ({len(df):,} sor)")

    except Exception as e:
        logging.error(f"Hiba a Parquet fájl létrehozásában vagy feltöltésében: {e}", exc_info=True)

# ===== VÉGSŐ ÜZENETEK =====
print(f"\n✅ PREPROCESSING BEFEJEZVE!")
print(f"📊 Feldolgozott rekordok: {total_records:,}")
print(f"📄 Kimeneti blob: {config.AZURE_CONTAINER_NAME}/{config.BLOB_CLEANED_DOCUMENTS_PARQUET}")