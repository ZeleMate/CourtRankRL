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
    from src.utils.azure_blob_storage import AzureBlobStorage
except ImportError as e:
    print(f"HIBA: Modul importálása sikertelen: {e}")
    sys.exit(1)

# Loggolás beállítása
logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

# Az Azure SDK túlzottan bőbeszédű naplózásának korlátozása WARNING szintre.
logging.getLogger("azure").setLevel(logging.WARNING)

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
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_surrogates(text):
    """Eltávolítja az érvénytelen surrogate párokat a stringből."""
    if not isinstance(text, str):
        return text
    return re.sub(r'[\ud800-\udfff]', '', text)

# Támogatott szövegfájl kiterjesztések
SUPPORTED_EXTENSIONS = tuple(ext.lower() for ext in config.SUPPORTED_TEXT_EXTENSIONS)

def main():
    """
    Letölti a nyers adatokat egy ideiglenes helyi könyvtárba,
    lokálisan feldolgozza őket, majd a tiszta Parquet fájlt feltölti az Azure-ba.
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    # Perzisztens helyi gyorsítótár a nyers adatoknak, hogy ne kelljen újra letölteni
    local_raw_data_dir = project_root / 'data_cache' / 'raw_documents'
    local_raw_data_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Helyi gyorsítótár használata a nyers adatokhoz: {local_raw_data_dir}")

    # A kimeneti Parquet fájlnak egy valóban ideiglenes könyvtárat használunk, ami törlődni fog
    output_processing_dir = tempfile.mkdtemp()
    logging.info(f"Ideiglenes kimeneti könyvtár létrehozva: {output_processing_dir}")

    total_records = 0
    success = False
    
    try:
        # 1. ===== ADATOK LETÖLTÉSE AZURE BLOB-BÓL LOKÁLIS GYORSÍTÓTÁRBA =====
        logging.info("Nyers adatok szinkronizálása a helyi gyorsítótárral...")
        all_blob_paths = blob_storage.list_blobs(path_prefix=config.BLOB_RAW_DATA_DIR)
        
        if not all_blob_paths:
            logging.warning(f"Nem találhatóak blobok a '{config.BLOB_RAW_DATA_DIR}/' prefix alatt. A szkript leáll.")
            return

        for blob_path in tqdm(all_blob_paths, desc="Fájlok szinkronizálása a helyi gyorsítótárba"):
            try:
                relative_blob_path = Path(*Path(blob_path).parts[1:])
                local_path = local_raw_data_dir / relative_blob_path

                # Csak akkor töltjük le, ha a fájl még nem létezik lokálisan
                if local_path.exists():
                    continue

                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                data = blob_storage.download_data(blob_path)
                with open(local_path, "wb") as f:
                    f.write(data)
            except Exception as e:
                logging.error(f"Hiba a(z) {blob_path} blob letöltésekor: {e}", exc_info=True)
        
        logging.info(f"Helyi gyorsítótár szinkronizálva. Összesen {len(all_blob_paths):,} fájl ellenőrizve.")

        # 2. ===== LOKÁLIS FÁJLOK FELDOLGOZÁSA A GYORSÍTÓTÁRBÓL =====
        logging.info("Helyi fájlok feldolgozásának megkezdése a gyorsítótárból...")
        
        document_files = {}
        json_files = {}

        all_local_files = [p for p in local_raw_data_dir.rglob('*') if p.is_file()]

        for file_path in all_local_files:
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                base_name = file_path.stem
                document_files[base_name] = file_path
            elif file_path.suffix.lower() == '.json':
                base_name = re.sub(r'\.(RTF|DOCX)_OBH$', '', file_path.stem, flags=re.IGNORECASE)
                json_files[base_name] = file_path

        all_records = []
        logging.info(f"Feldolgozásra váró dokumentum párok száma: {len(document_files):,}")

        for base_filename, text_filepath in tqdm(document_files.items(), desc="Dokumentumok feldolgozása"):
            json_filepath = json_files.get(base_filename)

            text_content = ""
            try:
                if text_filepath.suffix.lower() == '.rtf':
                    with open(text_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        rtf_content = f.read()
                    text_content = rtf_to_text(rtf_content, errors="ignore")
                elif text_filepath.suffix.lower() == '.docx':
                    doc = Document(text_filepath)
                    text_content = ' \\n'.join(para.text for para in doc.paragraphs if para.text.strip())
            except Exception as e:
                logging.warning(f"Nem sikerült kinyerni a szöveget a fájlból ({text_filepath}): {e}")
                continue
            
            cleaned_text_content = clean_text_for_embedding(text_content)

            if len(cleaned_text_content) < config.CLEANING_MIN_TEXT_LENGTH:
                logging.debug(f"Dokumentum átugorva, mert a tisztított szöveg túl rövid: {text_filepath.name}")
                continue

            extracted_metadata = {}
            all_related_ugyszam = []
            all_related_birosag = []

            if json_filepath:
                try:
                    with open(json_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        metadata_dict = json.load(f)
                    
                    if 'List' in metadata_dict and isinstance(metadata_dict['List'], list) and len(metadata_dict['List']) > 0:
                        extracted_metadata = metadata_dict['List'][0]
                        if 'KapcsolodoHatarozatok' in extracted_metadata and isinstance(extracted_metadata['KapcsolodoHatarozatok'], list):
                            for related_case in extracted_metadata['KapcsolodoHatarozatok']:
                                if isinstance(related_case, dict):
                                    all_related_ugyszam.append(related_case.get('KapcsolodoUgyszam'))
                                    all_related_birosag.append(related_case.get('KapcsolodoBirosag'))
                                else:
                                    logging.warning(f"A KapcsolodoHatarozatok lista egyik eleme nem szótár a {json_filepath} fájlban.")
                                    all_related_ugyszam.append(None)
                                    all_related_birosag.append(None)
                        if 'Jogszabalyhelyek' in extracted_metadata and not isinstance(extracted_metadata['Jogszabalyhelyek'], (str, int, float, bool)):
                            extracted_metadata['Jogszabalyhelyek'] = json.dumps(extracted_metadata['Jogszabalyhelyek'], ensure_ascii=False)
                        if 'KapcsolodoHatarozatok' in extracted_metadata and not isinstance(extracted_metadata['KapcsolodoHatarozatok'], (str, int, float, bool)):
                            extracted_metadata['KapcsolodoHatarozatok'] = json.dumps(extracted_metadata['KapcsolodoHatarozatok'], ensure_ascii=False)
                except json.JSONDecodeError:
                    logging.warning(f"Nem sikerült dekódolni a JSON fájlt: {json_filepath}")
                except Exception as e:
                    logging.warning(f"Hiba a JSON fájl feldolgozása közben ({json_filepath}): {e}")

            birosag_from_path = None
            try:
                relative_path_parts = text_filepath.relative_to(local_raw_data_dir).parts
                if len(relative_path_parts) > 1:
                     birosag_from_path = relative_path_parts[0]
            except Exception as e_path:
                 logging.warning(f"Váratlan hiba a bíróság nevének útvonalból történő kinyerése közben ({text_filepath}): {e_path}")

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
        
        total_records = len(all_records)

        # 3. ===== EGYESÍTETT, TISZTÍTOTT PARQUET LÉTREHOZÁSA ÉS FELTÖLTÉSE =====
        if not all_records:
            logging.warning("Nincs feldolgozható rekord, a Parquet fájl létrehozása és feltöltése átugorva.")
            return

        logging.info("Feldolgozás befejezve, egységes DataFrame létrehozása és feltöltése...")
        
        df = pd.DataFrame(all_records)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(clean_surrogates)

        df['birosag'] = df['birosag'].fillna('ISMERETLEN')

        expected_cols = [
            'doc_id', 'text', 'birosag', 'JogTerulet', 'Azonosito', 'MeghozoBirosag',
            'EgyediAzonosito', 'HatarozatEve', 'AllKapcsolodoUgyszam', 'AllKapcsolodoBirosag',
            'KapcsolodoHatarozatok', 'Jogszabalyhelyek'
        ]
        
        final_cols = [col for col in expected_cols if col in df.columns]
        other_cols = [col for col in df.columns if col not in final_cols]
        df = df[final_cols + other_cols]

        local_parquet_path = Path(output_processing_dir) / "cleaned_documents.parquet"
        df.to_parquet(
            path=local_parquet_path,
            engine='pyarrow',
            compression='snappy',
            index=False,
        )
        
        logging.info(f"Tisztított Parquet fájl sikeresen feltöltve ide: {local_parquet_path}")

        # 4. ===== PARQUET FÁJL FELTÖLTÉSE AZURE BLOB-BA =====
        logging.info(f"Parquet fájl feltöltése a(z) '{config.AZURE_CONTAINER_NAME}' konténerbe...")
        blob_path = config.BLOB_CLEANED_DOCUMENTS_PARQUET
        
        with open(local_parquet_path, "rb") as data:
            blob_storage.upload_data(data=data.read(), blob_path=blob_path)
        
        logging.info(f"Tisztított Parquet fájl sikeresen feltöltve ide: {blob_path} ({len(df):,} sor)")
        success = True

    except Exception as e:
        logging.error(f"Hiba történt a fő feldolgozási folyamatban: {e}", exc_info=True)
    finally:
        # 5. ===== IDEIGLENES KIMENETI KÖNYVTÁR TÖRLÉSE =====
        logging.info(f"Átmeneti kimeneti könyvtár törlése: {output_processing_dir}")
        shutil.rmtree(output_processing_dir, ignore_errors=True)
        # A nyers adatok gyorsítótára (`local_raw_data_dir`) megmarad a következő futtatáshoz.

        if success:
            print(f"\n✅ PREPROCESSING BEFEJEZVE!")
            print(f"📊 Feldolgozott rekordok: {total_records:,}")
            print(f"📄 Kimeneti blob: {config.AZURE_CONTAINER_NAME}/{config.BLOB_CLEANED_DOCUMENTS_PARQUET}")
        else:
            print(f"\n❌ PREPROCESSING SIKERTELEN!")
            print("Kérjük, ellenőrizze a logokat a hiba részleteiért.")


if __name__ == "__main__":
    main()