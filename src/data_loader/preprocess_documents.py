# Ez a szkript felelős a nyers dokumentumok (RTF, DOCX) és a hozzájuk tartozó
# JSON metaadatok feldolgozásáért, majd egyetlen "nyers" CSV fájlba történő mentéséért.
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
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs import config

# Loggolás beállítása a központi konfigurációból
# Ennek a config importálása UTÁN kell következnie
logging.basicConfig(
    level=config.LOGGING_LEVEL,
    format=config.LOGGING_FORMAT,
    # force=True # Szükséges lehet, ha a root logger már konfigurálva van máshol (pl. notebookban)
               # vagy ha a szkriptet többször importáljuk/futtatjuk ugyanabban a sessionben.
               # Óvatosan használandó, mivel felülírja a meglévő beállításokat.
)

# Adat könyvtár elérési útja a konfigurációból
# FIGYELEM: A `root_dir` beállítása itt a projekt gyökeréhez képest relatív 'data' mappára mutat.
# Győződj meg róla, hogy a `config.DATA_DIR` (ha használni szeretnéd) megfelelően van beállítva,
# vagy ez a `project_root / 'data'` megfelel a célnak.
# Jelenleg a szkript a `project_root / 'data'`-t használja, nem a `config.DATA_DIR`-t.
root_dir_to_scan = project_root / 'data' # Ez a könyvtár lesz rekurzívan bejárva
paths = list(root_dir_to_scan.rglob('*')) # Az összes fájl és mappa lekérése
records = [] # Az összegyűjtött rekordok listája

# Támogatott szövegfájl kiterjesztések (a configból is jöhetne, ha ott definiálva van)
# Jelenleg a config.SUPPORTED_TEXT_EXTENSIONS = ['.docx', '.rtf'] van beállítva.
# Használjuk azt a konzisztencia érdekében.
SUPPORTED_EXTENSIONS = tuple(ext.lower() for ext in config.SUPPORTED_TEXT_EXTENSIONS)

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

        records.append(record)

df = pd.DataFrame(records)

# Biztosítjuk, hogy a fontos oszlopok létezzenek, még ha üresek is egyes rekordoknál
# Ezeknek az oszlopoknak összhangban kell lenniük a FINAL_OUTPUT_COLUMNS-zal a generate_embeddings.py-ban
# (kivéve az 'embedding' oszlopot, ami később kerül hozzáadásra)
expected_cols_for_raw_csv = [
    'doc_id', 'text', 'birosag', 'JogTerulet', 'Azonosito', 'MeghozoBirosag',
    'EgyediAzonosito', 'HatarozatEve', 'AllKapcsolodoUgyszam', 'AllKapcsolodoBirosag',
    'KapcsolodoHatarozatok', 'Jogszabalyhelyek' # Jogszabalyhelyek is fontos lehet
]
for col in expected_cols_for_raw_csv:
    if col not in df.columns:
        df[col] = None # Hozzáadás None értékekkel, ha hiányzik

# Oszlopok sorrendjének beállítása az olvashatóság érdekében (opcionális)
# Csak a létező oszlopokat próbáljuk meg átrendezni
final_ordered_cols = [col for col in expected_cols_for_raw_csv if col in df.columns]
other_cols = [col for col in df.columns if col not in final_ordered_cols]
df = df[final_ordered_cols + other_cols]

# Kimeneti CSV fájl mentése
out_path = config.RAW_CSV_DATA_PATH
out_path.parent.mkdir(parents=True, exist_ok=True) # Mappa létrehozása, ha nem létezik
df.to_csv(out_path, index=False, encoding=config.CSV_ENCODING, errors='replace')

logging.info(f"Nyers EDA adatok feldolgozása befejezve. Mentve ide: {out_path}") # print helyett logging
# A print itt is maradhat, ha a felhasználói visszajelzés a cél a szkript végén.
print(f'A feldolgozott nyers adatokat tartalmazó CSV fájl mentve: {out_path}')