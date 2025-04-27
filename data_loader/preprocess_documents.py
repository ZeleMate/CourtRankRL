import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Calculate the project root directory
project_root = Path(__file__).resolve().parent.parent
# Add the project root to the Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Konfiguráció importálása
from configs import config

# Use the project's data directory instead of a hardcoded path
root_dir = project_root / 'data'
paths = list(root_dir.rglob('*'))
records = []

for path in tqdm(paths, desc="Fájlok feldolgozása"):
    if path.suffix.lower() in ('.rtf', '.docx'):
        base = path.stem # Kiterjesztés nélküli név, pl. '1400-P_20011_2022_60'
        text_path = path

        # ÚJ LOGIKA (kép alapján): JSON név = <base_name>.RTF_OBH.JSON
        json_filename = base + '.RTF_OBH.JSON' # Pl. '1400-P_20011_2022_60.RTF_OBH.JSON'
        json_path = path.with_name(json_filename)

        # Szöveg kinyerése egyszerűen
        if text_path.suffix.lower() == '.rtf':
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            text = re.sub(r'{\\[^}]+}', '', re.sub(r'[{}]', '', re.sub(r'\\[a-z]+\\s?', '', content)))
        else:
            from docx import Document
            doc = Document(str(text_path))
            text = ' '.join(para.text for para in doc.paragraphs)

        text = re.sub(r'\s+', ' ', text).strip()

        # Metaadat betöltés és specifikus mezők kinyerése
        extracted_metadata = {}
        # Initialize lists for all related case info
        all_related_ugyszam = []
        all_related_birosag = []

        # Csak akkor próbáljuk megnyitni, ha létezik a várt JSON fájl
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    metadata_dict = json.load(jf)

                    if 'List' in metadata_dict and isinstance(metadata_dict['List'], list) and len(metadata_dict['List']) > 0:
                        # Extract all key-value pairs from the first item in the List
                        extracted_metadata = metadata_dict['List'][0]

                        # Extract info from ALL related cases, if available
                        if 'KapcsolodoHatarozatok' in extracted_metadata and isinstance(extracted_metadata['KapcsolodoHatarozatok'], list):
                            for related_case in extracted_metadata['KapcsolodoHatarozatok']:
                                # Check if the item is a dictionary before accessing keys
                                if isinstance(related_case, dict):
                                    all_related_ugyszam.append(related_case.get('KapcsolodoUgyszam'))
                                    all_related_birosag.append(related_case.get('KapcsolodoBirosag'))
                                else:
                                    # Handle cases where items might not be dicts (optional logging/warning)
                                    print(f"Warning: Item in KapcsolodoHatarozatok is not a dictionary for {json_path}")
                                    all_related_ugyszam.append(None) # Add placeholder if needed
                                    all_related_birosag.append(None) # Add placeholder if needed

                        # Ensure Jogszabalyhelyek is handled correctly if it's complex (e.g., list or dict)
                        # For simplicity, converting to string if it's not already a simple type
                        if 'Jogszabalyhelyek' in extracted_metadata and not isinstance(extracted_metadata['Jogszabalyhelyek'], (str, int, float, bool)):
                            extracted_metadata['Jogszabalyhelyek'] = str(extracted_metadata['Jogszabalyhelyek'])
                        # Convert KapcsolodoHatarozatok to JSON string *after* extracting all ugyszam/birosag
                        if 'KapcsolodoHatarozatok' in extracted_metadata and not isinstance(extracted_metadata['KapcsolodoHatarozatok'], (str, int, float, bool)):
                            extracted_metadata['KapcsolodoHatarozatok'] = json.dumps(extracted_metadata['KapcsolodoHatarozatok'], ensure_ascii=False)  # Store as JSON string

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON for {json_path}")
            except Exception as e:
                print(f"Warning: Error processing JSON {json_path}: {e}")
        else:
             # Kiegészítő debug: Listázzuk ki a mappa tartalmát, hátha segít
             try:
                 parent_dir_contents = [f.name for f in path.parent.iterdir()]
             except Exception as list_e:
                 pass # Keep pass if the except block becomes empty

        # Kontextus kinyerése az elérési útból (fallback for birosag)
        birosag_from_path = None
        try:
            # Ensure root_dir is absolute for relative_to
            abs_root_dir = root_dir.resolve()
            abs_path = path.resolve()
            if abs_path.is_relative_to(abs_root_dir): # Check if path is under root_dir
                 rel_parts = abs_path.relative_to(abs_root_dir).parts
                 if len(rel_parts) > 1: # Need at least root/subdir/file structure
                    birosag_from_path = rel_parts[0]
            else:
                 print(f"Warning: Path {path} is not relative to root {root_dir}. Cannot determine birosag from path.")
        except ValueError as ve:
            # This might happen if path is not under root_dir, though is_relative_to should prevent it.
            print(f"Warning: ValueError calculating relative path for {path} from {root_dir}: {ve}")
            pass  # birosag_from_path remains None
        except Exception as e_path:
             print(f"Warning: Unexpected error getting birosag from path {path}: {e_path}")
             pass # birosag_from_path remains None

        # Rekord összeállítása
        record = {
            'text': text,
            # Add all extracted metadata fields (includes EgyediAzonosito, KapcsolodoHatarozatok as string, etc.)
            **extracted_metadata,
            # Add all related case info as JSON strings
            'AllKapcsolodoUgyszam': json.dumps(all_related_ugyszam, ensure_ascii=False) if all_related_ugyszam else None,
            'AllKapcsolodoBirosag': json.dumps(all_related_birosag, ensure_ascii=False) if all_related_birosag else None,
        }

        # Set doc_id: prioritize Azonosito from JSON, fallback to filename base
        record['doc_id'] = extracted_metadata.get('Azonosito', base)

        # Set birosag: prioritize MeghozoBirosag from JSON, fallback to path component
        record['birosag'] = extracted_metadata.get('MeghozoBirosag', birosag_from_path)

        # Remove potentially problematic fields if they exist but were not handled above
        record.pop('Szoveg', None)
        record.pop('RezumeSzovegKornyezet', None)
        record.pop('DownloadLink', None)  # Remove DownloadLink if present
        # Remove the original full metadata dict if it was accidentally added by **extracted_metadata
        record.pop('metadata', None)

        records.append(record)

df = pd.DataFrame(records)
# Ensure specific important columns exist, even if empty in some records
# Updated to include AllKapcsolodoUgyszam and AllKapcsolodoBirosag
for col in ['doc_id', 'text', 'birosag', 'JogTerulet', 'Azonosito', 'MeghozoBirosag', 'EgyediAzonosito', 'AllKapcsolodoUgyszam', 'AllKapcsolodoBirosag', 'KapcsolodoHatarozatok']:
    if col not in df.columns:
        df[col] = None

# Reorder columns for better readability (optional)
# Updated core_cols with new columns
core_cols = ['doc_id', 'text', 'birosag', 'JogTerulet', 'Azonosito', 'MeghozoBirosag', 'EgyediAzonosito', 'HatarozatEve', 'AllKapcsolodoUgyszam', 'AllKapcsolodoBirosag', 'KapcsolodoHatarozatok']
other_cols = [col for col in df.columns if col not in core_cols]
# Handle potential missing HatarozatEve if it wasn't in JSON
if 'HatarozatEve' not in df.columns:
    # Check if HatarozatEve exists in other_cols before removing from core_cols
    if 'HatarozatEve' in core_cols:
        core_cols.remove('HatarozatEve')
    if 'HatarozatEve' in other_cols:
        other_cols.remove('HatarozatEve') # Ensure it's not in others either

df = df[core_cols + other_cols]

out_path = config.RAW_CSV_DATA_PATH
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False, encoding=config.CSV_ENCODING)
print(f'Raw EDA adat elmentve: {out_path}')