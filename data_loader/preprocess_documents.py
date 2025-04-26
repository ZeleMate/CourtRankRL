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

# Use the DATA_DIR from the config file
root_dir = config.DATA_DIR
# root_dir = project_root / 'data' # Original calculation
# root_dir = Path('/Users/zelenyianszkimate/Documents/Szakdolgozat/BHGY-k') # Original hardcoded path
paths = list(root_dir.rglob('*'))
records = []

for path in tqdm(paths, desc="Fájlok feldolgozása"):
    if path.suffix.lower() in ('.rtf', '.docx'):
        base = path.stem
        text_path = path
        json_path = path.with_suffix('.json')

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

        # Metaadat betöltés
        metadata = {}
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    metadata = json.load(jf)
            except:
                pass

        # Kontextus kinyerése az elérési útból
        try:
            rel_parts = path.relative_to(root_dir).parts
            birosag = rel_parts[0] if len(rel_parts) > 0 else None
            jogterulet = rel_parts[1] if len(rel_parts) > 1 else None
        except ValueError:
            birosag = None
            jogterulet = None

        record = {
            'doc_id': base,
            'text': text,
            'birosag': birosag,
            'jogterulet': jogterulet,
            'metadata': json.dumps(metadata, ensure_ascii=False)
        }
        records.append(record)

df = pd.DataFrame(records)
out_path = config.RAW_CSV_DATA_PATH
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False, encoding=config.CSV_ENCODING)
print(f'Raw EDA adat elmentve: {out_path}')