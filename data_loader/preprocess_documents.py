import pandas as pd
import json
import re
from pathlib import Path
from tqdm import tqdm

root_dir = Path('/Users/zelenyianszkimate/Documents/Szakdolgozat/BHGY-k')
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
out_path = Path('/Users/zelenyianszkimate/Documents/Szakdolgozat/processed_data/raw_data_for_eda.csv')
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False, encoding='utf-8')
print(f'Raw EDA adat elmentve: {out_path}')