#!/usr/bin/env python3
"""
Dokumentum feldolgozás CourtRankRL számára teljes Docling használatával.
Docling-gel dolgozza fel a RTF és DOCX fájlokat, majd intelligens chunkolást végez.
"""

import json
import sys
import tqdm
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.types.doc.document import DoclingDocument
from typing import Union

# UTF-8 kódolás beállítása
sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
sys.stderr.reconfigure(encoding='utf-8')  # type: ignore

def clean_unicode_text(text: str) -> str:
    """
    Unicode szöveg tisztítása problémás karakterek eltávolításával.
    """
    if not text:
        return ""

    try:
        # Először próbáljuk meg normális UTF-8 kódolással
        text.encode('utf-8')
        return text
    except UnicodeEncodeError:
        # Ha vannak problémás karakterek, tisztítjuk őket
        import unicodedata

        # 1. lépés: surrogates eltávolítása
        text = text.encode('utf-8', errors='replace').decode('utf-8')

        # 2. lépés: egyéb problémás karakterek kezelése
        cleaned_chars = []
        for char in text:
            try:
                char.encode('utf-8')
                cleaned_chars.append(char)
            except UnicodeEncodeError:
                # Helyettesítjük a problémás karaktert
                cleaned_chars.append('?')

        return ''.join(cleaned_chars)

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config

def process_document(filepath: Path):
    """
    Dokumentum feldolgozása hibrid módszerrel.
    RTF fájlokhoz szöveg kinyerés + DoclingDocument konverzió,
    DOCX fájlokhoz teljes Docling feldolgozás.
    Visszaadja a DoclingDocument objektumot a strukturált adatokért.
    """
    try:
        if filepath.suffix.lower() == '.rtf':
            # RTF fájlokhoz: szöveg kinyerés majd DoclingDocument létrehozása
            from striprtf.striprtf import rtf_to_text
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_content = f.read()
            text = rtf_to_text(rtf_content)

            if not text.strip():
                print(f"Üres RTF fájl: {filepath.name}")
                return None

            # Szöveg tisztítása Unicode problémák elkerülésére
            cleaned_text = clean_unicode_text(text.strip())

            # Egyszerűbb megoldás: csak a szöveget adjuk vissza, majd később konvertáljuk
            # DoclingDocument formátumba a chunkolás előtt
            return {
                'text': cleaned_text,
                'filepath': str(filepath),
                'filename': filepath.name,
                'is_rtf': True
            }


        elif filepath.suffix.lower() == '.docx':
            # DOCX fájlokhoz teljes Docling feldolgozás
            converter = DocumentConverter()
            result = converter.convert(str(filepath))

            if result.document is None:
                print(f"Sikertelen DOCX feldolgozás: {filepath.name}")
                return None

            return result.document

        else:
            print(f"Nem támogatott fájltípus: {filepath.suffix}")
            return None

    except Exception as e:
        print(f"Hiba a feldolgozás során ({filepath.name}): {e}")
        return None

def chunk_document(doc_data: Union[DoclingDocument, dict]) -> list:
    """
    Dokumentum chunkokra bontása.
    DoclingDocument esetén HierarchicalChunker,
    RTF szöveg esetén egyszerű szöveg chunkolás.
    """
    try:
        if isinstance(doc_data, dict) and doc_data.get('is_rtf'):
            # RTF szöveg egyszerű chunkolása
            text = doc_data['text']
            # Egyszerű szöveg chunkolás (karakter alapú)
            chunk_size = 8000  # Konfigurálható lenne
            chunk_overlap = 200

            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk_text = text[i:i + chunk_size]
                if chunk_text.strip():  # Csak nem üres chunkok
                    # Chunk szöveg tisztítása
                    cleaned_chunk_text = clean_unicode_text(chunk_text)
                    chunks.append({
                        'text': cleaned_chunk_text,
                        'chunk_type': 'rtf_text',
                        'page_numbers': set(),
                        'headings': [],
                        'captions': []
                    })

            return chunks

        elif isinstance(doc_data, DoclingDocument):  # DoclingDocument objektum
            # DoclingDocument esetén HierarchicalChunker használata
            chunker = HierarchicalChunker()
            chunks = list(chunker.chunk(doc_data))
            return chunks
        else:
            print(f"Ismeretlen dokumentum formátum: {type(doc_data)}")
            return []

    except Exception as e:
        print(f"Hiba a chunkolás során: {e}")
        return []

def save_chunks_to_jsonl(chunks: list, doc_id: str, output_file: Path):
    """
    Chunkok mentése JSONL formátumban DoclingDocument vagy RTF metadata-val.
    """
    # Chunker példány létrehozása Docling chunkokhoz
    chunker = HierarchicalChunker()

    with open(output_file, 'a', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                # RTF chunk (egyszerű dictionary)
                chunk_text = chunk['text']
                cleaned_chunk_text = clean_unicode_text(chunk_text)
                metadata = {
                    'chunk_type': chunk.get('chunk_type', 'rtf_text'),
                    'hierarchy_level': 0,
                    'page_numbers': list(chunk.get('page_numbers', set())),
                    'headings': chunk.get('headings', []),
                    'captions': chunk.get('captions', []),
                    'content_type': 'RTF_TextChunk',
                }
            else:
                # Docling chunk objektum
                chunk_text = chunker.contextualize(chunk)
                # Szöveg tisztítása Unicode problémák elkerülésére
                cleaned_chunk_text = clean_unicode_text(chunk_text)

                # Gazdag DoclingDocument alapú metadata gyűjtése
                metadata = {
                    'chunk_type': getattr(chunk, 'chunk_type', 'unknown'),
                    'hierarchy_level': getattr(chunk, 'hierarchy_level', 0),
                    'page_numbers': list(getattr(chunk, 'page_numbers', set())),
                    'headings': [{'text': h.text, 'level': getattr(h, 'level', 0)}
                               for h in getattr(chunk, 'headings', [])],
                    'captions': [c.text for c in getattr(chunk, 'captions', [])],
                    'content_type': type(chunk).__name__,
                }

            # Chunk adatok struktúrája (tisztított szöveggel)
            final_text = cleaned_chunk_text
            chunk_data = {
                'chunk_id': f"{doc_id}_{i}",
                'doc_id': doc_id,
                'text': final_text,
                'chunk_index': i,
                'metadata': metadata
            }
            f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')

def process_all_documents():
    """
    Összes dokumentum feldolgozása teljes Docling használatával.
    """
    raw_dir = config.RAW_DATA_DIR
    output_file = config.CHUNKS_JSONL

    # Kimeneti fájl törlése, ha létezik
    if output_file.exists():
        output_file.unlink()

    print(f"Dokumentumok keresése: {str(raw_dir)}")

    # Összes támogatott fájl keresése rekurzívan
    supported_ext = config.SUPPORTED_TEXT_EXTENSIONS
    files = []
    for ext in supported_ext:
        files.extend(list(raw_dir.rglob(f"*{ext}")))

    print(f"Talált fájlok: {len(files)}")

    # Process all files
    print(f"Processing all {len(files)} files...")

    processed_count = 0
    total_chunks = 0

    for filepath in tqdm.tqdm(files, desc="Dokumentumok feldolgozása"):
        doc_id = filepath.stem

        print(f"Feldolgozás: {filepath.name}")

        # Dokumentum feldolgozása teljes Docling-gal
        doc_data = process_document(filepath)

        if doc_data is None:
            continue

        # Intelligens chunkolás Docling-gel
        chunks = chunk_document(doc_data)

        if not chunks:
            print(f"  Nincs chunk: {filepath.name}")
            continue

        # Mentés gazdag metadata-val
        save_chunks_to_jsonl(chunks, doc_id, output_file)

        processed_count += 1
        total_chunks += len(chunks)
        print(f"  Elkészült: {len(chunks)} chunk")

    print(f"\\nÖsszesen feldolgozott dokumentumok: {processed_count}")
    print(f"Összes chunk: {total_chunks}")
    print(f"Eredmény: {str(output_file)}")

if __name__ == '__main__':
    try:
        process_all_documents()
    except UnicodeEncodeError as e:
        print(f"Unicode kódolási hiba: {e}")
        print("Próbálja meg UTF-8 kódolással futtatni a script-et")
    except Exception as e:
        print(f"Hiba történt: {e}")