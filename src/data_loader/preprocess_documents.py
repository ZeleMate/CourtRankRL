#!/usr/bin/env python3
"""
Dokumentum feldolgozás CourtRankRL számára teljes Docling használatával.
Docling-gel dolgozza fel a RTF és DOCX fájlokat, majd intelligens chunkolást végez.
"""

import json
import sys
import tqdm
import argparse
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
            src_tag = 'rtf' if isinstance(chunk, dict) else 'doc'
            chunk_data = {
                'chunk_id': f"{doc_id}_{src_tag}_{i}",
                'doc_id': doc_id,
                'text': final_text,
                'chunk_index': i,
                'metadata': metadata
            }
            f.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')

def get_processed_doc_ids(output_file: Path) -> set:
    """
    Már feldolgozott dokumentum azonosítók kinyerése a chunks fájlból.
    """
    processed_ids = set()

    if not output_file.exists():
        return processed_ids

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_ids.add(data['doc_id'])
                except:
                    continue
    except Exception as e:
        print(f"Figyelmeztetés: Nem sikerült beolvasni a meglévő chunks fájlt: {e}")

    return processed_ids

def process_all_documents(resume_mode: bool = False):
    """
    Összes dokumentum feldolgozása teljes Docling használatával.
    Resume funkcióval - folytatja ahol abbahagyta.
    """
    raw_dir = config.RAW_DATA_DIR
    output_file = config.CHUNKS_JSONL

    if resume_mode:
        print("=== RESUME MÓD - FOLYTATÁS AHOL ABBAHAGYTUK ===")
    else:
        print("=== TELJES ÚJRAFELDOLGOZÁS ===")

    print(f"Dokumentumok keresése: {str(raw_dir)}")

    # Összes támogatott fájl keresése rekurzívan
    supported_ext = config.SUPPORTED_TEXT_EXTENSIONS
    all_files = []
    for ext in supported_ext:
        all_files.extend(list(raw_dir.rglob(f"*{ext}")))

    print(f"Talált fájlok összesen: {len(all_files)}")

    # Már feldolgozott dokumentumok azonosítása (csak resume módban)
    processed_ids = set()
    if resume_mode:
        processed_ids = get_processed_doc_ids(output_file)
        print(f"Már feldolgozott dokumentumok: {len(processed_ids)}")

    # Szűrés - csak a feldolgozatlan fájlokat tartjuk meg
    files_to_process = []
    skipped_count = 0

    for filepath in all_files:
        doc_id = filepath.stem
        if resume_mode and doc_id in processed_ids:
            skipped_count += 1
        else:
            files_to_process.append(filepath)

    print(f"Feldolgozandó fájlok: {len(files_to_process)}")
    if resume_mode:
        print(f"Kihagyott fájlok (már feldolgozott): {skipped_count}")

    if not files_to_process:
        if resume_mode:
            print("Nincs feldolgozandó fájl - minden dokumentum már feldolgozva!")
        else:
            print("Nincs feldolgozandó fájl!")
        return

    # Statisztika a feldolgozandó fájlokról
    file_types = {}
    for filepath in files_to_process:
        ext = filepath.suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1

    print(f"Feldolgozandó fájl típusok: {file_types}")

    # Process files
    print(f"\nFeldolgozás megkezdése ({len(files_to_process)} fájl)...")

    if resume_mode:
        processed_count = len(processed_ids)  # Már feldolgozottakat is beleszámítjuk
    else:
        processed_count = 0

    total_chunks = 0
    error_count = 0

    for filepath in tqdm.tqdm(files_to_process, desc="Dokumentumok feldolgozása"):
        doc_id = filepath.stem

        try:
            # Dokumentum feldolgozása teljes Docling-gal
            doc_data = process_document(filepath)

            if doc_data is None:
                error_count += 1
                continue

            # Intelligens chunkolás Docling-gel
            chunks = chunk_document(doc_data)

            if not chunks:
                error_count += 1
                continue

            # Mentés gazdag metadata-val
            save_chunks_to_jsonl(chunks, doc_id, output_file)

            processed_count += 1
            total_chunks += len(chunks)

        except Exception as e:
            print(f"Hiba a(z) {filepath.name} feldolgozásakor: {e}")
            error_count += 1
            continue

    print("\n=== FELDOLGOZÁS EREDMÉNYE ===")
    if resume_mode:
        print(f"Összesen feldolgozott dokumentumok: {processed_count}")
        print(f"Újrafeldolgozott dokumentumok: {len(files_to_process)}")
        print(f"Kihagyott dokumentumok: {skipped_count}")
    else:
        print(f"Feldolgozott dokumentumok: {processed_count}")
    print(f"Hibás dokumentumok: {error_count}")
    print(f"Összes generált chunk: {total_chunks}")
    print(f"Eredmény fájl: {str(output_file)}")

    if error_count > 0:
        print(f"Figyelmeztetés: {error_count} dokumentum feldolgozása sikertelen volt!")

def main():
    """
    Fő függvény CLI argumentumokkal.
    """
    parser = argparse.ArgumentParser(description='CourtRankRL dokumentum feldolgozó')
    parser.add_argument('--resume', action='store_true',
                       help='Folytatás mód - csak a feldolgozatlan fájlokat dolgozza fel')
    parser.add_argument('--reset', action='store_true',
                       help='Teljes reset - törli a meglévő chunks fájlt és újrakezdi')
    parser.add_argument('--docx-only', action='store_true',
                       help='Csak DOCX fájlok feldolgozása')

    args = parser.parse_args()

    # Reset mód
    if args.reset:
        output_file = config.CHUNKS_JSONL
        if output_file.exists():
            print(f"Reset: {output_file} törlése...")
            output_file.unlink()
            print("Reset kész.")

    # DOCX-only mód
    if args.docx_only:
        process_docx_only()
        return

    # Normál vagy resume feldolgozás
    try:
        process_all_documents(resume_mode=args.resume)
    except UnicodeEncodeError as e:
        print(f"Unicode kódolási hiba: {e}")
        print("Próbálja meg UTF-8 kódolással futtatni a script-et")
    except Exception as e:
        print(f"Hiba történt: {e}")

def process_docx_only():
    """
    Csak DOCX fájlok feldolgozása.
    """
    print("=== DOCX ONLY MÓD ===")

    raw_dir = config.RAW_DATA_DIR
    output_file = config.CHUNKS_JSONL

    # Csak DOCX fájlok keresése
    docx_files = list(raw_dir.rglob('*.docx')) + list(raw_dir.rglob('*.DOCX'))
    print(f"Talált DOCX fájlok: {len(docx_files)}")

    if not docx_files:
        print("Nincs feldolgozandó DOCX fájl!")
        return

    # Már feldolgozott DOCX dokumentumok
    processed_ids = get_processed_doc_ids(output_file)
    docx_processed_ids = {doc_id for doc_id in processed_ids
                         if any(f.stem == doc_id for f in docx_files)}

    print(f"Már feldolgozott DOCX dokumentumok: {len(docx_processed_ids)}")

    # Szűrés
    files_to_process = []
    for filepath in docx_files:
        doc_id = filepath.stem
        if doc_id not in docx_processed_ids:
            files_to_process.append(filepath)

    print(f"Feldolgozandó DOCX fájlok: {len(files_to_process)}")

    if not files_to_process:
        print("Minden DOCX fájl már feldolgozva!")
        return

    # Feldolgozás
    processed_count = len(docx_processed_ids)
    total_chunks = 0
    error_count = 0

    for filepath in tqdm.tqdm(files_to_process, desc="DOCX feldolgozás"):
        doc_id = filepath.stem

        try:
            doc_data = process_document(filepath)
            if doc_data is None:
                error_count += 1
                continue

            chunks = chunk_document(doc_data)
            if not chunks:
                error_count += 1
                continue

            save_chunks_to_jsonl(chunks, doc_id, output_file)
            processed_count += 1
            total_chunks += len(chunks)

        except Exception as e:
            print(f"DOCX hiba ({filepath.name}): {e}")
            error_count += 1

    print("\n=== DOCX FELDOLGOZÁS EREDMÉNYE ===")
    print(f"Feldolgozott DOCX dokumentumok: {processed_count}")
    print(f"Generált chunkok: {total_chunks}")
    print(f"Hibás DOCX dokumentumok: {error_count}")

if __name__ == '__main__':
    main()