#!/usr/bin/env python3
"""CourtRankRL DOCX feldolgozó – Docling + JSONL output, megbízható --resume támogatással."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import tqdm

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from configs import config
except ImportError:  # pragma: no cover
    import configs.config as config

_converter: Optional[DocumentConverter] = None


def get_document_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def minimal_normalize_text(text: str) -> str:
    if not text:
        return ""
    try:
        import re
        import unicodedata

        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = unicodedata.normalize("NFC", text)
        return text.strip()
    except Exception:
        return text


def extract_metadata_from_path(filepath: Path) -> Dict[str, str]:
    """
    Metaadat kinyerése a fájl útvonalából.
    
    Könyvtárstruktúra: /data/raw/[Bíróság]/[jogterület]/[case_id]/[fájlnév].DOCX
    Az évszámot NEM a könyvtárból, hanem az ügyszámból nyerjük ki a regex alapú feldolgozással.
    """
    parts = filepath.parts
    court = domain = case_id = ""

    if len(parts) >= 4:
        court = parts[-4]
        domain = parts[-3]
        # parts[-2] a case_id könyvtár, NEM az év!
        # Az évszámot később az ügyszámból nyerjük ki
        case_id = Path(parts[-1]).stem

    if not all([court, domain]):
        stem = filepath.stem
        tokens = stem.split('_')
        if len(tokens) >= 2:
            court, domain = tokens[:2]
            case_id = '_'.join(tokens)
        else:
            case_id = stem

    return {
        'court': court.strip(),
        'domain': domain.strip(),
        'year': '',  # Év üresen marad, később az ügyszámból nyerjük ki
        'doc_id': case_id,
    }


def extract_metadata_from_docling_text(text: str, max_lines: int = 30, check_end_for_date: bool = True) -> Dict[str, str]:
    """
    Metadata kinyerése a dokumentum szövegéből.
    
    Args:
        text: A dokumentum szövege
        max_lines: Maximum hány sort vizsgáljon az elején (default: 30)
        check_end_for_date: Ha True, a dokumentum végéről is keres dátumot (default: True)
    """
    import re

    court = ""
    case_id = ""
    year = ""
    lines = text.split('\n')
    header_lines = lines[:max_lines]

    court_patterns = [
        r'([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+)*\s*(?:[Íí]télőtábla|[Tt]örvényszék|[Jj]árásbíróság|[Kk]özigazgatási\s+[és\s+]*[Mm]unkaügyi\s+[Bb]íróság))',
        r'([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+\s*[Bb]íróság)',
    ]
    case_patterns = [
        # === 1. TÖRVÉNYSZÉKI FORMÁTUMOK (pl. 5.G.20.033/2023) ===
        r'(\d+\.[A-Z]{1,3}\.\d+\.\d+/\d{4}\.?\s+szám)',
        r'(\d+\.[A-Z]{1,3}\.\d+\.\d+/\d{4})',
        
        # === 2. JÁRÁSBÍRÓSÁGI FORMÁTUMOK (pl. 14.B.105/2021/136) ===
        r'(\d+\.[A-Z]{1,3}\.\d+/\d{4}/\d+\.?\s+szám)',
        r'(\d+\.[A-Z]{1,3}\.\d+/\d{4}/\d+)',
        
        # === 3. TÖBBBETŰS KÓDOK (pl. 14.BPK.694/2020/4, 16.FK.43/2016/12) ===
        r'(\d+\.[A-Z][a-z]{1,3}\.\d+/\d{4}/\d+\.?\s+szám)',
        r'(\d+\.[A-Z][a-z]{1,3}\.\d+/\d{4}/\d+)',
        
        # === 4. RÓMAI SZÁMOS FORMÁTUMOK (pl. Bf.II.123/2020/5) ===
        r'(Bf\.[IVXLCDM]+\.\d+/\d{4}/\d+\.?\s+szám)',
        r'(Bf\.[IVXLCDM]+\.\d+/\d{4}/\d+)',
        
        # === 5. P-BETŰS FORMÁTUMOK ===
        # P kezdetű egyszerű (pl. P.20.126/2023/5)
        r'(P\.\d+\.\d+/\d{4}/\d+\.?\s+szám)',
        r'(P\.\d+\.\d+/\d{4}/\d+)',
        # Szám + P kezdetű (pl. 6.P.20.126/2023/5)
        r'(\d+\.P\.\d+\.\d+/\d{4}/\d+\.?\s+szám)',
        r'(\d+\.P\.\d+\.\d+/\d{4}/\d+)',
        
        # === 6. FELLEBBVITELI FORMÁTUMOK (Kfv, Pfv, Kfv) ===
        r'([KPk][fF][vV][\._]\d+[\._]\d+/\d{4}/\d+\.?\s+szám)',
        r'([KPk][fF][vV][\._]\d+[\._]\d+/\d{4}/\d+)',
        
        # === 7. SPECIÁLIS KÓDOK (Are, Fk, Bk, stb.) ===
        r'([A-Z][a-z]{1,2}[\._]\d+[\._]\d+/\d{4}/\d+\.?\s+szám)',
        r'([A-Z][a-z]{1,2}[\._]\d+[\._]\d+/\d{4}/\d+)',
        
        # === 8. RÉGEBBI/ÁLTALÁNOS FORMÁTUMOK (fallback) ===
        # Egyetlen nagybetű kód (régi stílus)
        r'(\d+\.[A-Z]\.\d+/\d{4}/\d+\.?\s+szám)',
        r'(\d+\.[A-Z]\.\d+/\d{4}/\d+)',
        r'([A-Z]\.\d+\.\d+/\d{4}/\d+\.?\s+szám)',
        r'([A-Z]\.\d+\.\d+/\d{4}/\d+)',
    ]

    # Bíróság és ügyszám keresése az első max_lines sorból
    for line in header_lines:
        candidate = line.strip()
        if not candidate:
            continue
        if not court:
            for pattern in court_patterns:
                match = re.search(pattern, candidate, re.IGNORECASE)
                if match:
                    court = match.group(1).strip()
                    break
        if not case_id:
            for pattern in case_patterns:
                match = re.search(pattern, candidate)
                if match:
                    case_id = match.group(1).strip()
                    break
        if court and case_id:
            break

    # Év kinyerése: először a dokumentum VÉGÉRŐL keressük a határozat dátumát
    # Formátum: "Helység, ÉÉÉÉ. hónap nap." (pl. "Sárvár, 2020. március 2.")
    if check_end_for_date and len(lines) > 10:
        # Utolsó 50 sor vizsgálata (ahol a dátum általában van)
        end_lines = lines[-50:]
        date_patterns = [
            r'[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+,\s*(\d{4})\.\s+(?:január|február|március|április|május|június|július|augusztus|szeptember|október|november|december)',
            r'\b(\d{4})\.\s+(?:január|február|március|április|május|június|július|augusztus|szeptember|október|november|december)\s+\d{1,2}\.',
            r',\s*(\d{4})\.\s+\w+\s+\d{1,2}\.',  # Általános: ", 2020. szó szám."
        ]
        
        for line in reversed(end_lines):
            candidate = line.strip()
            if not candidate:
                continue
            for pattern in date_patterns:
                match = re.search(pattern, candidate, re.IGNORECASE)
                if match:
                    extracted_year = match.group(1)
                    # Validálás: 1990-2099 között legyen az év
                    if 1990 <= int(extracted_year) <= 2099:
                        year = extracted_year
                        break
            if year:
                break

    # Ha nem találtunk évet a végén, próbáljuk az ügyszámból
    if not year and case_id:
        year_match = re.search(r'/(\d{4})/', case_id)
        if year_match:
            year = year_match.group(1)

    # Domain kinyerése az ügyszámból ha lehetséges
    domain = ""
    if case_id:
        domain_match = re.match(r'^(\d+[-_.])?([A-Za-z]+)', case_id)
        if domain_match:
            domain = domain_match.group(2).upper()
    
    return {'court': court, 'doc_id': case_id, 'year': year, 'domain': domain}


def iter_docling_chunks(docling_document: DoclingDocument) -> Iterator[Dict[str, object]]:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

    chunker = HybridChunker()
    for chunk in chunker.chunk(docling_document):
        chunk_text = getattr(chunk, 'text', None)
        if chunk_text and str(chunk_text).strip():
            yield {'text': chunk_text}


def normalize_raw_path(path: Path | str, root: Path) -> str:
    candidate = Path(path)
    try:
        rel = candidate.relative_to(root)
    except ValueError:
        rel = candidate
    return rel.as_posix()


def list_raw_documents(raw_root: Path) -> List[Path]:
    suffixes = {ext.lower() for ext in getattr(config, 'SUPPORTED_TEXT_EXTENSIONS', ['.docx'])}
    collected: List[Path] = []
    for path in raw_root.rglob('*'):
        if path.is_file() and path.suffix.lower() in suffixes:
            collected.append(path)
    collected.sort(key=lambda p: normalize_raw_path(p, raw_root))
    return collected


def load_processed_paths(manifest_path: Path, raw_root: Path, chunks_file: Path) -> Set[str]:
    """
    Betölti a már feldolgozott fájlok útvonalait manifest és chunks fájlból.
    
    Optimalizálva pandas.read_json() használatával (agents.md szerint) - 10-30x gyorsabb
    mint kézi json.loads() parsing nagy fájloknál.
    """
    import pandas as pd
    
    processed_paths: Set[str] = set()
    legacy_doc_ids: Set[str] = set()

    # Manifest parsing pandas-szal
    if manifest_path.exists():
        try:
            df_manifest = pd.read_json(manifest_path, lines=True, encoding='utf-8')
            
            # Raw path extraction
            for _, entry in df_manifest.iterrows():
                raw_path = entry.get('raw_path') or entry.get('source_path')
                if raw_path:
                    processed_paths.add(normalize_raw_path(raw_path, raw_root))
                    continue
                doc_id = entry.get('doc_id')
                if doc_id:
                    legacy_doc_ids.add(doc_id)
        except (ValueError, FileNotFoundError):
            # Fallback: ha pandas parsing sikertelen, marad az eredeti set üres
            pass

    # Chunks parsing pandas-szal (csak ha szükséges)
    if legacy_doc_ids and chunks_file.exists():
        try:
            # Chunked reading memória-hatékonyság miatt (2.9M sor esetén)
            for chunk_df in pd.read_json(chunks_file, lines=True, encoding='utf-8', chunksize=50000):
                for _, chunk in chunk_df.iterrows():
                    doc_id = chunk.get('doc_id')
                    if doc_id in legacy_doc_ids:
                        source_path = chunk.get('source_path')
                        if source_path:
                            processed_paths.add(normalize_raw_path(source_path, raw_root))
                            legacy_doc_ids.discard(doc_id)
                    if not legacy_doc_ids:
                        break
                if not legacy_doc_ids:
                    break
        except (ValueError, FileNotFoundError):
            # Fallback: ha pandas parsing sikertelen, marad az eredeti set
            pass

    return processed_paths


def append_processed_entry(manifest_path: Path, raw_rel_path: str, doc_id: str, chunk_count: int) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'raw_path': raw_rel_path,
        'doc_id': doc_id,
        'chunk_count': int(chunk_count),
        'timestamp': int(time.time()),
    }
    with open(manifest_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


def process_document_with_docling(filepath: Path) -> Optional[DoclingDocument]:
    converter = get_document_converter()
    result = converter.convert(str(filepath))
    return result.document


def process_single_file_worker(filepath_str: str, raw_root_str: str, tmp_dir_str: str, batch_size: int) -> Optional[Dict[str, object]]:
    filepath = Path(filepath_str)
    raw_root = Path(raw_root_str)
    tmp_dir = Path(tmp_dir_str)

    try:
        docling_doc = process_document_with_docling(filepath)
        if docling_doc is None:
            print(f"Figyelmeztetés: Docling nem tudta feldolgozni: {filepath.name}")
            return None

        path_meta = extract_metadata_from_path(filepath)
        doc_id = path_meta.get('doc_id') or filepath.stem
        doc_metadata = {
            'court': (path_meta.get('court') or '').strip(),
            'domain': (path_meta.get('domain') or '').strip(),
            'year': (path_meta.get('year') or '').strip(),
            'source_path': str(filepath),
        }

        # Először gyűjtsük össze az ÖSSZES chunk-ot egy listába
        all_chunks: List[Dict[str, object]] = list(iter_docling_chunks(docling_doc))
        
        if not all_chunks:
            print(f"Figyelmeztetés: Nincs chunk a dokumentumból: {filepath.name}")
            return None

        # Metaadat kinyerés: első és utolsó chunk-ok szövegéből
        # Első 2000 karakter az ügyszámhoz, bírósághoz
        # Utolsó 2000 karakter a határozat dátumához (év kinyerése)
        first_text_parts: List[str] = []
        last_text_parts: List[str] = []
        
        char_count = 0
        for chunk in all_chunks:
            text = chunk.get('text', '')
            if text and char_count < 2000:
                first_text_parts.append(str(text))
                char_count += len(str(text))
        
        # Utolsó 2000 karakter gyűjtése (visszafelé haladva)
        char_count = 0
        for chunk in reversed(all_chunks):
            text = chunk.get('text', '')
            if text and char_count < 2000:
                last_text_parts.insert(0, str(text))
                char_count += len(str(text))

        if first_text_parts or last_text_parts:
            # Teljes szöveg összeállítása metaadat kinyeréshez
            full_preview = minimal_normalize_text(
                '\n'.join(first_text_parts) + '\n[...]\n' + '\n'.join(last_text_parts)
            )
            docling_meta = extract_metadata_from_docling_text(full_preview)
            if docling_meta.get('court'):
                doc_metadata['court'] = docling_meta['court']
            # Priorizáljuk a szövegből kinyert ügyszámot
            if docling_meta.get('doc_id') and docling_meta['doc_id'].strip():
                doc_id = docling_meta['doc_id']
                # Ügyszámból és dátumból kinyert year és domain felülírja a path-ból nyertet
                if docling_meta.get('year'):
                    doc_metadata['year'] = docling_meta['year']
                docling_domain = (docling_meta.get('domain') or '').strip()
                if (
                    docling_domain
                    and len(docling_domain) >= 3
                    and not docling_domain.isupper()
                    and not doc_metadata.get('domain')
                ):
                    doc_metadata['domain'] = docling_domain

        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"tmp_{filepath.stem}_{mp.current_process().pid}.jsonl"
        batch_buffer: List[str] = []
        chunk_count = 0

        def create_chunk_payload(chunk: Dict[str, object], index: int) -> Dict[str, object]:
            """Helper to create consistent chunk payload."""
            return {
                'chunk_id': f"{doc_id}_{index}",
                'doc_id': doc_id,
                'text': chunk['text'],
                'court': doc_metadata.get('court', ''),
                'domain': doc_metadata.get('domain', ''),
                'year': doc_metadata.get('year', ''),
                'source_path': str(filepath),
            }

        with open(tmp_path, 'w', encoding='utf-8') as handle:
            def flush_buffer() -> None:
                if batch_buffer:
                    handle.write(''.join(batch_buffer))
                    batch_buffer.clear()

            # Process all chunks
            for index, chunk in enumerate(all_chunks):
                payload = create_chunk_payload(chunk, index)
                batch_buffer.append(json.dumps(payload, ensure_ascii=False) + '\n')
                chunk_count += 1
                if len(batch_buffer) >= batch_size:
                    flush_buffer()

            flush_buffer()

        try:
            del docling_doc
            del all_chunks
            del first_text_parts
            del last_text_parts
        except Exception:
            pass
        gc.collect()

        return {
            'doc_id': doc_id,
            'chunk_count': chunk_count,
            'tmp_file': str(tmp_path),
            'raw_rel_path': normalize_raw_path(filepath, raw_root),
        }

    except Exception as e:
        print(f"Hiba a {filepath.name} feldolgozása során: {type(e).__name__}: {str(e)[:100]}")
        return None


def append_temp_to_output(tmp_file: Path, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'a', encoding='utf-8') as out_handle, open(tmp_file, 'r', encoding='utf-8') as in_handle:
        for line in in_handle:
            out_handle.write(line)


def process_all_docx_parallel(output_file: Path, resume_mode: bool = False, num_workers: Optional[int] = None) -> None:
    raw_root = config.RAW_DATA_DIR
    print("=== DOCX FELDOLGOZÁS INDÍTÁSA ===")
    print(f"Forráskönyvtár: {raw_root}")

    raw_files = list_raw_documents(raw_root)
    print(f"Talált DOCX fájlok: {len(raw_files):,}")

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    print(f"Használt worker folyamatok: {num_workers}")

    processed_paths: Set[str] = set()
    if resume_mode:
        processed_paths = load_processed_paths(config.PROCESSED_DOCS_LIST, raw_root, output_file)
        print(f"Már feldolgozott fájlok: {len(processed_paths):,}")

    pending: List[Tuple[Path, str]] = []
    for path in raw_files:
        rel_path = normalize_raw_path(path, raw_root)
        if resume_mode and rel_path in processed_paths:
            continue
        pending.append((path, rel_path))

    if not pending:
        print("Nincs feldolgozandó fájl. A futás sikeresen befejeződött.")
        return

    tmp_dir = output_file.parent / '.tmp_chunks'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    processed_count = len(processed_paths)
    total_chunks = 0
    error_count = 0

    iterator = iter(pending)
    futures: Dict[Future, Tuple[Path, str]] = {}
    max_in_flight = max(1, num_workers * 2)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        def submit_next() -> None:
            try:
                next_path, rel = next(iterator)
            except StopIteration:
                return
            future = executor.submit(
                process_single_file_worker,
                str(next_path),
                str(raw_root),
                str(tmp_dir),
                int(getattr(config, 'CHUNK_WRITE_BATCH_SIZE', 200)),
            )
            futures[future] = (next_path, rel)

        for _ in range(min(max_in_flight, len(pending))):
            submit_next()

        pbar = tqdm.tqdm(total=len(pending), desc="Dokumentumok feldolgozása", unit="dokumentum")
        try:
            while futures:
                ready = as_completed(list(futures.keys()), timeout=None)
                for future in ready:
                    pending_path, rel_path = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception:
                        result = None

                    if result is None:
                        error_count += 1
                        print(f"Figyelmeztetés: feldolgozási hiba, kimarad: {pending_path}")
                    else:
                        tmp_path = Path(result['tmp_file'])
                        append_temp_to_output(tmp_path, output_file)
                        tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]

                        append_processed_entry(
                            config.PROCESSED_DOCS_LIST,
                            result['raw_rel_path'],
                            result['doc_id'],
                            result['chunk_count'],
                        )
                        processed_paths.add(result['raw_rel_path'])
                        processed_count += 1
                        total_chunks += int(result['chunk_count'])

                    pbar.update(1)
                    gc.collect()

                    submit_next()
        finally:
            pbar.close()

    print("\n=== FELDOLGOZÁS ÖSSZESÍTÉSE ===")
    print(f"Összes feldolgozott fájl: {processed_count:,}")
    print(f"Frissen generált chunkok: {total_chunks:,}")
    print(f"Sikertelen feldolgozások: {error_count:,}")


def process_all_documents(resume_mode: bool = False, workers: Optional[int] = None) -> None:
    process_all_docx_parallel(config.CHUNKS_JSONL, resume_mode=resume_mode, num_workers=workers)


def main() -> None:
    parser = argparse.ArgumentParser(description='CourtRankRL dokumentum feldolgozó')
    parser.add_argument('--resume', action='store_true', help='Folytatás mód – csak a kimaradt fájlokat dolgozza fel')
    parser.add_argument('--reset', action='store_true', help='Chunks és állapotfájlok törlése')
    parser.add_argument('--workers', type=int, default=None, help='Worker folyamatok száma')

    args = parser.parse_args()

    if args.reset:
        targets = [config.CHUNKS_JSONL, config.PROCESSED_DOCS_LIST]
        for target in targets:
            if target.exists():
                print(f"Reset: {target} törlése...")
                target.unlink()
        tmp_dir = config.CHUNKS_JSONL.parent / '.tmp_chunks'
        if tmp_dir.exists():
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"Ideiglenes könyvtár törölve: {tmp_dir}")
        print("Reset kész.")
        return

    try:
        process_all_docx_parallel(config.CHUNKS_JSONL, resume_mode=args.resume, num_workers=args.workers)
    except UnicodeEncodeError as exc:
        print(f"Unicode kódolási hiba: {exc}")
        print("Futtasd a scriptet UTF-8 kódolásra beállított környezetben.")
    except Exception as exc:
        print(f"Hiba történt a feldolgozás során: {exc}")


if __name__ == '__main__':
    main()
