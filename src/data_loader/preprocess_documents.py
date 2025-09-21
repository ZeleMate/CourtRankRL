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
    parts = filepath.parts
    court = domain = year = case_id = ""

    if len(parts) >= 4:
        court = parts[-4]
        domain = parts[-3]
        year = parts[-2]
        case_id = Path(parts[-1]).stem

    if not all([court, domain, year]):
        stem = filepath.stem
        tokens = stem.split('_')
        if len(tokens) >= 4:
            court, domain, year = tokens[:3]
            case_id = '_'.join(tokens)
        else:
            case_id = stem

    return {
        'court': court,
        'domain': domain,
        'year': year,
        'case_identifier': case_id,
    }


def extract_metadata_from_docling_text(text: str) -> Dict[str, str]:
    import re

    court = ""
    case_id = ""
    lines = text.split('\n')[:15]

    court_patterns = [
        r'([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+)*\s*(?:[Íí]télőtábla|[Tt]örvényszék|[Jj]árásbíróság|[Kk]özigazgatási\s+[és\s+]*[Mm]unkaügyi\s+[Bb]íróság))',
        r'([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+\s*[Bb]íróság)',
    ]
    case_patterns = [
        r'(Bf\.[IVXLCDM]+\.\d+/\d+/\d+\.?\s+szám)',
        r'(\d+\.[A-Z]\.\d+/\d+/\d+\.?\s+szám)',
        r'([A-Z]\.\d+\.\d+/\d+/\d+\.?\s+szám)',
    ]

    for line in lines:
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

    return {'court': court, 'case_identifier': case_id}


def iter_docling_chunks(docling_document: DoclingDocument) -> Iterator[Dict[str, object]]:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

    chunker = HybridChunker()
    for chunk in chunker.chunk(docling_document):
        chunk_text = getattr(chunk, 'text', None)
        if chunk_text and str(chunk_text).strip():
            yield {
                'text': chunk_text,
                'chunk_type': getattr(chunk, 'chunk_type', 'docling_chunk'),
                'hierarchy_level': getattr(chunk, 'hierarchy_level', 0),
                'page_numbers': getattr(chunk, 'page_numbers', [1]),
                'headings': getattr(chunk, 'headings', []),
                'captions': getattr(chunk, 'captions', []),
            }


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
    processed_paths: Set[str] = set()
    legacy_doc_ids: Set[str] = set()

    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_path = entry.get('raw_path') or entry.get('source_path')
                if raw_path:
                    processed_paths.add(normalize_raw_path(raw_path, raw_root))
                    continue
                doc_id = entry.get('doc_id')
                if doc_id:
                    legacy_doc_ids.add(doc_id)

    if legacy_doc_ids and chunks_file.exists():
        with open(chunks_file, 'r', encoding='utf-8') as handle:
            for line in handle:
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = chunk.get('doc_id')
                if doc_id in legacy_doc_ids:
                    source_path = chunk.get('source_path')
                    if source_path:
                        processed_paths.add(normalize_raw_path(source_path, raw_root))
                        legacy_doc_ids.discard(doc_id)
                if not legacy_doc_ids:
                    break

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
            return None

        path_meta = extract_metadata_from_path(filepath)
        doc_id = path_meta.get('case_identifier') or filepath.stem
        doc_metadata = {
            'court': path_meta.get('court', ''),
            'domain': path_meta.get('domain', ''),
            'year': path_meta.get('year', ''),
            'source_path': str(filepath),
        }

        pre_chunks: List[Dict[str, object]] = []
        pre_text_parts: List[str] = []
        char_cap = 2000
        char_count = 0

        chunk_iter = iter_docling_chunks(docling_doc)
        for _ in range(10):
            try:
                chunk = next(chunk_iter)
            except StopIteration:
                break
            pre_chunks.append(chunk)
            text = chunk.get('text', '')
            if text:
                pre_text_parts.append(str(text))
                char_count += len(str(text))
            if char_count >= char_cap:
                break

        if pre_text_parts:
            preview = minimal_normalize_text('\n'.join(pre_text_parts[:5]))
            docling_meta = extract_metadata_from_docling_text(preview)
            if docling_meta.get('court'):
                doc_metadata['court'] = docling_meta['court']
            if docling_meta.get('case_identifier'):
                doc_id = docling_meta['case_identifier']

        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"tmp_{filepath.stem}_{mp.current_process().pid}.jsonl"
        batch_buffer: List[str] = []
        chunk_count = 0
        running_index = 0

        with open(tmp_path, 'w', encoding='utf-8') as handle:
            def flush_buffer() -> None:
                if batch_buffer:
                    handle.write(''.join(batch_buffer))
                    batch_buffer.clear()

            for chunk in pre_chunks:
                payload = {
                    'chunk_id': f"{doc_id}_{running_index}",
                    'doc_id': doc_id,
                    'text': chunk['text'],
                    'court': doc_metadata.get('court', ''),
                    'domain': doc_metadata.get('domain', ''),
                    'year': doc_metadata.get('year', ''),
                    'source_path': str(filepath),
                }
                batch_buffer.append(json.dumps(payload, ensure_ascii=False) + '\n')
                running_index += 1
                chunk_count += 1
                if len(batch_buffer) >= batch_size:
                    flush_buffer()

            for chunk in chunk_iter:
                payload = {
                    'chunk_id': f"{doc_id}_{running_index}",
                    'doc_id': doc_id,
                    'text': chunk['text'],
                    'court': doc_metadata.get('court', ''),
                    'domain': doc_metadata.get('domain', ''),
                    'year': doc_metadata.get('year', ''),
                    'source_path': str(filepath),
                }
                batch_buffer.append(json.dumps(payload, ensure_ascii=False) + '\n')
                running_index += 1
                chunk_count += 1
                if len(batch_buffer) >= batch_size:
                    flush_buffer()

            flush_buffer()

        if chunk_count == 0:
            tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            return None

        try:
            del docling_doc
            del pre_chunks
            del pre_text_parts
            del chunk_iter
        except Exception:
            pass
        gc.collect()

        return {
            'doc_id': doc_id,
            'chunk_count': chunk_count,
            'tmp_file': str(tmp_path),
            'raw_rel_path': normalize_raw_path(filepath, raw_root),
        }

    except Exception:
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
