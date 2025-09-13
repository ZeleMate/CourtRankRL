#!/usr/bin/env python3
"""
Dense Index Builder for CourtRankRL
FAISS index építése Gemini Batch API alapú embeddingekkel (gemini-embedding-001).
"""

import json
import numpy as np
import faiss
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from google import genai

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config

class DenseIndexBuilder:
    """FAISS index építése Gemini Batch API embeddingekkel (streaming, költséghatékony)."""

    def __init__(self):
        self.client = self._init_gemini_client()
        self.logger = self._setup_logging()
        self.stats = {
            'total_chunks': 0,
            'processed_chunks': 0,
            'failed_chunks': 0,
            'api_calls': 0,
            'start_time': time.time()
        }

    def _setup_logging(self) -> logging.Logger:
        """Logging beállítása config alapján."""
        logger = logging.getLogger('DenseIndexBuilder')
        level_str = str(config.LOGGING_LEVEL).upper()
        logger.setLevel(getattr(logging, level_str, logging.WARNING))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(config.LOGGING_FORMAT)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _validate_chunk_data(self, chunk: Dict) -> bool:
        """Chunk adat validálása."""
        required_fields = ['chunk_id', 'text']
        for field in required_fields:
            if field not in chunk:
                self.logger.warning(f"Missing required field '{field}' in chunk")
                return False

        if not chunk['text'].strip():
            self.logger.warning(f"Empty text in chunk {chunk['chunk_id']}")
            return False

        return True

    def _validate_embedding(self, embedding: List[float], chunk_id: str) -> bool:
        """Embedding validálása."""
        if not isinstance(embedding, list) or len(embedding) == 0:
            self.logger.warning(f"Invalid embedding for chunk {chunk_id}: not a list or empty")
            return False

        if len(embedding) != config.EMBEDDING_DIMENSION:
            self.logger.warning(f"Invalid embedding dimension for chunk {chunk_id}: expected {config.EMBEDDING_DIMENSION}, got {len(embedding)}")
            return False

        if not all(isinstance(x, (int, float)) for x in embedding):
            self.logger.warning(f"Invalid embedding values for chunk {chunk_id}: not all numeric")
            return False

        return True

    def _init_gemini_client(self):
        """Gemini GenAI Client inicializálása (google.genai)."""
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        # A google.genai kliens a környezeti változóból olvassa a kulcsot
        return genai.Client(api_key=api_key)

    def _write_batch_requests(self, out_path: Path, batch_items: List[Dict]) -> None:
        """Batch API-hoz szükséges JSONL fájl generálása.

        Minden sor: {"key": chunk_id, "request": {"output_dimensionality": D, "content": {"parts": [{"text": ...}]}}}
        """
        with open(out_path, 'w', encoding='utf-8') as f:
            for item in batch_items:
                line = {
                    "key": item['chunk_id'],
                    "request": {
                        "output_dimensionality": int(config.EMBEDDING_DIMENSION),
                        "content": {"parts": [{"text": item['text']}]}
                    }
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _parse_embedding_line(self, line: str) -> Dict:
        """Egy eredménysor feldolgozása; visszaadja {key: str, embedding: List[float]} vagy üres dictet."""
        try:
            obj = json.loads(line)
        except Exception:
            return {}

        key = obj.get('key') or obj.get('id')
        embedding = None

        # Lehetséges struktúrák támogatása
        if 'embedding' in obj:
            if isinstance(obj['embedding'], dict) and 'values' in obj['embedding']:
                embedding = obj['embedding']['values']
            elif isinstance(obj['embedding'], list):
                embedding = obj['embedding']
        elif 'response' in obj and isinstance(obj['response'], dict):
            emb = obj['response'].get('embedding')
            if isinstance(emb, dict) and 'values' in emb:
                embedding = emb['values']

        if key is None or embedding is None:
            return {}

        return {"key": key, "embedding": embedding}

    def _submit_batch_and_fetch(self, requests_path: Path, max_retries: int = 3) -> Dict[str, List[float]]:
        """Fájl feltöltése, batch job létrehozása és eredmények letöltése. Vissza: chunk_id -> embedding."""
        self.stats['api_calls'] += 1

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Uploading batch file: {requests_path.name}")
                # Gemini Batch API - próbáljuk MIME type nélkül először
                uploaded = self.client.files.upload(file=str(requests_path))

                self.logger.info(f"Creating batch embedding job (attempt {attempt + 1}/{max_retries})")
                job = self.client.batches.create_embeddings(
                    model="gemini-embedding-001",
                    src={"file_name": uploaded.name},
                )

                # Intelligens polling exponenciális backoff-fal
                job_name = job.name
                poll_interval = 5  # seconds
                max_poll_time = 3600  # 1 hour maximum
                start_time = time.time()

                while time.time() - start_time < max_poll_time:
                    time.sleep(poll_interval)
                    job = self.client.batches.get(name=job_name)

                    state = getattr(job, 'state', None)
                    state_name = getattr(state, 'name', str(state)) if state else 'UNKNOWN'

                    if state_name in ('JOB_STATE_SUCCEEDED', 'SUCCEEDED'):
                        self.logger.info(f"Batch job completed successfully")
                        break
                    elif state_name in ('JOB_STATE_FAILED', 'FAILED', 'CANCELLED', 'JOB_STATE_CANCELLED'):
                        error_msg = getattr(job, 'error', {}).get('message', 'Unknown error')
                        raise RuntimeError(f"Batch job failed: {state_name} - {error_msg}")
                    elif state_name == 'JOB_STATE_RUNNING':
                        self.logger.debug(f"Job running... ({int(time.time() - start_time)}s elapsed)")
                    elif state_name == 'JOB_STATE_PENDING':
                        self.logger.debug(f"Job pending... ({int(time.time() - start_time)}s elapsed)")

                    # Exponenciális backoff polling esetén
                    poll_interval = min(poll_interval * 1.2, 30)  # Max 30 seconds
                else:
                    raise RuntimeError(f"Batch job timeout after {max_poll_time} seconds")

                # Eredmények letöltése
                self.logger.info("Downloading results...")
                result_file_name = job.dest.file_name
                content_bytes = self.client.files.download(file=result_file_name)
                content = content_bytes.decode('utf-8')

                embeddings_map: Dict[str, List[float]] = {}
                failed_lines = 0

                for line_num, line in enumerate(content.splitlines(), 1):
                    if not line.strip():
                        continue

                    parsed = self._parse_embedding_line(line)
                    if not parsed:
                        failed_lines += 1
                        self.logger.debug(f"Failed to parse line {line_num}")
                        continue

                    chunk_id = parsed['key']
                    embedding = parsed['embedding']

                    if self._validate_embedding(embedding, chunk_id):
                        embeddings_map[chunk_id] = embedding
                    else:
                        failed_lines += 1

                if failed_lines > 0:
                    self.logger.warning(f"Failed to parse {failed_lines} embedding lines")

                self.logger.info(f"Successfully processed {len(embeddings_map)} embeddings from batch")
                return embeddings_map

            except Exception as e:
                self.logger.error(f"Batch processing error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff for retries

    def build_index(self, chunks_jsonl: Path, resume_from: Optional[int] = None):
        """FAISS index építése batch embeddingekkel (streamelve)."""
        print("=== DENSE INDEX ÉPÍTÉS ===")
        print(f"Chunks fájl: {chunks_jsonl}")
        print(f"Model: gemini-embedding-001")
        print(f"Embedding dimenzió: {config.EMBEDDING_DIMENSION}")

        # Streaming paraméterek
        batch_size = int(os.getenv('EMBED_BATCH_SIZE', '50000'))
        print(f"Batch méret: {batch_size}")

        # FAISS index előkészítés vagy betöltés (resume esetén)
        index = None
        chunk_id_map = {}

        if resume_from is not None and config.FAISS_INDEX_PATH.exists():
            print(f"Resume from row {resume_from}")
            index = faiss.read_index(str(config.FAISS_INDEX_PATH))
            with open(config.CHUNK_ID_MAP_PATH, 'r', encoding='utf-8') as f:
                chunk_id_map = json.load(f)
            current_row = len(chunk_id_map)
        else:
            current_row = 0

        tmp_dir = Path("data/processed/tmp_embeddings")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        batch_items: List[Dict] = []
        processed_lines = 0

        def flush_batch(items: List[Dict]):
            nonlocal index, chunk_id_map, current_row
            if not items:
                return

            valid_items = [it for it in items if self._validate_chunk_data(it)]
            if not valid_items:
                self.logger.warning("No valid items in batch")
                return

            req_path = tmp_dir / f"embedding_requests_{current_row}.json"
            self._write_batch_requests(req_path, valid_items)
            emb_map = self._submit_batch_and_fetch(req_path)

            # Align és normalizálás
            aligned: List[List[float]] = []
            aligned_ids: List[str] = []
            failed_embeddings = 0

            for it in valid_items:
                cid = it['chunk_id']
                emb = emb_map.get(cid)
                if emb is None:
                    self.logger.debug(f"No embedding found for chunk {cid}")
                    self.stats['failed_chunks'] += 1
                    failed_embeddings += 1
                    continue

                if not self._validate_embedding(emb, cid):
                    self.stats['failed_chunks'] += 1
                    failed_embeddings += 1
                    continue

                vec = np.array(emb, dtype=np.float32)
                # Normalizálás koszinuszhoz
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                aligned.append(vec)
                aligned_ids.append(cid)

            if not aligned:
                self.logger.warning(f"No valid embeddings in batch (failed: {failed_embeddings})")
                return

            arr = np.vstack(aligned).astype(np.float32)
            if index is None:
                # IndexFlatIP használata koszinusz hasonlósághoz (normalizált vektorok)
                index = faiss.IndexIVFFlat(arr.shape[1])
            index.add(arr)

            for cid in aligned_ids:
                chunk_id_map[current_row] = cid
                current_row += 1

            self.stats['processed_chunks'] += len(aligned)

            # Progress riport
            elapsed = time.time() - self.stats['start_time']
            progress = self.stats['processed_chunks'] / max(1, self.stats['total_chunks']) * 100
            print(f"Progress: {self.stats['processed_chunks']:,}/{self.stats['total_chunks']:,} "
                  f"({progress:.1f}%) - Elapsed: {elapsed:.1f}s")

            try:
                req_path.unlink(missing_ok=True)
            except Exception:
                pass

        # Total chunks számolása
        with open(chunks_jsonl, 'r', encoding='utf-8') as f:
            self.stats['total_chunks'] = sum(1 for _ in f)

        print(f"Total chunks to process: {self.stats['total_chunks']:,}")

        # Kérések készítése streamelve a chunks.jsonl-ből
        with open(chunks_jsonl, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if resume_from is not None and line_num <= resume_from:
                    continue

                try:
                    d = json.loads(line.strip())
                    if not self._validate_chunk_data(d):
                        self.stats['failed_chunks'] += 1
                        continue

                    batch_items.append({
                        'chunk_id': d['chunk_id'],
                        'text': d['text'],
                    })

                    if len(batch_items) >= batch_size:
                        flush_batch(batch_items)
                        batch_items = []

                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error at line {line_num}: {e}")
                    self.stats['failed_chunks'] += 1
                    continue

        # Maradék flush
        if batch_items:
            flush_batch(batch_items)

        if index is None or self.stats['processed_chunks'] == 0:
            raise ValueError("No embeddings were created. Check API key and input data.")

        # Mentés
        print(f"\nMentés: {config.FAISS_INDEX_PATH}")
        faiss.write_index(index, str(config.FAISS_INDEX_PATH))

        print(f"Chunk ID mapping mentés: {config.CHUNK_ID_MAP_PATH}")
        with open(config.CHUNK_ID_MAP_PATH, 'w', encoding='utf-8') as f:
            json.dump(chunk_id_map, f, ensure_ascii=False)

        # Statisztikák
        elapsed = time.time() - self.stats['start_time']
        print("\n=== FELDOLGOZÁS EREDMÉNYE ===")
        print(f"Sikeres embeddingek: {self.stats['processed_chunks']:,}")
        print(f"Sikertelen chunkok: {self.stats['failed_chunks']:,}")
        print(f"API hívások: {self.stats['api_calls']}")
        print(f"Teljes idő: {elapsed:.2f}s")
        print(f"Átlagos feldolgozási sebesség: {self.stats['processed_chunks'] / elapsed:.1f} chunk/s")

def main():
    """Build dense index from chunks with command line options."""
    import argparse

    parser = argparse.ArgumentParser(description='CourtRankRL Dense Index Builder')
    parser.add_argument('--resume-from', type=int, default=None,
                       help='Resume from specific line number in chunks file')
    parser.add_argument('--reset', action='store_true',
                       help='Reset and rebuild index from scratch')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size (default: 50000)')

    args = parser.parse_args()

    if not config.CHUNKS_JSONL.exists():
        print(f"❌ HIBA: Chunks fájl nem található: {config.CHUNKS_JSONL}")
        return

    # Reset mód
    if args.reset:
        if config.FAISS_INDEX_PATH.exists():
            print(f"Reset: {config.FAISS_INDEX_PATH} törlése...")
            config.FAISS_INDEX_PATH.unlink(missing_ok=True)
        if config.CHUNK_ID_MAP_PATH.exists():
            print(f"Reset: {config.CHUNK_ID_MAP_PATH} törlése...")
            config.CHUNK_ID_MAP_PATH.unlink(missing_ok=True)
        print("Reset kész.\n")

    # Batch size override
    if args.batch_size:
        import os
        os.environ['EMBED_BATCH_SIZE'] = str(args.batch_size)
        print(f"Batch méret felülírva: {args.batch_size}")

    try:
        builder = DenseIndexBuilder()
        builder.build_index(config.CHUNKS_JSONL, resume_from=args.resume_from)

        print("\n✅ DENSE INDEX SIKERESEN FELÉPÍTVE!")
        print(f"FAISS index: {config.FAISS_INDEX_PATH}")
        print(f"Chunk mapping: {config.CHUNK_ID_MAP_PATH}")

    except KeyboardInterrupt:
        print("\n⚠️  FELHASZNÁLÓI MEGSZAKÍTÁS")
        print("Használja --resume-from kapcsolót a folytatáshoz")
        print(f"Példa: python create_embeddings_gemini_api.py --resume-from {builder.stats['processed_chunks'] if 'builder' in locals() else 0}")
    except Exception as e:
        print(f"\n❌ HIBA: {e}")
        if 'builder' in locals():
            print(f"Hibás chunkok: {builder.stats['failed_chunks']}")
            print(f"API hívások: {builder.stats['api_calls']}")

if __name__ == '__main__':
    main() 