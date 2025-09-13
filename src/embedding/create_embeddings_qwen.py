#!/usr/bin/env python3
"""
Dense Index Builder for CourtRankRL
FAISS index építése Qwen3-Embedding-0.6B alapú embeddingekkel.
MEGJEGYZÉS: Ez a verzió egyszerűsített a stabilitás érdekében.
"""

import json
import numpy as np
import faiss
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
import torch
from sentence_transformers import SentenceTransformer

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Egyszerűsített config - kerüljük a komplex importokat
class SimpleConfig:
    """Egyszerűsített config osztály tesztelés céljából."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    INDEX_DIR = DATA_DIR / "index"
    MODELS_DIR = DATA_DIR / "models"

    # Fájlok
    CHUNKS_JSONL = PROCESSED_DATA_DIR / "chunks.jsonl"
    FAISS_INDEX_PATH = INDEX_DIR / "faiss_index_qwen.bin"
    CHUNK_ID_MAP_PATH = INDEX_DIR / "chunk_id_map_qwen.json"

    # Modell beállítások
    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_DIMENSION = 1024

    # Retrieval beállítások
    LOGGING_LEVEL = logging.WARNING

# Használjuk az egyszerűsített config-ot
config = SimpleConfig()

class DenseIndexBuilder:
    """FAISS index építése Qwen3-Embedding-0.6B embeddingekkel."""

    def __init__(self):
        self.logger = self._setup_logging()
        self.device = self._get_device()
        self.model = self._init_qwen_model()
        self.stats = {
            'total_chunks': 0,
            'processed_chunks': 0,
            'failed_chunks': 0,
            'start_time': time.time()
        }

    def _setup_logging(self) -> logging.Logger:
        """Logging beállítása config alapján."""
        logger = logging.getLogger('DenseIndexBuilder')
        level_str = str(config.LOGGING_LEVEL).upper()
        logger.setLevel(getattr(logging, level_str, logging.WARNING))

        # Egyszerűsített logging tesztelés céljából
        if not logger.handlers:
            handler = logging.StreamHandler()
            # Egyszerű formázás
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _get_device(self) -> str:
        """Eszköz kiválasztása (GPU ha elérhető, különben CPU)."""
        if torch.cuda.is_available():
            device = 'cuda'
            self.logger.info("CUDA elérhető, GPU használata")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            self.logger.info("MPS elérhető, Apple Silicon GPU használata")
        else:
            device = 'cpu'
            self.logger.info("CPU használata")

        return device

    def _init_qwen_model(self) -> SentenceTransformer:
        """Qwen3-Embedding-0.6B modell inicializálása."""
        model_name = "Qwen/Qwen3-Embedding-0.6B"

        try:
            self.logger.info(f"Qwen3-Embedding-0.6B modell betöltése: {model_name}")
            model = SentenceTransformer(model_name, trust_remote_code=True)

            # Embedding dimenzió ellenőrzése
            if hasattr(model, 'encode'):
                # Teszt encoding hogy működik-e
                test_embedding = model.encode("teszt szöveg", normalize_embeddings=True)
                actual_dim = len(test_embedding)
                self.logger.info(f"Modell betöltve, embedding dimenzió: {actual_dim}")

                # Ellenőrizzük hogy egyezik-e a config-gal
                if actual_dim != config.EMBEDDING_DIMENSION:
                    self.logger.warning(f"Modell embedding dimenzió ({actual_dim}) nem egyezik a config-ban beállítottal ({config.EMBEDDING_DIMENSION})")
                    self.logger.info(f"Config frissítése szükséges: EMBEDDING_DIMENSION = {actual_dim}")

            return model

        except Exception as e:
            raise RuntimeError(f"Qwen3-Embedding-0.6B modell betöltési hiba: {e}")

    def _validate_chunk_data(self, chunk: Dict) -> bool:
        """Chunk adat validálása."""
        required_fields = ['chunk_id', 'text']
        for field in required_fields:
            if field not in chunk:
                self.logger.warning(f"Hiányzó kötelező mező '{field}' a chunk-ban")
                return False

        if not chunk['text'].strip():
            self.logger.warning(f"Üres szöveg a chunk-ban: {chunk['chunk_id']}")
            return False

        return True

    def _validate_embedding(self, embedding: np.ndarray, chunk_id: str) -> bool:
        """Embedding validálása."""
        if not isinstance(embedding, np.ndarray) or embedding.size == 0:
            self.logger.warning(f"Érvénytelen embedding a chunk-hoz {chunk_id}: nem ndarray vagy üres")
            return False

        if len(embedding) != config.EMBEDDING_DIMENSION:
            self.logger.warning(f"Érvénytelen embedding dimenzió a chunk-hoz {chunk_id}: elvárt {config.EMBEDDING_DIMENSION}, kapott {len(embedding)}")
            return False

        if not np.isfinite(embedding).all():
            self.logger.warning(f"Érvénytelen embedding értékek a chunk-hoz {chunk_id}: nem véges számok")
            return False

        return True

    def _generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Embeddingek generálása batch-ben."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            self.logger.debug(f"Batch feldolgozása: {i}-{min(i + batch_size, len(texts))}")

            try:
                # Embedding generálás normalizálással koszinusz hasonlósághoz
                batch_embeddings = self.model.encode(
                    batch_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    device=self.device
                )

                # Validálás
                for j, emb in enumerate(batch_embeddings):
                    if not self._validate_embedding(emb, f"batch_{i+j}"):
                        self.logger.warning(f"Érvénytelen embedding a batch {i+j}. pozícióban")
                        # Helyettesítjük null vektorral
                        batch_embeddings[j] = np.zeros(config.EMBEDDING_DIMENSION, dtype=np.float32)

                embeddings.extend(batch_embeddings)

            except Exception as e:
                self.logger.error(f"Batch embedding generálási hiba: {e}")
                # Hibás batch esetén null vektorok hozzáadása
                for _ in batch_texts:
                    embeddings.append(np.zeros(config.EMBEDDING_DIMENSION, dtype=np.float32))

        return embeddings

    def build_index(self, chunks_jsonl: Path, resume_from: Optional[int] = None):
        """FAISS index építése batch embeddingekkel."""
        print("=== DENSE INDEX ÉPÍTÉS ===")
        print(f"Chunks fájl: {chunks_jsonl}")
        print(f"Modell: Qwen3-Embedding-0.6B")
        print(f"Eszköz: {self.device}")
        print(f"Embedding dimenzió: {config.EMBEDDING_DIMENSION}")

        # Batch méret beállítása
        batch_size = 32  # Qwen esetében kisebb batch méret optimális
        print(f"Batch méret: {batch_size}")

        # FAISS index előkészítés vagy betöltés (resume esetén)
        index = None
        chunk_id_map = {}

        if resume_from is not None and config.FAISS_INDEX_PATH.exists():
            print(f"Folytatás a {resume_from}. sortól")
            index = faiss.read_index(str(config.FAISS_INDEX_PATH))
            with open(config.CHUNK_ID_MAP_PATH, 'r', encoding='utf-8') as f:
                chunk_id_map = json.load(f)
            current_row = len(chunk_id_map)
        else:
            current_row = 0

        batch_items: List[Dict] = []
        processed_lines = 0

        def flush_batch(items: List[Dict]):
            nonlocal index, chunk_id_map, current_row
            if not items:
                return

            valid_items = [it for it in items if self._validate_chunk_data(it)]
            if not valid_items:
                self.logger.warning("Nincs érvényes elem a batch-ben")
                return

            # Embeddingek generálása
            texts = [it['text'] for it in valid_items]
            embeddings = self._generate_embeddings_batch(texts, batch_size=batch_size)

            # Align és validáció
            aligned: List[np.ndarray] = []
            aligned_ids: List[str] = []
            failed_embeddings = 0

            for i, (it, emb) in enumerate(zip(valid_items, embeddings)):
                cid = it['chunk_id']

                if not self._validate_embedding(emb, cid):
                    self.stats['failed_chunks'] += 1
                    failed_embeddings += 1
                    continue

                # Már normalizálva van a model.encode-ból (normalize_embeddings=True)
                aligned.append(emb.astype(np.float32))
                aligned_ids.append(cid)

            if not aligned:
                self.logger.warning(f"Nincs érvényes embedding a batch-ben (sikertelen: {failed_embeddings})")
                return

            arr = np.vstack(aligned).astype(np.float32)
            if index is None:
                # IndexFlatIP használata koszinusz hasonlósághoz (már normalizált vektorok)
                # Egyszerűbb index típus tesztelés céljából
                try:
                    index = faiss.IndexFlatIP(arr.shape[1])  # Inner product koszinusz hasonlósághoz
                    self.logger.info("IndexFlatIP használata")
                except Exception as e:
                    self.logger.warning(f"IndexFlatIP hiba, IndexFlatL2 használata: {e}")
                    index = faiss.IndexFlatL2(arr.shape[1])  # L2 distance fallback

            index.add(arr)

            for cid in aligned_ids:
                chunk_id_map[str(current_row)] = cid  # String kulcsok használata konzisztenciáért
                current_row += 1

            self.stats['processed_chunks'] += len(aligned)

            # Progress riport
            elapsed = time.time() - self.stats['start_time']
            progress = self.stats['processed_chunks'] / max(1, self.stats['total_chunks']) * 100
            print(f"Progress: {self.stats['processed_chunks']:,}/{self.stats['total_chunks']:,} "
                  f"({progress:.1f}%) - Elapsed: {elapsed:.1f}s")

        # Total chunks számolása
        with open(chunks_jsonl, 'r', encoding='utf-8') as f:
            self.stats['total_chunks'] = sum(1 for _ in f)

        print(f"Összes feldolgozandó chunk: {self.stats['total_chunks']:,}")

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
                    self.logger.error(f"JSON dekódolási hiba a {line_num}. sorban: {e}")
                    self.stats['failed_chunks'] += 1
                    continue

        # Maradék flush
        if batch_items:
            flush_batch(batch_items)

        if index is None or self.stats['processed_chunks'] == 0:
            raise ValueError("Nem készült el egyetlen embedding sem. Ellenőrizze a bemeneti adatokat.")

        # Mentés
        print(f"\nMentés: {config.FAISS_INDEX_PATH}")
        faiss.write_index(index, str(config.FAISS_INDEX_PATH))

        print(f"Chunk ID mapping mentés: {config.CHUNK_ID_MAP_PATH}")
        with open(config.CHUNK_ID_MAP_PATH, 'w', encoding='utf-8') as f:
            json.dump(chunk_id_map, f, ensure_ascii=False, indent=2)

        # Statisztikák
        elapsed = time.time() - self.stats['start_time']
        print("\n=== FELDOLGOZÁS EREDMÉNYE ===")
        print(f"Sikeres embeddingek: {self.stats['processed_chunks']:,}")
        print(f"Sikertelen chunkok: {self.stats['failed_chunks']:,}")
        print(f"Teljes idő: {elapsed:.2f}s")
        print(f"Átlagos feldolgozási sebesség: {self.stats['processed_chunks'] / elapsed:.1f} chunk/s")

def main():
    """Build dense index from chunks with command line options."""
    import argparse

    parser = argparse.ArgumentParser(description='CourtRankRL Dense Index Builder - Qwen3')
    parser.add_argument('--resume-from', type=int, default=None,
                       help='Folytatás adott sorszámtól a chunks fájlban')
    parser.add_argument('--reset', action='store_true',
                       help='Reset és teljes újjáépítés')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch méret felülírása (alapértelmezett: 32)')
    parser.add_argument('--test-only', action='store_true',
                       help='Csak teszt futtatás, nem építi fel az indexet')

    args = parser.parse_args()

    if args.test_only:
        print("🧪 QWEN3 TESZT MOD")
        print("=" * 50)

        try:
            print("1. Modell betöltése...")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)
            print("✅ Modell betöltve")

            print("2. Teszt embedding generálás...")
            test_texts = [
                "Ez egy teszt mondat magyar bírósági határozatokhoz.",
                "A szerződés érvénytelen, mert hiányzik az aláírás."
            ]
            embeddings = model.encode(test_texts, normalize_embeddings=True)
            print(f"✅ Embeddingek generálva: {len(embeddings)} db, dimenzió: {len(embeddings[0])}")

            print("3. FAISS teszt...")
            import faiss
            import numpy as np
            arr = np.array(embeddings, dtype=np.float32)
            index = faiss.IndexFlatIP(arr.shape[1])
            index.add(arr)
            print("✅ FAISS index létrehozva és feltöltve")

            print("\n✅ MINDEN TESZT SIKERES!")
            print("A Qwen3-Embedding-0.6B modell használatra kész.")

        except Exception as e:
            print(f"❌ TESZT HIBA: {e}")
            import traceback
            traceback.print_exc()

        return

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

    try:
        builder = DenseIndexBuilder()
        builder.build_index(config.CHUNKS_JSONL, resume_from=args.resume_from)

        print("\n✅ DENSE INDEX SIKERESEN FELÉPÍTVE!")
        print(f"FAISS index: {config.FAISS_INDEX_PATH}")
        print(f"Chunk mapping: {config.CHUNK_ID_MAP_PATH}")

    except KeyboardInterrupt:
        print("\n⚠️  FELHASZNÁLÓI MEGSZAKÍTÁS")
        print("Használja --resume-from kapcsolót a folytatáshoz")
        print(f"Példa: python create_embeddings_qwen.py --resume-from {builder.stats['processed_chunks'] if 'builder' in locals() else 0}")
    except Exception as e:
        print(f"\n❌ HIBA: {e}")
        if 'builder' in locals():
            print(f"Hibás chunkok: {builder.stats['failed_chunks']}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()