#!/usr/bin/env python3
"""
BM25S Index Builder for CourtRankRL
Agents.md specifikáció szerint implementálva.

Főbb jellemzők:
- BM25S könyvtár natív tokenizáló használata
- Token ID-k, szókincs és hossz metaadatok cache-elése
- BM25S index fájlok (scores/indices/indptr/vocab/params)
- Chunk ID lista és token statisztikák (hossz eloszlás, összes docs)
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import tqdm

import bm25s
from bm25s import tokenization

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config

class BM25Index:
    """BM25S alapú index a CourtRankRL rendszerhez."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # BM25S modell és adatok
        self.bm25s_model = None
        self.corpus: List[str] = []
        self.chunk_ids: List[str] = []
        self.total_docs = 0
        self.avg_doc_length = 0.0
        self.doc_lengths: List[int] = []
        self.vocab: Dict[str, int] = {}
        self.token_ids: List[List[int]] = []

    def _tokenize_query(self, query: str):
        """Tokenizálás egyetlen lekérdezéshez."""
        return bm25s.tokenize([query])

    @staticmethod
    def _has_tokens(tokenized: Any) -> bool:
        """Ellenőrzi, hogy vannak-e tokenek."""
        if hasattr(tokenized, 'lengths') and tokenized.lengths:
            return any(length > 0 for length in tokenized.lengths)
        if hasattr(tokenized, 'ids') and tokenized.ids:
            return any(len(entry) > 0 for entry in tokenized.ids)
        return False

    def _load_token_cache(self, expected_docs: int) -> Optional[tokenization.Tokenized]:
        """Token cache betöltése, ha létezik és érvényes."""
        cache_dir = config.BM25_TOKEN_CACHE_DIR
        ids_path = cache_dir / "token_ids.npy"
        vocab_path = cache_dir / "vocab.json"

        if not (config.BM25_USE_CACHE and ids_path.exists() and vocab_path.exists()):
            return None

        try:
            token_ids_array = np.load(ids_path, allow_pickle=True)
            token_ids: List[List[int]] = token_ids_array.tolist()

            if expected_docs and len(token_ids) != expected_docs:
                return None

            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab: Dict[str, int] = json.load(f)

            return tokenization.Tokenized(ids=token_ids, vocab=vocab)
        except Exception:
            # Ha bármilyen hiba van, cache-t figyelmen kívül hagyjuk
            return None

    def _save_token_cache(self, tokenized: tokenization.Tokenized) -> None:
        """Token cache mentése a gyors újjáépítéshez."""
        if not config.BM25_USE_CACHE:
            return

        cache_dir = config.BM25_TOKEN_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        ids_path = cache_dir / "token_ids.npy"
        vocab_path = cache_dir / "vocab.json"

        np.save(ids_path, np.array(tokenized.ids, dtype=object), allow_pickle=True)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(tokenized.vocab, f, ensure_ascii=False)

    def _tokenize_corpus(self, texts: List[str]) -> tokenization.Tokenized:
        """Tokenizált korpusz előállítása cache figyelembevételével."""
        cached = self._load_token_cache(len(texts))
        if cached is not None:
            print("Token cache betöltve — tokenizálás kihagyva")
            return cached

        print("Tokenizálás...")
        if config.BM25_STOPWORDS is None:
            tokenized_result = bm25s.tokenize(
                texts,
                return_ids=True,
                show_progress=True
            )
        else:
            tokenized_result = bm25s.tokenize(
                texts,
                stopwords=config.BM25_STOPWORDS,
                return_ids=True,
                show_progress=True
            )

        tokenized = cast(tokenization.Tokenized, tokenized_result)

        self._save_token_cache(tokenized)
        return tokenized

    def _build_index_from_texts(self, texts: List[str], chunk_ids: List[str]) -> None:
        """BM25S index építése szövegekből."""
        tokenized = self._tokenize_corpus(texts)

        # Token ID-k és szókincs mentése
        self.token_ids = list(tokenized.ids)
        self.vocab = dict(tokenized.vocab)
        self.corpus = texts
        self.chunk_ids = chunk_ids
        self.total_docs = len(texts)

        # Dokumentumhosszok számítása
        self.doc_lengths = [len(ids) for ids in self.token_ids]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

        # BM25S index építése
        print("BM25S index építése...")
        self.bm25s_model = bm25s.BM25(k1=self.k1, b=self.b)

        if config.BM25_USE_NUMBA:
            try:
                self.bm25s_model.activate_numba_scorer()
                print("Numba gyorsítás aktiválva")
            except ImportError:
                print("Numba nem érhető el – CPU alapú scorer marad")

        self.bm25s_model.index(tokenized)
        self.bm25s_model.corpus = np.array(self.chunk_ids)

        print(f"BM25S index kész: {self.total_docs} dokumentum")

    def build_index(self, chunks_jsonl: Path) -> None:
        """BM25S index építése chunks JSONL fájlból."""
        print(f"BM25S index építése: {chunks_jsonl}")

        # JSONL fájl beolvasása
        texts = []
        chunk_ids = []

        with open(chunks_jsonl, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    print(f"JSON hiba {line_num}. sorban")
                    continue

                chunk_id = chunk.get('chunk_id', '').strip()
                text = chunk.get('text', '').strip()

                if chunk_id and text:
                    texts.append(text)
                    chunk_ids.append(chunk_id)

        if not texts:
            print("Nincs érvényes chunk a feldolgozáshoz")
            return

        print(f"Feldolgozandó chunkok: {len(texts)}")

        # Index építése
        self._build_index_from_texts(texts, chunk_ids)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """BM25S keresés."""
        if self.bm25s_model is None or self.total_docs == 0:
            return []

        # Tokenizálás
        query_tokens = self._tokenize_query(query)
        if not self._has_tokens(query_tokens):
            return []

        try:
            # Keresés
            k = min(top_k, len(self.chunk_ids))
            documents, scores = self.bm25s_model.retrieve(
                query_tokens,
                corpus=self.chunk_ids,
                k=k,
                return_as="tuple",
                show_progress=False
            )

            # Eredmények feldolgozása
            results = []
            doc_indices = documents[0] if len(documents) > 0 else []
            score_values = scores[0] if len(scores) > 0 else []

            for doc_idx, score_val in zip(doc_indices, score_values):
                if isinstance(doc_idx, (int, np.integer)) and 0 <= doc_idx < len(self.chunk_ids):
                    doc_id = self.chunk_ids[int(doc_idx)]
                    results.append((doc_id, float(score_val)))

            return results

        except Exception as e:
            print(f"BM25S keresési hiba: {e}")
            return []

    def _get_length_histogram(self, bins: int = 20) -> List[int]:
        """Hossz eloszlás hisztogram."""
        if not self.doc_lengths:
            return []

        max_len = max(self.doc_lengths)
        if max_len == 0:
            return []

        step = math.ceil(max_len / bins)
        histogram = [0] * bins

        for length in self.doc_lengths:
            bucket = min(length // step, bins - 1)
            histogram[bucket] += 1

        return histogram

    def save(self, output_path: Path) -> None:
        """BM25S index mentése."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # BM25S model könyvtár
        bm25_dir = config.BM25_INDEX_DIR
        bm25_dir.mkdir(parents=True, exist_ok=True)
        model_dir = bm25_dir / "bm25s_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # BM25S model mentése
        if self.bm25s_model:
            self.bm25s_model.save(str(model_dir))

        # Chunk ID-k mentése
        chunk_ids_path = bm25_dir / "chunk_ids.json"
        with open(chunk_ids_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_ids, f, ensure_ascii=False, indent=2)

        # Statisztikák mentése
        histogram = self._get_length_histogram()
        stats = {
            'total_docs': self.total_docs,
            'avg_doc_length': self.avg_doc_length,
            'length_histogram': histogram,
            'min_doc_length': min(self.doc_lengths) if self.doc_lengths else 0,
            'max_doc_length': max(self.doc_lengths) if self.doc_lengths else 0,
            'vocab_size': len(self.vocab),
        }

        stats_path = config.BM25_STATS_PATH
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # Index metaadatok
        index_data = {
            'model_dir': str(model_dir),
            'chunk_ids_path': str(chunk_ids_path),
            'stats_path': str(stats_path),
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

        print(f"BM25S index mentve: {model_dir}")

    @classmethod
    def load(cls, input_path: Path) -> 'BM25Index':
        """BM25S index betöltése."""
        with open(input_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        model_dir = Path(index_data['model_dir'])
        chunk_ids_path = Path(index_data['chunk_ids_path'])
        stats_path = Path(index_data['stats_path'])

        # Index példány létrehozása
        index = cls()

        # Chunk ID-k betöltése
        if chunk_ids_path.exists():
            with open(chunk_ids_path, 'r', encoding='utf-8') as f:
                index.chunk_ids = json.load(f)

        # Statisztikák betöltése
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            index.total_docs = stats.get('total_docs', 0)
            index.avg_doc_length = stats.get('avg_doc_length', 0.0)
            index.doc_lengths = stats.get('doc_lengths', [])

        # BM25S model betöltése
        if model_dir.exists():
            try:
                index.bm25s_model = bm25s.BM25.load(str(model_dir), load_corpus=True)
                print("✅ BM25S modell betöltve")
            except Exception as e:
                print(f"BM25S modell betöltési hiba: {e}")
                index.bm25s_model = None

        return index

def main():
    """BM25S index építése."""
    import argparse

    parser = argparse.ArgumentParser(description='BM25S Index Builder')
    parser.add_argument('--overwrite', action='store_true', help='Felülírja a meglévő indexet')

    args = parser.parse_args()

    # Chunks fájl ellenőrzése
    if not config.CHUNKS_JSONL.exists():
        print(f"Hiba: Chunks fájl nem található: {config.CHUNKS_JSONL}")
        return

    # Meglévő index ellenőrzése
    if config.BM25_INDEX_PATH.exists() and not args.overwrite:
        print(f"BM25S index már létezik: {config.BM25_INDEX_PATH}")
        print("Használja a --overwrite kapcsolót az újjáépítéshez")
        return

    print("=== BM25S INDEX ÉPÍTÉS ===")
    print(f"Forrás: {config.CHUNKS_JSONL}")

    # Index építése
    bm25 = BM25Index(k1=config.BM25_K1, b=config.BM25_B)

    try:
        bm25.build_index(config.CHUNKS_JSONL)
        bm25.save(config.BM25_INDEX_PATH)

        print("✅ Sikeres index építés")
        print(f"Dokumentumok: {bm25.total_docs:,}")
        print(f"Átlag hossz: {bm25.avg_doc_length:.1f}")
        print(f"Index: {config.BM25_INDEX_PATH}")

    except Exception as e:
        print(f"❌ Hiba: {e}")
        return

if __name__ == '__main__':
    main()
