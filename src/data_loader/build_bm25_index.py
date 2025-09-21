#!/usr/bin/env python3
"""
BM25 Sparse Index Builder for CourtRankRL
Agents.md specifikáció alapján implementálva.

Főbb jellemzők:
- Hugging Face tokenizálás (AutoTokenizer)
- Minimal postings és statisztikai struktúra
- Streaming feldolgozás memória-hatékonyság érdekében
- JSON output formátum
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
from collections import defaultdict
import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config

class BM25Index:
    """BM25 sparse index with minimal structure per agents.md spec."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # Minimal structure per agents.md spec
        self.postings = defaultdict(list)  # term -> [(doc_id, term_freq), ...]
        self.doc_lengths = {}  # doc_id -> length
        self.doc_freqs = {}  # term -> document frequency
        self.avg_doc_length = 0
        self.total_docs = 0
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None

    def _ensure_tokenizer(self) -> PreTrainedTokenizerBase:
        """Betölti a HF tokenizálót egyszer és újrahasznosítja."""
        if self._tokenizer is None:
            load_dotenv()
            hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            if not hf_token:
                raise RuntimeError("Hugging Face token szükséges (.env: HUGGINGFACE_TOKEN)")

            tokenizer_name = getattr(config, 'BM25_TOKENIZER_NAME', None) or config.EMBEDDINGGEMMA_MODEL_NAME
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
                self._tokenizer = cast(PreTrainedTokenizerBase, tokenizer)
            except Exception as exc:
                raise RuntimeError(f"HF tokenizáló betöltési hiba ({tokenizer_name}): {exc}")

        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("A tokenizáló inicializálása sikertelen")
        return tokenizer

    def _tokenize(self, text: str) -> List[str]:
        """HF tokenizálás, kisbetűsítve a BM25 konzisztenciához."""
        tokenizer = self._ensure_tokenizer()
        raw_tokens = tokenizer.tokenize(text)
        if not raw_tokens:
            return []
        return [token.lower() for token in raw_tokens if token.strip()]

    def build_index_streaming(self, chunks_jsonl: Path):
        """Build minimal BM25 index from chunks JSONL file."""
        print(f"BM25 index építése: {chunks_jsonl}")

        # Sorok számolása külön passzal a progress bar-hoz
        with open(chunks_jsonl, 'r', encoding='utf-8') as counter_handle:
            total_lines = sum(1 for _ in counter_handle)

        if total_lines == 0:
            print("Nincs feldolgozandó chunk – üres a chunks.jsonl.")
            self.avg_doc_length = 0
            self.total_docs = 0
            return

        print(f"Feldolgozandó chunkok: {total_lines:,}")

        processed_docs = 0

        with open(chunks_jsonl, 'r', encoding='utf-8') as handle:
            iterator = tqdm.tqdm(handle, desc="Chunkok feldolgozása", total=total_lines)
            for line_num, raw_line in enumerate(iterator):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"JSON hiba {line_num + 1}. sorban: {exc}")
                    continue

                chunk_id = str(chunk.get('chunk_id', '')).strip()
                if not chunk_id:
                    continue

                text = chunk.get('text', '')
                if not isinstance(text, str):
                    text = str(text)

                tokens = self._tokenize(text)
                if not tokens:
                    continue

                doc_length = len(tokens)
                self.doc_lengths[chunk_id] = doc_length

                term_freqs: Dict[str, int] = defaultdict(int)
                for token in tokens:
                    term_freqs[token] += 1

                for term, freq in term_freqs.items():
                    self.postings[term].append((chunk_id, freq))
                    self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

                processed_docs += 1

        if processed_docs == 0:
            print("Nem sikerült érvényes chunkot feldolgozni.")
            self.avg_doc_length = 0
            self.total_docs = 0
            return

        # Átlagos dokumentum hossz számítása
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

        self.total_docs = len(self.doc_lengths)

        print(f"BM25 index kész: {self.total_docs:,} dokumentum, {len(self.doc_freqs):,} egyedi kifejezés")

    def build_index(self, chunks_jsonl: Path):
        """Legacy method for backward compatibility."""
        self.build_index_streaming(chunks_jsonl)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """BM25 keresés minimal struktúrával."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        if self.total_docs == 0 or self.avg_doc_length <= 0:
            return []

        # Dokumentum pontszámok gyűjtése
        doc_scores = defaultdict(float)

        for term in query_tokens:
            if term in self.postings:
                # IDF számítása
                df = self.doc_freqs[term]
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5))

                # Minden dokumentum pontszámítása ebben a term-ben
                for doc_id, term_freq in self.postings[term]:
                    doc_length = self.doc_lengths[doc_id]

                    # BM25 score számítása
                    numerator = term_freq * (self.k1 + 1)
                    denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                    tf_score = numerator / denominator

                    doc_scores[doc_id] += idf * tf_score

        # Rendezés és top-k kiválasztás
        sorted_scores = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    def save(self, output_path: Path):
        """BM25 index mentése JSON formátumban - minimal struktúra."""
        # Minimal index struktúra per agents.md spec
        index_data = {
            'k1': self.k1,
            'b': self.b,
            'postings': dict(self.postings),  # term -> [(doc_id, term_freq), ...]
            'doc_lengths': self.doc_lengths,  # doc_id -> length
            'doc_freqs': self.doc_freqs,      # term -> document frequency
            'avg_doc_length': self.avg_doc_length,
            'total_docs': self.total_docs
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, separators=(',', ':'))

        print(f"BM25 index mentve: {output_path}")

    @classmethod
    def load(cls, input_path: Path) -> 'BM25Index':
        """BM25 index betöltése JSON fájlból."""
        with open(input_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        index = cls(k1=index_data['k1'], b=index_data['b'])
        index.postings = defaultdict(list, index_data['postings'])
        index.doc_lengths = index_data['doc_lengths']
        index.doc_freqs = index_data['doc_freqs']
        index.avg_doc_length = index_data['avg_doc_length']
        index.total_docs = index_data['total_docs']

        return index

def main():
    """BM25 index építése chunks JSONL-ból per agents.md spec."""
    import argparse

    parser = argparse.ArgumentParser(description='BM25 Index Builder for CourtRankRL')
    parser.add_argument('--overwrite', action='store_true',
                       help='Felülírja a meglévő indexet')

    args = parser.parse_args()

    # Ellenőrzés: chunks fájl létezik-e
    if not config.CHUNKS_JSONL.exists():
        print(f"Hiba: Chunks fájl nem található: {config.CHUNKS_JSONL}")
        return

    # Ellenőrzés: már létezik-e index
    if config.BM25_INDEX_PATH.exists() and not args.overwrite:
        print(f"BM25 index már létezik: {config.BM25_INDEX_PATH}")
        print("Használja a --overwrite kapcsolót az újjáépítéshez")
        return

    print("=== BM25 INDEX ÉPÍTÉS ===")
    print(f"Chunks forrás: {config.CHUNKS_JSONL}")

    # BM25 index építése
    bm25 = BM25Index(k1=config.BM25_K1, b=config.BM25_B)

    try:
        # Streaming feldolgozás
        bm25.build_index_streaming(config.CHUNKS_JSONL)

        # Index mentése
        print(f"\nIndex mentése: {config.BM25_INDEX_PATH}")
        bm25.save(config.BM25_INDEX_PATH)

        print("\n=== SIKERES INDEX ÉPÍTÉS ===")
        print(f"Dokumentumok száma: {bm25.total_docs:,}")
        print(f"Egyedi kifejezések: {len(bm25.doc_freqs):,}")
        print(f"Átlagos dokumentum hossz: {bm25.avg_doc_length:.1f} token")
        print(f"Index fájl: {config.BM25_INDEX_PATH}")

    except Exception as e:
        print(f"\n❌ HIBA: {e}")
        return

if __name__ == '__main__':
    main()
