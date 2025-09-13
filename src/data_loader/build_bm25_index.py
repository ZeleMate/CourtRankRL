#!/usr/bin/env python3
"""
BM25 Sparse Index Builder for CourtRankRL
Implements BM25 algorithm for efficient text retrieval.
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict, Counter
import re
import tqdm

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config

class BM25Index:
    """BM25 sparse index implementation with memory-efficient processing."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.doc_freqs = defaultdict(int)
        self.term_freqs = []  # List of term frequency dicts per document
        self.doc_ids = []
        self.avg_doc_length = 0
        self.total_docs = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split on whitespace
        return re.findall(r'\b\w+\b', text.lower())

    def build_index_streaming(self, chunks_jsonl: Path, batch_size: int = 10000):
        """Build BM25 index from chunks JSONL file using streaming processing."""
        print(f"Building BM25 index from {chunks_jsonl} (streaming mode)")

        # Count total lines first
        total_lines = sum(1 for _ in open(chunks_jsonl, 'r', encoding='utf-8'))
        print(f"Total chunks to process: {total_lines:,}")

        # Process in batches to save memory
        processed_docs = 0
        duplicate_count = 0
        seen_chunk_ids = set()

        with open(chunks_jsonl, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm.tqdm(f, desc="Processing chunks", total=total_lines)):
                try:
                    chunk = json.loads(line.strip())
                    chunk_id = chunk['chunk_id']
                    text = chunk['text']

                    # Skip duplicate chunks
                    if chunk_id in seen_chunk_ids:
                        duplicate_count += 1
                        continue

                    seen_chunk_ids.add(chunk_id)

                    # Tokenize
                    tokens = self._tokenize(text)

                    if not tokens:
                        continue

                    # Store document info
                    self.doc_ids.append(chunk_id)
                    doc_length = len(tokens)
                    self.doc_lengths.append(doc_length)

                    # Term frequencies for this document
                    term_freq = Counter(tokens)
                    self.term_freqs.append(term_freq)

                    # Update document frequencies
                    for term in set(tokens):
                        self.doc_freqs[term] += 1

                    processed_docs += 1

                except json.JSONDecodeError as e:
                    print(f"JSON decode error at line {line_num + 1}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing chunk at line {line_num + 1}: {e}")
                    continue

        # Calculate average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        self.total_docs = len(self.doc_ids)

        print(f"BM25 index built: {self.total_docs:,} unique documents, {len(self.doc_freqs):,} unique terms")
        if duplicate_count > 0:
            print(f"Skipped {duplicate_count:,} duplicate chunks")

    def build_index(self, chunks_jsonl: Path):
        """Legacy method for backward compatibility."""
        self.build_index_streaming(chunks_jsonl)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 scoring."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []

        for i, doc_id in tqdm.tqdm(enumerate(self.doc_ids), desc="Searching"):
            score = 0.0
            doc_length = self.doc_lengths[i]
            term_freq = self.term_freqs[i]

            for term in query_tokens:
                if term in term_freq:
                    # BM25 scoring formula
                    tf = term_freq[term]
                    df = self.doc_freqs.get(term, 0)
                    if df == 0:
                        continue

                    idf = math.log((len(self.doc_ids) - df + 0.5) / (df + 0.5))

                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)

                    score += idf * (numerator / denominator)

            if score > 0:
                scores.append((doc_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, output_path: Path):
        """Save BM25 index to JSON file."""
        index_data = {
            'k1': self.k1,
            'b': self.b,
            'doc_lengths': self.doc_lengths,
            'doc_freqs': dict(self.doc_freqs),
            'term_freqs': [dict(tf) for tf in self.term_freqs],
            'doc_ids': self.doc_ids,
            'avg_doc_length': self.avg_doc_length
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

        print(f"BM25 index saved to {output_path}")

    @classmethod
    def load(cls, input_path: Path) -> 'BM25Index':
        """Load BM25 index from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        index = cls(k1=index_data['k1'], b=index_data['b'])
        index.doc_lengths = index_data['doc_lengths']
        index.doc_freqs = defaultdict(int, index_data['doc_freqs'])
        index.term_freqs = [Counter(tf) for tf in index_data['term_freqs']]
        index.doc_ids = index_data['doc_ids']
        index.avg_doc_length = index_data['avg_doc_length']

        return index

def main():
    """Build BM25 index from chunks with optimized memory usage."""
    import argparse

    parser = argparse.ArgumentParser(description='BM25 Index Builder')
    parser.add_argument('--batch-size', type=int, default=50000,
                       help='Batch size for processing (default: 50000)')
    parser.add_argument('--memory-limit', action='store_true',
                       help='Use memory-efficient streaming mode')

    args = parser.parse_args()

    if not config.CHUNKS_JSONL.exists():
        print(f"Error: Chunks file not found: {config.CHUNKS_JSONL}")
        return

    print("=== BM25 INDEX ÉPÍTÉS ===")
    print(f"Chunks fájl: {config.CHUNKS_JSONL}")

    # Build index with streaming
    bm25 = BM25Index(k1=config.BM25_K1, b=config.BM25_B)

    try:
        if args.memory_limit:
            print("Memory-efficient streaming mode aktiválva")
            bm25.build_index_streaming(config.CHUNKS_JSONL, batch_size=args.batch_size)
        else:
            print("Standard mode - használja a --memory-limit kapcsolót nagy fájloknál")
            bm25.build_index(config.CHUNKS_JSONL)

        # Save index
        print(f"\\nIndex mentése: {config.BM25_INDEX_PATH}")
        bm25.save(config.BM25_INDEX_PATH)

        print("\\n=== SIKERES INDEX ÉPÍTÉS ===")
        print(f"Dokumentumok: {bm25.total_docs:,}")
        print(f"Unika kifejezések: {len(bm25.doc_freqs):,}")
        print(f"Átlagos dokumentum hossz: {bm25.avg_doc_length:.1f} token")

    except MemoryError:
        print("\\n❌ MEMORY ERROR - Használja a --memory-limit kapcsolót!")
        print("Példa: python build_bm25_index.py --memory-limit --batch-size 10000")
    except Exception as e:
        print(f"\\n❌ HIBA: {e}")
        print("Próbálja újra memory-efficient módban.")

if __name__ == '__main__':
    main()
