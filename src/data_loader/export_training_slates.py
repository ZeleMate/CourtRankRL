#!/usr/bin/env python3
"""
CourtRankRL - Export training slates for GRPO RL training

Agents.md specifikáció alapján:
- Baseline candidate slates szerializálása JSONL formátumba
- Chunk text + metadata + scores
- IDs szinkronizálva chunks.jsonl és qrels-szel
- Cloud GRPO notebook ingestálható formátum
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from src.search.hybrid_search import HybridRetriever


def load_chunks_metadata(chunks_file: Path) -> Dict[str, Dict]:
    """Load chunks metadata by chunk_id."""
    chunks_data = {}

    if not chunks_file.exists():
        print(f"Figyelmeztetés: Chunks fájl nem található: {chunks_file}")
        return chunks_data

    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                chunk_data = json.loads(line.strip())
                chunk_id = chunk_data.get('chunk_id')
                if chunk_id:
                    chunks_data[chunk_id] = chunk_data
            except json.JSONDecodeError:
                continue

    print(f"Betöltött chunk metaadatok: {len(chunks_data)}")
    return chunks_data


def load_qrels(qrels_file: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels file - agents.md formátum: query_id\tchunk_id\trelevance."""
    qrels = {}

    if not qrels_file.exists():
        print(f"Figyelmeztetés: Qrels fájl nem található: {qrels_file}")
        return qrels

    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id = parts[0]
                chunk_id = parts[1]  # Note: agents.md spec uses chunk_id in column 2
                relevance = int(parts[2])

                if query_id not in qrels:
                    qrels[query_id] = {}

                # Convert chunk_id to doc_id for consistency with other parts
                doc_id = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
                qrels[query_id][doc_id] = relevance

    print(f"Betöltött qrels: {len(qrels)} query")
    return qrels


def extract_slate_features(
    query_id: str,
    retriever: HybridRetriever,
    chunks_data: Dict[str, Dict],
    qrels: Dict[str, Dict[str, int]]
) -> Optional[Dict]:
    """Extract slate features for a single query."""
    try:
        # Retrieve candidates
        retriever.retrieve_candidates(query_id, top_k=config.TOP_K_BASELINE)

        bm25_results = retriever.get_last_doc_scores("bm25")
        dense_results = retriever.get_last_doc_scores("dense")

        if not bm25_results and not dense_results:
            return None

        # Merge and deduplicate candidates
        all_candidates = {}
        for doc_id, score in bm25_results:
            all_candidates[doc_id] = {
                'bm25_score': score,
                'dense_score': dense_results.get(doc_id, 0.0) if dense_results else 0.0
            }

        for doc_id, score in dense_results:
            if doc_id not in all_candidates:
                all_candidates[doc_id] = {
                    'bm25_score': bm25_results.get(doc_id, 0.0) if bm25_results else 0.0,
                    'dense_score': score
                }

        # Sort by combined score
        sorted_candidates = sorted(
            all_candidates.items(),
            key=lambda x: x[1]['bm25_score'] + x[1]['dense_score'],
            reverse=True
        )

        # Limit to group size
        group_size = min(len(sorted_candidates), config.GRPO_GROUP_SIZE)
        slate_candidates = sorted_candidates[:group_size]

        # Build slate
        slate = {
            'query_id': query_id,
            'candidates': []
        }

        for doc_id, scores in slate_candidates:
            # Find relevant chunks for this doc_id
            relevant_chunks = []
            for chunk_id, chunk_data in chunks_data.items():
                if chunk_id.startswith(doc_id + '_'):
                    relevant_chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_data.get('text', '')[:config.EMBEDDING_MAX_LENGTH],  # Trim to config length
                        'metadata': {
                            'court': chunk_data.get('court', ''),
                            'domain': chunk_data.get('domain', ''),
                            'year': chunk_data.get('year', ''),
                            'doc_id': chunk_data.get('doc_id', doc_id)
                        }
                    })

            if not relevant_chunks:
                # Fallback: create a placeholder chunk
                relevant_chunks = [{
                    'chunk_id': f"{doc_id}_placeholder_0",
                    'text': f"Document {doc_id} - no detailed chunks available",
                    'metadata': {
                        'court': 'unknown',
                        'domain': 'unknown',
                        'year': 'unknown',
                        'doc_id': doc_id
                    }
                }]

            slate['candidates'].append({
                'doc_id': doc_id,
                'bm25_score': scores['bm25_score'],
                'dense_score': scores['dense_score'],
                'chunks': relevant_chunks,
                'relevance': qrels.get(query_id, {}).get(doc_id, 0)
            })

        return slate

    except Exception as e:
        print(f"Hiba slate készítésekor {query_id}: {e}")
        return None


def export_training_slates(
    output_file: Path,
    retriever: HybridRetriever,
    chunks_data: Dict[str, Dict],
    qrels: Dict[str, Dict[str, int]]
):
    """Export all training slates to JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Training slate export indítása...")
    print(f"Output: {output_file}")

    exported_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for query_id in qrels.keys():
            slate = extract_slate_features(query_id, retriever, chunks_data, qrels)
            if slate:
                f.write(json.dumps(slate, ensure_ascii=False) + '\n')
                exported_count += 1

            if exported_count % 10 == 0 and exported_count > 0:
                print(f"Exportálva: {exported_count} slates...")

    print(f"Exportálás kész: {exported_count} slates mentve")


def main():
    """Main export function."""
    print("=== TRAINING SLATES EXPORT ===")

    # Load data
    chunks_data = load_chunks_metadata(config.CHUNKS_JSONL)
    qrels = load_qrels(config.BASELINE_QRELS_FILE)

    if not qrels:
        print("Hiba: Nincs qrels adat az exportáláshoz")
        return

    if not chunks_data:
        print("Figyelmeztetés: Nincs chunks metaadat - placeholder chunkok lesznek használva")

    # Initialize retriever
    retriever = HybridRetriever()

    # Export slates
    export_training_slates(
        config.GRPO_SLATE_EXPORT_PATH,
        retriever,
        chunks_data,
        qrels
    )

    print("=== EXPORT SIKERES ===")


if __name__ == '__main__':
    main()
