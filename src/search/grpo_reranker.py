#!/usr/bin/env python3
"""
CourtRankRL GRPO Slate Preparation
Based on agents.md specification: Cloud-only GRPO training.

Főbb jellemzők:
- Slate preparation for cloud GRPO training (RunPod)
- Baseline candidate retrieval and feature extraction
- JSONL export for cloud notebook consumption
- Local execution: BASELINE ONLY (no 4B model inference)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from src.scripts.hybrid_retrieval import HybridRetriever


# Slate preparation configuration (chunk-based)
GRPO_SLATE_SIZE = getattr(config, "GRPO_SLATE_SIZE", 20)

def prepare_training_slates(
    retriever: HybridRetriever, 
    qrels: Dict[str, Dict[str, int]], 
    output_path: Optional[Path] = None
) -> List[Dict]:
    """
    Prepare baseline candidate slates for GRPO training.
    Agents.md: serializes baseline candidate slates into JSONL for GRPO training.
    
    **CHUNK-BASED**: Uses top-scoring chunks (not doc-level aggregation) to provide
    relevant context to the GRPO model. Each slate contains the most relevant chunks
    as determined by BM25 and FAISS retrieval.
    
    Args:
        retriever: HybridRetriever instance
        qrels: Relevance judgments (query_id -> doc_id -> relevance)
        output_path: Optional path to save slates as JSONL (for cloud training)
                     If None, returns in-memory list only (for local training)
    
    Returns:
        List of slate dictionaries
    """
    print("Slate preparation indítása (chunk-based)...")

    slates = []

    for query_id, relevance_data in qrels.items():
        try:
            # Get baseline candidates (chunk-level!)
            retriever.retrieve_candidates(query_id, top_k=GRPO_SLATE_SIZE * 2)
            bm25_chunk_results = retriever.get_last_chunk_scores("bm25")
            dense_chunk_results = retriever.get_last_chunk_scores("dense")

            if not bm25_chunk_results:
                continue

            # Merge chunk-level scores
            all_chunk_candidates = {}
            for chunk_id, score in bm25_chunk_results:
                all_chunk_candidates[chunk_id] = {"chunk_id": chunk_id, "bm25_score": score}

            for chunk_id, score in dense_chunk_results:
                if chunk_id in all_chunk_candidates:
                    all_chunk_candidates[chunk_id]["faiss_score"] = score
                else:
                    all_chunk_candidates[chunk_id] = {"chunk_id": chunk_id, "faiss_score": score}

            # Sort chunks by combined score and take top slate_size
            sorted_chunk_candidates = sorted(
                all_chunk_candidates.values(),
                key=lambda x: x.get("bm25_score", 0) + x.get("faiss_score", 0),
                reverse=True
            )[:GRPO_SLATE_SIZE]

            # Ensure metadata loaded
            retriever._ensure_metadata_loaded()

            # Add metadata and relevance labels
            for candidate in sorted_chunk_candidates:
                chunk_id = candidate["chunk_id"]
                
                # Get doc_id from chunk_id
                doc_id = retriever._chunk_id_to_doc_id(chunk_id)
                candidate["doc_id"] = doc_id
                
                # Relevance from qrels (doc-level)
                candidate["relevance"] = relevance_data.get(doc_id, 0)
                candidate["query_id"] = query_id
                
                # Metadata from chunk (court, domain, year, FULL chunk text)
                chunk_meta = retriever.chunk_metadata.get(chunk_id, {}) if retriever.chunk_metadata else {}
                candidate["court"] = chunk_meta.get("court", "")
                candidate["domain"] = chunk_meta.get("domain", "")
                candidate["year"] = chunk_meta.get("year", "")
                
                # FULL chunk text (not preview!) - this is the key improvement
                # Load from chunks.jsonl to get complete text
                candidate["text"] = chunk_meta.get("text_preview", "")  # fallback to 500 char preview

            # Load full chunk texts from chunks.jsonl (one-time read per query)
            chunk_texts = retriever._load_chunk_texts(
                [c["chunk_id"] for c in sorted_chunk_candidates]
            )
            for candidate in sorted_chunk_candidates:
                chunk_id = candidate["chunk_id"]
                if chunk_id in chunk_texts:
                    candidate["text"] = chunk_texts[chunk_id]

            slates.append({
                "query_id": query_id,
                "slate": sorted_chunk_candidates
            })

        except Exception as e:
            print(f"Hiba a {query_id} slate preparation során: {e}")
            continue

    # Optionally save to JSONL
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for slate in slates:
                f.write(json.dumps(slate, ensure_ascii=False) + '\n')
        print(f"Slates exportálva: {len(slates)} slate -> {output_path}")
    else:
        print(f"Slates előkészítve: {len(slates)} slate (memóriában)")

    return slates

# Backward compatibility wrapper
def export_slates_for_grpo_training(
    retriever: HybridRetriever, 
    qrels: Dict[str, Dict[str, int]], 
    output_path: Path
) -> List[Dict]:
    """Backward compatibility wrapper for prepare_training_slates()."""
    return prepare_training_slates(retriever, qrels, output_path)

def load_qrels(qrels_file: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels file for training.
    
    Agents.md specifies: qrels file must contain doc_id values (NOT chunk_id).
    Format: tab-separated with header `query_id\tdoc_id\trelevance`
    """
    qrels = {}

    if not qrels_file.exists():
        return qrels

    with open(qrels_file, 'r', encoding='utf-8') as f:
        # Skip header line if present
        first_line = f.readline().strip()
        if not first_line.startswith('query_id'):
            # Not a header, process as data
            parts = first_line.split('\t')
            if len(parts) >= 3:
                query_id = parts[0]
                doc_id = parts[1]  # Already doc_id as per agents.md
                relevance = int(parts[2])
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = relevance
        
        # Process remaining lines
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id = parts[0]
                doc_id = parts[1]  # Already doc_id as per agents.md
                relevance = int(parts[2])

                if query_id not in qrels:
                    qrels[query_id] = {}

                qrels[query_id][doc_id] = relevance

    return qrels

if __name__ == '__main__':
    # This module is for slate preparation only.
    # For actual GRPO training, use notebooks/grpo_train_runpod.ipynb on cloud.
    print("CourtRankRL GRPO Slate Preparation")
    print("====================================")
    print("This module prepares slates for cloud GRPO training.")
    print("For training: use grpo_train_runpod.ipynb on RunPod")
    print("For querying: use 'uv run courtrankrl query' (baseline only)")
