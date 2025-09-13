#!/usr/bin/env python3
"""
Hybrid Search Engine for CourtRankRL
Combines BM25 sparse and FAISS dense retrieval with RRF fusion.
"""

import json
import numpy as np
import faiss
import sys
from pathlib import Path
from typing import List, Tuple
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from src.data_loader.build_bm25_index import BM25Index

class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense search."""

    def __init__(self):
        self.bm25 = None
        self.faiss_index = None
        self.chunk_id_map = {}
        self._load_indexes()

    def _load_indexes(self):
        """Load BM25 and FAISS indexes."""
        # Load BM25
        if config.BM25_INDEX_PATH.exists():
            self.bm25 = BM25Index.load(config.BM25_INDEX_PATH)
            print(f"BM25 index loaded: {len(self.bm25.doc_ids)} documents")

        # Load FAISS
        if config.FAISS_INDEX_PATH.exists():
            self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
            print(f"FAISS index loaded: {self.faiss_index.ntotal} vectors")

            # Load chunk ID mapping
            if config.CHUNK_ID_MAP_PATH.exists():
                with open(config.CHUNK_ID_MAP_PATH, 'r', encoding='utf-8') as f:
                    self.chunk_id_map = json.load(f)

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query using Gemini."""
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        genai.configure(api_key=api_key) 
        result = genai.embed_content(
            model=config.MODEL_NAME,
            content=query,
            task_type="retrieval_query"
        )

        embedding = np.array(result['embedding'])
        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    def retrieve_candidates(self, query: str, top_k: int = 100) -> List[Tuple[str, float, str]]:
        """
        Retrieve candidates using hybrid BM25 + dense search with RRF fusion.

        Returns: List of (chunk_id, score, method)
        """
        candidates = []

        # BM25 retrieval
        if self.bm25:
            bm25_results = self.bm25.search(query, top_k=top_k)
            for chunk_id, score in bm25_results:
                candidates.append((chunk_id, score, 'bm25'))

        # Dense retrieval
        if self.faiss_index:
            query_embedding = self._embed_query(query)
            distances, indices = self.faiss_index.search(
                np.array([query_embedding]), top_k
            )

            for distance, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    chunk_id = self.chunk_id_map.get(str(idx))
                    if chunk_id:
                        # Convert distance to similarity score
                        score = 1.0 / (1.0 + distance)
                        candidates.append((chunk_id, score, 'dense'))

        # Remove duplicates and sort by score
        seen = set()
        unique_candidates = []
        for chunk_id, score, method in candidates:
            if chunk_id not in seen:
                unique_candidates.append((chunk_id, score, method))
                seen.add(chunk_id)

        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        return unique_candidates[:top_k]

    def fuse_results(self, candidates: List[Tuple[str, float, str]],
                    k: int = 60) -> List[Tuple[str, float]]:
        """
        Fuse BM25 and dense results using Reciprocal Rank Fusion.

        RRF score = sum(1/(k + rank)) for each method
        """
        rrf_scores = {}

        # Group by chunk_id
        by_chunk = {}
        for chunk_id, score, method in candidates:
            if chunk_id not in by_chunk:
                by_chunk[chunk_id] = []
            by_chunk[chunk_id].append((score, method))

        # Calculate RRF scores
        for chunk_id, results in by_chunk.items():
            rrf_score = 0.0

            # Sort results by score for ranking
            results.sort(key=lambda x: x[0], reverse=True)

            for rank, (score, method) in enumerate(results, 1):
                rrf_score += 1.0 / (k + rank)

            rrf_scores[chunk_id] = rrf_score

        # Sort by RRF score
        fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused_results

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Main retrieval method: get candidates -> fuse -> return top_k.
        """
        if not self.bm25 and not self.faiss_index:
            print("Error: No indexes loaded")
            return []

        # Get candidates
        candidates = self.retrieve_candidates(query, top_k=config.TOP_K_BASELINE)

        # Fuse results
        fused_results = self.fuse_results(candidates, k=config.RRF_K)

        return fused_results[:top_k]

def main():
    """Test hybrid retrieval."""
    retriever = HybridRetriever()

    query = "családi jogi ügy"
    results = retriever.retrieve(query, top_k=5)

    print(f"Query: {query}")
    print("Results:")
    for i, (chunk_id, score) in enumerate(results, 1):
        print(f"{i}. {chunk_id} (score: {score:.4f})")

if __name__ == '__main__':
    main()
