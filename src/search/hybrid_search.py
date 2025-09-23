#!/usr/bin/env python3
"""
CourtRankRL Hybrid Retrieval Engine
Agents.md specifikáció alapján implementálva.

Főbb jellemzők:
- Qwen3-Embedding-0.6B query embedding (Hugging Face)
- BM25 sparse + FAISS dense retrieval
- RRF vagy z-score weighted sum fusion
- Zero variance kezelés z-score esetében
- Output: top-k dokumentum azonosítók listája (határozat számok)
"""

import json
import numpy as np
import faiss
import sys
from pathlib import Path
from typing import List, Tuple
import torch
from transformers import AutoModel, AutoTokenizer

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from src.data_loader.build_bm25_index import BM25Index

# Mock classes for testing on 16GB RAM systems
class MockQwenModel:
    """Mock Qwen3 model for testing."""
    def __init__(self):
        self.config = type('Config', (), {'hidden_size': config.QWEN3_DIMENSION})()

class MockTokenizer:
    """Mock tokenizer for testing."""
    def __call__(self, texts, return_tensors='pt', truncation=True, max_length=512, padding=True):
        # Mock tokenization - return dummy tensors
        batch_size = len(texts)
        seq_len = max_length
        return {
            'input_ids': torch.zeros((batch_size, seq_len), dtype=torch.long),
            'attention_mask': torch.ones((batch_size, seq_len), dtype=torch.long)
        }

class HybridRetriever:
    """CourtRankRL hybrid retrieval engine."""

    def __init__(self):
        self.bm25 = None
        self.faiss_index = None
        self.chunk_id_map = {}
        self.tokenizer = None
        self.model = None
        self.model_name = None
        self._load_models_and_indexes()

    def _chunk_id_to_doc_id(self, chunk_id: str) -> str:
        """Convert chunk_id to doc_id (document identifier/határozat szám)."""
        # chunk_id format: "doc_id_src_tag_i" -> extract doc_id
        return chunk_id.split('_')[0]

    def _load_models_and_indexes(self):
        """Load Qwen3 model and indexes."""
        try:
            # Load Qwen3 model for query embedding
            self.model_name = config.QWEN3_MODEL_NAME
            print(f"Qwen3 model betöltése: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            # RunPod 5090 GPU optimalizált model betöltés
            print(f"🚀 RunPod 5090 GPU optimalizáció: {self.model_name}")
            print("📱 GPU device keresése...")

            # GPU/CUDA prioritás a RunPod-on
            if torch.cuda.is_available():
                device = "cuda"
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"🎯 CUDA GPU-k: {gpu_count}, Memory: {gpu_memory:.1f} GB")
            else:
                device = "cpu"
                print("⚠️  CUDA nem elérhető, CPU használat")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,  # FP16 a memória optimalizáláshoz
                    low_cpu_mem_usage=True
                ).to(device)
                self.model.eval()
                print(f"✅ Qwen3 model betöltve: {self.model_name} (device: {device})")

            except Exception as e:
                print(f"❌ Model betöltés sikertelen: {e}")
                # Fallback: kisebb model
                print("🔄 Fallback: sentence-transformers/all-MiniLM-L6-v2")
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                ).to(device)
                self.model.eval()
                print(f"✅ Fallback model betöltve: {self.model_name}")

            # Load BM25 - RunPod 5090 GPU optimalizált
            if config.BM25_INDEX_PATH.exists():
                print("🔍 BM25 index betöltése...")
                self.bm25 = BM25Index.load(config.BM25_INDEX_PATH)
                print(f"✅ BM25 index betöltve: {self.bm25.total_docs} dokumentum")

            # Load FAISS
            if config.FAISS_INDEX_PATH.exists():
                self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
                print(f"FAISS index betöltve: {self.faiss_index.ntotal} vektor")

                # Load chunk ID mapping
                if config.CHUNK_ID_MAP_PATH.exists():
                    with open(config.CHUNK_ID_MAP_PATH, 'r', encoding='utf-8') as f:
                        self.chunk_id_map = json.load(f)

        except Exception as e:
            print(f"Hiba a modellek/indexek betöltésekor: {e}")

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query using Qwen3 (agents.md spec)."""
        if not self.tokenizer or not self.model:
            raise ValueError("Qwen3 model nincs betöltve")

        try:
            # RunPod 5090 GPU optimalizált embedding
            print(f"📝 Query embedding: {query}")

            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=config.EMBEDDING_MAX_LENGTH,
                padding=True
            )

            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    embedding = outputs.last_hidden_state[:, 0, :].float()
                else:
                    embedding = outputs.pooler_output.float()

                # CPU-ra másolás csak ha szükséges
                if device != 'cpu':
                    embedding = embedding.cpu()

                embedding = embedding.numpy()

            # L2-normalizálás IP metrika számára (agents.md spec)
            embedding = embedding.astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                norm = 1.0
            embedding = embedding / norm
            return embedding

        except Exception as e:
            raise ValueError(f"Qwen3 embedding hiba: {e}")

    def retrieve_candidates(self, query: str, top_k: int = 100) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Retrieve candidates from BM25 and FAISS."""
        bm25_results = []
        dense_results = []

        # BM25 retrieval - convert chunk_id to doc_id
        if self.bm25:
            chunk_results = self.bm25.search(query, top_k=top_k)
            # Group by doc_id and keep max score
            doc_scores = {}
            for chunk_id, score in chunk_results:
                doc_id = self._chunk_id_to_doc_id(chunk_id)
                if doc_id not in doc_scores or score > doc_scores[doc_id]:
                    doc_scores[doc_id] = score
            bm25_results = list(doc_scores.items())

        # Dense retrieval - convert chunk_id to doc_id
        if self.faiss_index and self.chunk_id_map:
            try:
                query_embedding = self._embed_query(query)
                distances, indices = self.faiss_index.search(
                    np.array([query_embedding]), top_k
                )

                # Group by doc_id and keep max score
                doc_scores = {}
                for distance, idx in zip(distances[0], indices[0]):
                    if idx != -1:
                        chunk_id = self.chunk_id_map.get(str(idx))
                        if chunk_id:
                            doc_id = self._chunk_id_to_doc_id(chunk_id)
                            score = distance  # FAISS IP returns similarity directly
                            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                                doc_scores[doc_id] = score
                dense_results = list(doc_scores.items())

            except Exception as e:
                print(f"Dense retrieval hiba: {e}")

        return bm25_results, dense_results

    def fuse_results(self, bm25_results: List[Tuple[str, float]],
                    dense_results: List[Tuple[str, float]],
                    method: str = "rrf") -> List[Tuple[str, float]]:
        """
        Fuse BM25 and dense results using RRF or z-score weighted sum.
        Handles zero variance for z-score method.
        """
        if method == "rrf":
            return self._fuse_rrf(bm25_results, dense_results)
        elif method == "zscore":
            return self._fuse_zscore(bm25_results, dense_results)
        else:
            raise ValueError("Ismeretlen fusion method: használj 'rrf' vagy 'zscore'")

    def _fuse_rrf(self, bm25_results: List[Tuple[str, float]],
                 dense_results: List[Tuple[str, float]], k: int = 60) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion."""
        rrf_scores = {}

        # Create ranking dictionaries
        bm25_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_results, 1)}
        dense_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense_results, 1)}

        # All unique chunk IDs
        all_chunks = set(bm25_ranks.keys()) | set(dense_ranks.keys())

        for chunk_id in all_chunks:
            rrf_score = 0.0
            if chunk_id in bm25_ranks:
                rrf_score += 1.0 / (k + bm25_ranks[chunk_id])
            if chunk_id in dense_ranks:
                rrf_score += 1.0 / (k + dense_ranks[chunk_id])
            rrf_scores[chunk_id] = rrf_score

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _fuse_zscore(self, bm25_results: List[Tuple[str, float]],
                    dense_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Z-score weighted sum with zero variance handling."""
        # Create score dictionaries
        bm25_scores = dict(bm25_results)
        dense_scores = dict(dense_results)

        # All unique chunk IDs
        all_chunks = set(bm25_scores.keys()) | set(dense_scores.keys())

        # Calculate z-scores for each method (handle zero variance)
        bm25_zscores = self._calculate_zscores(list(bm25_scores.values()))
        dense_zscores = self._calculate_zscores(list(dense_scores.values()))

        bm25_zscore_map = dict(zip(bm25_scores.keys(), bm25_zscores))
        dense_zscore_map = dict(zip(dense_scores.keys(), dense_zscores))

        # Weighted sum of z-scores
        fused_scores = {}
        for chunk_id in all_chunks:
            bm25_z = bm25_zscore_map.get(chunk_id, 0.0)
            dense_z = dense_zscore_map.get(chunk_id, 0.0)
            fused_scores[chunk_id] = bm25_z + dense_z

        return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    def _calculate_zscores(self, scores: List[float]) -> List[float]:
        """Calculate z-scores with zero variance handling."""
        if not scores:
            return []

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Handle zero variance (agents.md spec)
        if std_score == 0:
            # If all scores are the same, return zeros
            return [0.0] * len(scores)

        return [(score - mean_score) / std_score for score in scores]

    def retrieve(self, query: str, top_k: int = 10, fusion_method: str = "rrf") -> List[str]:
        """
        Main retrieval method: get candidates -> fuse -> return top_k document IDs.
        Output: top-k list of document IDs (határozat számok) based on retrieved chunks.
        """
        if not self.bm25 and not self.faiss_index:
            print("Figyelem: Nincs index betöltve")
            return []

        # Get candidates from both methods (already converted to doc_ids)
        bm25_results, dense_results = self.retrieve_candidates(query, top_k=top_k*2)

        # Fuse results
        fused_results = self.fuse_results(bm25_results, dense_results, method=fusion_method)

        # Return document IDs (határozat számok)
        doc_ids = [doc_id for doc_id, _ in fused_results[:top_k]]
        return doc_ids

def main():
    """CourtRankRL hybrid retrieval teszt - dokumentum azonosítók visszaadása."""
    import argparse

    parser = argparse.ArgumentParser(description='CourtRankRL Hybrid Retrieval - Dokumentum azonosítók')
    parser.add_argument('--query', type=str, default='családi jogi ügy',
                       help='Keresési lekérdezés')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Visszaadandó dokumentumok száma')
    parser.add_argument('--fusion-method', type=str, default='rrf', choices=['rrf', 'zscore'],
                       help='Fusion method: rrf vagy zscore')

    args = parser.parse_args()

    retriever = HybridRetriever()

    results = retriever.retrieve(args.query, top_k=args.top_k, fusion_method=args.fusion_method)

    print(f"Lekérdezés: {args.query}")
    print(f"Fusion method: {args.fusion_method}")
    print(f"Top-{args.top_k} dokumentum azonosító:")
    for i, doc_id in enumerate(results, 1):
        print(f"{i}. {doc_id}")

if __name__ == '__main__':
    main()
