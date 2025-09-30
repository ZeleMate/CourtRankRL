#!/usr/bin/env python3
"""
CourtRankRL Hybrid Retrieval Engine
Agents.md specifikáció alapján implementálva.

Főbb jellemzők:
- google/embeddinggemma-300m query embedding (HF, MPS támogatás)
- BM25S sparse + FAISS dense retrieval token cache metaadatokkal
- RRF vagy z-score weighted sum fusion (zero variance védelemmel)
- Dokumentum- és chunk-szintű score aggregáció GRPO feature exporthoz
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import faiss
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from src.data_loader.build_bm25_index import BM25Index


class HybridRetriever:
    """CourtRankRL hybrid retrieval engine BM25S + FAISS integrációval."""

    def __init__(self) -> None:
        self.device = self._detect_device()
        self.model_name = getattr(config, "EMBEDDING_GEMMA_MODEL_NAME", "google/embeddinggemma-300m")
        self.tokenizer: Optional[Any] = None
        self.model: Optional[nn.Module] = None

        self.bm25: Optional[BM25Index] = None
        self.bm25_stats: Dict[str, float] = {}
        self.faiss_index = None
        self.chunk_id_map: Dict[str, str] = {}
        self.doc_id_map: Dict[str, str] = {}
        self.chunk_to_doc: Dict[str, str] = {}
        self.known_chunk_ids: Set[str] = set()
        
        # Chunk metadata (agents.md: court, domain, year for GRPO)
        self.chunk_metadata: Dict[str, Dict[str, Any]] = {}

        # Lefutások közben tárolt részletes score-ok (GRPO feature export)
        self.last_chunk_scores: Dict[str, List[Tuple[str, float]]] = {
            "bm25": [],
            "dense": [],
        }
        self.last_doc_scores: Dict[str, List[Tuple[str, float]]] = {
            "bm25": [],
            "dense": [],
            "fused": [],
        }

        self._load_models_and_indexes()

    @staticmethod
    def _strip_chunk_suffix(chunk_id: str) -> str:
        if "_" not in chunk_id:
            return chunk_id
        base, suffix = chunk_id.rsplit("_", 1)
        if suffix.isdigit():
            return base
        return chunk_id

    def _chunk_id_to_doc_id(self, chunk_id: str) -> str:
        if chunk_id in self.chunk_to_doc:
            return self.chunk_to_doc[chunk_id]

        if chunk_id in self.known_chunk_ids:
            doc_id = self._strip_chunk_suffix(chunk_id)
            self.chunk_to_doc[chunk_id] = doc_id
            return doc_id

        if "_" in chunk_id:
            base, suffix = chunk_id.rsplit("_", 1)
            if suffix.isdigit():
                first_chunk_candidate = f"{base}_0"
                if first_chunk_candidate in self.known_chunk_ids:
                    self.chunk_to_doc[chunk_id] = base
                    return base

        return chunk_id

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_models_and_indexes(self) -> None:
        try:
            self._load_transformer_model()
            self._load_bm25_index()
            self._load_faiss_index()
            self._load_id_maps()
        except Exception as exc:
            print(f"❌ Hiba a komponensek betöltésekor: {exc}")
            raise

    def _load_transformer_model(self) -> None:
        print(f"🔧 Modell betöltése ({self.device}) – {self.model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                cache_dir="/tmp/huggingface_cache",
            )
            self.tokenizer = tokenizer

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if self.device == "cuda":
                model_kwargs.update(
                    {
                        "dtype": torch.float16,
                        "device_map": "auto",
                        "attn_implementation": "flash_attention_2",
                    }
                )
            else:
                model_kwargs.update(
                    {
                        "dtype": torch.float32,
                        "device_map": None,
                        "attn_implementation": "eager",
                    }
                )

            model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
            model_module = cast(nn.Module, model)
            if self.device in {"cuda", "mps"}:
                model_module = model_module.to(self.device)
            model_module.eval()
            self.model = model_module
            print("✅ EmbeddingGemma modell betöltve")
        except Exception as exc:
            print(f"❌ EmbeddingGemma betöltési hiba: {exc}")
            fallback = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"🔄 Tartalék modell: {fallback}")
            self.model_name = fallback
            tokenizer = AutoTokenizer.from_pretrained(
                fallback,
                    trust_remote_code=True,
                cache_dir="/tmp/huggingface_cache",
            )
            self.tokenizer = tokenizer
            model = AutoModel.from_pretrained(fallback, trust_remote_code=True)
            model_module = cast(nn.Module, model)
            if self.device in {"cuda", "mps"}:
                model_module = model_module.to(self.device)
            model_module.eval()
            self.model = model_module
            print("✅ Fallback modell betöltve")

    def _load_bm25_index(self) -> None:
        if not config.BM25_INDEX_PATH.exists():
            print("⚠️  BM25 index nem található")
            return

        self.bm25 = BM25Index.load(config.BM25_INDEX_PATH)
        if not self.bm25:
            print("⚠️  BM25 index betöltése sikertelen")
            return

        self.bm25_stats = {
            "avg_doc_length": getattr(self.bm25, "avg_doc_length", 0.0),
            "total_docs": getattr(self.bm25, "total_docs", 0),
        }
        print(f"✅ BM25 index betöltve ({self.bm25.total_docs} dokumentum)")

    def _load_faiss_index(self) -> None:
        if not config.FAISS_INDEX_PATH.exists():
            print("ℹ️  FAISS index nem található, csak BM25 fog futni")
            return

        try:
            index = faiss.read_index(str(config.FAISS_INDEX_PATH))
            gpu_res_cls = getattr(faiss, "StandardGpuResources", None)
            if self.device == "cuda" and callable(gpu_res_cls):
                res = gpu_res_cls()  # type: ignore[call-arg]
                index = faiss.index_cpu_to_gpu(res, 0, index)  # type: ignore[attr-defined]
                print("🎯 FAISS GPU mód aktiválva")
            else:
                print("ℹ️  FAISS CPU/MPS módban fut")
            self.faiss_index = index
            print(f"✅ FAISS index betöltve ({self.faiss_index.ntotal} vektor)")
        except Exception as exc:
            print(f"⚠️  FAISS betöltési hiba: {exc}")

    def _load_id_maps(self) -> None:
        self.chunk_id_map = {}
        self.doc_id_map = {}
        self.chunk_to_doc = {}
        self.known_chunk_ids = set()

        chunk_map_path = getattr(config, "CHUNK_ID_MAP_PATH", None)
        if chunk_map_path and Path(chunk_map_path).exists():
            try:
                with open(chunk_map_path, "r", encoding="utf-8") as handle:
                    self.chunk_id_map = json.load(handle)
                self.known_chunk_ids = set(self.chunk_id_map.values())
                for chunk_id in self.known_chunk_ids:
                    self.chunk_to_doc[chunk_id] = self._strip_chunk_suffix(chunk_id)
                print(f"🔗 Chunk ID mapping betöltve ({len(self.chunk_id_map)} elem)")
            except Exception as exc:
                print(f"⚠️  Chunk ID mapping betöltési hiba: {exc}")

        doc_map_path = getattr(config, "DOC_ID_MAP_PATH", None)
        if doc_map_path and Path(doc_map_path).exists():
            try:
                with open(doc_map_path, "r", encoding="utf-8") as handle:
                    self.doc_id_map = json.load(handle)
                print(f"🗂️  Doc ID mapping betöltve ({len(self.doc_id_map)} elem)")
            except Exception as exc:
                print(f"⚠️  Doc ID mapping betöltési hiba: {exc}")
        
        # Chunks metadata betöltése (agents.md: court, domain, year for GRPO)
        chunks_path = getattr(config, "CHUNKS_JSONL", None)
        if chunks_path and Path(chunks_path).exists():
            try:
                self.chunk_metadata = {}
                with open(chunks_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            chunk = json.loads(line)
                            chunk_id = chunk.get("chunk_id", "")
                            if chunk_id:
                                self.chunk_metadata[chunk_id] = {
                                    "doc_id": chunk.get("doc_id", ""),
                                    "court": chunk.get("court", ""),
                                    "domain": chunk.get("domain", ""),
                                    "year": chunk.get("year", ""),
                                    "text": chunk.get("text", ""),
                                }
                print(f"📋 Chunk metadata betöltve ({len(self.chunk_metadata)} chunk)")
            except Exception as exc:
                print(f"⚠️  Chunk metadata betöltési hiba: {exc}")

    def _embed_query(self, query: str) -> np.ndarray:
        if not self.tokenizer or not self.model:
            raise ValueError("Nincs betöltött embedding modell")

        try:
            tokenizer = self.tokenizer
            model = self.model
            if tokenizer is None or model is None:
                raise ValueError("Nincs betöltött tokenizer vagy modell")

            inputs = tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=getattr(config, "EMBEDDING_MAX_LENGTH", 1024),
                padding=True,
                return_attention_mask=True,
            )

            model_device = next(model.parameters()).device
            inputs = {k: v.to(model_device, non_blocking=True) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

                if hasattr(outputs, "last_hidden_state"):
                    embedding = outputs.last_hidden_state[:, 0, :].float()
                elif hasattr(outputs, "pooler_output"):
                    embedding = outputs.pooler_output.float()
                else:
                    embedding = torch.mean(outputs.last_hidden_state, dim=1).float()

                if embedding.ndim == 2 and embedding.shape[0] == 1:
                    embedding = embedding.squeeze(0)
                if model_device.type != "cpu":
                    embedding = embedding.cpu()

                embedding = embedding.numpy().astype(np.float32)

            norm = np.linalg.norm(embedding)
            if norm == 0:
                norm = 1.0
            return embedding / norm
        except Exception as exc:
            raise ValueError(f"Lekérdezés embedding hiba: {exc}")

    def _aggregate_by_doc(
        self, chunk_scores: List[Tuple[str, float]]
    ) -> Tuple[List[Tuple[str, float]], Dict[str, List[Tuple[str, float]]]]:
        doc_scores: Dict[str, float] = {}
        doc_chunks: Dict[str, List[Tuple[str, float]]] = {}

        for chunk_id, score in chunk_scores:
            doc_id = self._chunk_id_to_doc_id(chunk_id)
            doc_chunks.setdefault(doc_id, []).append((chunk_id, score))
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

        ranked_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_docs, doc_chunks

    def retrieve_candidates(
        self, query: str, top_k: int = 10
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        bm25_results: List[Tuple[str, float]] = []
        dense_results: List[Tuple[str, float]] = []

        if self.bm25:
            chunk_hits = self.bm25.search(query, top_k=top_k)
            bm25_results, _ = self._aggregate_by_doc(chunk_hits)
            self.last_chunk_scores["bm25"] = chunk_hits
            self.last_doc_scores["bm25"] = bm25_results
        else:
            self.last_chunk_scores["bm25"] = []
            self.last_doc_scores["bm25"] = []

        if self.faiss_index and self.chunk_id_map:
            try:
                query_emb = self._embed_query(query)
                distances, indices = self.faiss_index.search(np.expand_dims(query_emb, axis=0), top_k)
                dense_chunks: List[Tuple[str, float]] = []
                for distance, idx in zip(distances[0], indices[0]):
                    if idx == -1:
                        continue
                    chunk_id = self.chunk_id_map.get(str(idx))
                    doc_id = self.doc_id_map.get(str(idx)) if not chunk_id else None
                    if not chunk_id and not doc_id:
                        continue
                    identifier = chunk_id if chunk_id else doc_id
                    if identifier is None:
                        continue
                    dense_chunks.append((identifier, float(distance)))

                dense_results, _ = self._aggregate_by_doc(dense_chunks)
                self.last_chunk_scores["dense"] = dense_chunks
                self.last_doc_scores["dense"] = dense_results
            except Exception as exc:
                print(f"Dense retrieval hiba: {exc}")
        else:
            self.last_chunk_scores["dense"] = []
            self.last_doc_scores["dense"] = []

        return bm25_results, dense_results

    @staticmethod
    def _calculate_zscores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        if std_score == 0:
            return [0.0] * len(scores)
        return [float((score - mean_score) / std_score) for score in scores]

    def _fuse_rrf(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        k: int = config.RRF_K,
    ) -> List[Tuple[str, float]]:
        rrf_scores: Dict[str, float] = {}
        bm25_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_results, start=1)}
        dense_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_results, start=1)}
        all_docs = set(bm25_ranks.keys()) | set(dense_ranks.keys())

        for doc_id in all_docs:
            score = 0.0
            if doc_id in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[doc_id])
            if doc_id in dense_ranks:
                score += 1.0 / (k + dense_ranks[doc_id])
            rrf_scores[doc_id] = score

        return sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    def _fuse_zscore(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        bm25_scores = dict(bm25_results)
        dense_scores = dict(dense_results)
        all_docs = set(bm25_scores.keys()) | set(dense_scores.keys())

        bm25_z = dict(zip(bm25_scores.keys(), self._calculate_zscores(list(bm25_scores.values()))))
        dense_z = dict(zip(dense_scores.keys(), self._calculate_zscores(list(dense_scores.values()))))

        fused: Dict[str, float] = {}
        for doc_id in all_docs:
            score = bm25_z.get(doc_id, 0.0) + dense_z.get(doc_id, 0.0)
            fused[doc_id] = float(score)

        return sorted(fused.items(), key=lambda item: item[1], reverse=True)

    def fuse_results(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        method: str = "rrf",
    ) -> List[Tuple[str, float]]:
        if method == "rrf":
            fused = self._fuse_rrf(bm25_results, dense_results)
        elif method == "zscore":
            fused = self._fuse_zscore(bm25_results, dense_results)
        else:
            raise ValueError("Ismeretlen fusion metódus: használj 'rrf' vagy 'zscore' értéket")

        self.last_doc_scores["fused"] = fused
        return fused

    def retrieve(self, query: str, top_k: int = 10, fusion_method: str = "rrf") -> List[str]:
        if not self.bm25 and not self.faiss_index:
            print("Figyelem: Nincs elérhető index")
            return []

        bm25_results, dense_results = self.retrieve_candidates(query, top_k=config.TOP_K_BASELINE)
        fused = self.fuse_results(bm25_results, dense_results, method=fusion_method)
        top_docs = [doc_id for doc_id, _ in fused[:top_k]]
        return top_docs

    # Segédfüggvények GRPO feature exporthoz
    def get_last_doc_scores(self, source: str) -> List[Tuple[str, float]]:
        return list(self.last_doc_scores.get(source, []))

    def get_last_chunk_scores(self, source: str) -> List[Tuple[str, float]]:
        return list(self.last_chunk_scores.get(source, []))
    
    def get_doc_metadata(self, doc_id: str) -> Dict[str, Any]:
        """
        Visszaadja a dokumentum metaadatait (court, domain, year) a chunks-ból.
        Agents.md: GRPO-nak szüksége van ezekre az adatokra.
        """
        # Első chunk keresése a doc_id-hoz
        for chunk_id, metadata in self.chunk_metadata.items():
            if metadata.get("doc_id") == doc_id:
                return {
                    "court": metadata.get("court", ""),
                    "domain": metadata.get("domain", ""),
                    "year": metadata.get("year", ""),
                    "text": metadata.get("text", "")[:500],  # első 500 karakter
                }
        return {"court": "", "domain": "", "year": "", "text": ""}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CourtRankRL hibrid kereső")
    parser.add_argument("--query", type=str, default="családi jogi ügy", help="Lekérdezés")
    parser.add_argument("--top-k", type=int, default=config.TOP_K_RERANKED, help="Visszaadandó dokumentumok száma")
    parser.add_argument(
        "--fusion-method",
        type=str,
        default="rrf",
        choices=["rrf", "zscore"],
        help="Fusion metódus",
    )
    parser.add_argument("--show-device", action="store_true", help="Eszközinformáció megjelenítése")

    args = parser.parse_args()

    retriever = HybridRetriever()
    if args.show_device:
        print(f"🔌 Eszköz: {retriever.device}")
        print(f"📊 BM25 dokumentumok: {retriever.bm25_stats.get('total_docs', 0)}")

    doc_ids = retriever.retrieve(args.query, top_k=args.top_k, fusion_method=args.fusion_method)

    print(f"🔍 Lekérdezés: {args.query}")
    print(f"⚙️  Fusion: {args.fusion_method}")
    print("📋 Eredmények:")
    for idx, doc_id in enumerate(doc_ids, start=1):
        print(f"{idx}. {doc_id}")


if __name__ == "__main__":
    main()
