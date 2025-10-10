#!/usr/bin/env python3
"""
CourtRankRL Hybrid Retrieval

Agents.md specifikáció alapján:
- BM25 + FAISS hybrid retrieval lokális futtatásra
- Memória-optimalizált FAISS index (PQ quantization)
- RRF fusion (parameter-free)
- Hungarian status messages

Használat:
    python scripts/hybrid_retrieval.py

Előfeltételek:
- chunks.jsonl
- bm25/ könyvtár (bm25s_model/, chunk_ids.json, bm25_stats.json)
- faiss_index.bin (PQ-compressed)
- chunk_id_map.npy
- sample_queries.txt

Output:
- pipeline_results.jsonl
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
import bm25s
from huggingface_hub import login
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# GPU check
print("🔌 GPU Információk:")
print(f"  CUDA elérhető: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU név: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ℹ️ CUDA nem elérhető - CPU módban fut")

# HF token
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(token=hf_token)
    print("✅ HuggingFace token betöltve")
else:
    print("⚠️ HuggingFace token nem található (HUGGINGFACE_TOKEN env var)")

# === Konfiguráció ===
BASE_PATH = Path(os.getenv("WORKSPACE_PATH", "/Users/zelenyianszkimate/Documents/CourtRankRL"))

# Input fájlok
CHUNKS_PATH = BASE_PATH / "data" / "processed" / "chunks.jsonl"
BM25_INDEX_DIR = BASE_PATH / "data" / "index" / "bm25" / "bm25s_model"
BM25_CHUNK_IDS_PATH = BASE_PATH / "data" / "index" / "bm25" / "chunk_ids.json"
BM25_STATS_PATH = BASE_PATH / "data" / "index" / "bm25" / "bm25_stats.json"
FAISS_INDEX_PATH = BASE_PATH / "data" / "index" / "faiss_index.bin"
CHUNK_ID_MAP_PATH = BASE_PATH / "data" / "index" / "chunk_id_map.npy"
QUERIES_PATH = BASE_PATH / "data" / "qrels" / "sample_queries.txt"

# Output fájl
OUTPUT_PATH = BASE_PATH / "data" / "qrels" / "pipeline_results.jsonl"

# Model & parameters
EMBEDDING_MODEL = "google/embeddinggemma-300m"
EMBEDDING_DIM = 768

# Retrieval parameters (agents.md szerint)
TOP_K_BASELINE = 300  # Retrieve top-300 from each source
TOP_K_OUTPUT = 20     # Final top-K after fusion
RRF_K = 60            # RRF parameter (agents.md)
FAISS_NPROBE = 64     # CRITICAL: recall optimization (agents.md)

print("📂 Workspace és fájlok:")
print(f"  Base path: {BASE_PATH}")
print(f"  Chunks: {CHUNKS_PATH}")
print(f"  BM25 index: {BM25_INDEX_DIR}")
print(f"  FAISS index: {FAISS_INDEX_PATH}")
print(f"  Queries: {QUERIES_PATH}")
print(f"  Output: {OUTPUT_PATH}")
print()
print("⚙️ Retrieval konfiguráció:")
print(f"  Top-K baseline: {TOP_K_BASELINE}")
print(f"  Top-K output: {TOP_K_OUTPUT}")
print(f"  RRF_K: {RRF_K}")
print(f"  FAISS nprobe: {FAISS_NPROBE}")

# Fájl ellenőrzés
required_files = [
    (CHUNKS_PATH, "chunks.jsonl"),
    (BM25_INDEX_DIR, "BM25 index dir"),
    (BM25_CHUNK_IDS_PATH, "chunk_ids.json"),
    (FAISS_INDEX_PATH, "faiss_index.bin"),
    (CHUNK_ID_MAP_PATH, "chunk_id_map.npy"),
    (QUERIES_PATH, "sample_queries.txt"),
]

print("\n📋 Fájl ellenőrzés:")
all_ok = True
for fpath, fname in required_files:
    exists = fpath.exists()
    status = "✅" if exists else "❌"
    print(f"  {status} {fname}")
    if not exists:
        all_ok = False

if not all_ok:
    raise FileNotFoundError("❌ Hiányzó fájlok! Futtasd előbb a build és embedding pipeline-t.")

print("\n✅ Konfiguráció és fájlellenőrzés kész")

# === BM25 Index Betöltése ===
print("\n" + "="*60)
print("🔧 BM25 INDEX BETÖLTÉSE")
print("="*60)

# Load BM25S model
bm25_model = bm25s.BM25.load(str(BM25_INDEX_DIR), load_corpus=True)
print(f"✅ BM25S model betöltve: {BM25_INDEX_DIR}")

# Load chunk IDs
with open(BM25_CHUNK_IDS_PATH, 'r', encoding='utf-8') as f:
    bm25_chunk_ids = json.load(f)
print(f"✅ Chunk IDs betöltve: {len(bm25_chunk_ids):,} elem")

# Load stats
with open(BM25_STATS_PATH, 'r', encoding='utf-8') as f:
    bm25_stats = json.load(f)
print("✅ BM25 stats betöltve:")
print(f"   Total docs: {bm25_stats.get('total_docs', 0):,}")
print(f"   Avg length: {bm25_stats.get('avg_doc_length', 0):.1f}")

# === EmbeddingGemma Modell Betöltése ===
print("\n" + "="*60)
print("🔧 EMBEDDINGGEMMA MODELL BETÖLTÉSE")
print("="*60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📥 Modell betöltése: {EMBEDDING_MODEL}")
print(f"   Device: {device}")

try:
    # CPU módban float32 használata (agents.md: EmbeddingGemma does NOT support float16)
    model_kwargs = {} if device == "cpu" else {"torch_dtype": torch.bfloat16}

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device,
        cache_folder=str(BASE_PATH / ".hf_cache"),
        model_kwargs=model_kwargs,
    )

    print("✅ EmbeddingGemma modell betöltve (Sentence Transformers)")
    print(f"   Max seq length: {model.max_seq_length}")
    print(f"   Embedding dim: {EMBEDDING_DIM}")
    print(f"   Precision: {'bfloat16 (GPU)' if device == 'cuda' else 'float32 (CPU)'}")

    # Test embedding
    test_query = "családi jogi ügy"
    test_emb = model.encode(
        test_query,
        prompt_name="query",  # Query prompt (agents.md)
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization
    )

    test_norm = np.linalg.norm(test_emb)
    print("\n✅ Test embedding:")
    print(f"   Shape: {test_emb.shape}")
    print(f"   L2 norm: {test_norm:.6f} (kell: ~1.0)")
    print(f"   Non-zero: {np.count_nonzero(test_emb)}/{len(test_emb)}")

    if test_norm < 0.9 or test_norm > 1.1:
        raise ValueError(f"❌ Invalid embedding norm: {test_norm:.6f}")

    print("✅ Embedding validáció sikeres")

except Exception as e:
    print(f"❌ Modell betöltési hiba: {e}")
    print(f"   Típus: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    raise

# === FAISS Index Betöltése ===
print("\n" + "="*60)
print("🔧 FAISS INDEX BETÖLTÉSE")
print("="*60)

# Load FAISS index from disk
print("📥 FAISS index betöltése...")
faiss_index = faiss.read_index(str(FAISS_INDEX_PATH), faiss.IO_FLAG_MMAP)
print(f"✅ Index betöltve: {faiss_index.ntotal:,} vektor")
print(f"   Index típus: {type(faiss_index).__name__}")
print(f"   Dimenzió: {faiss_index.d}")

# KRITIKUS: nprobe beállítása (agents.md szerint)
if hasattr(faiss_index, 'nprobe'):
    faiss_index.nprobe = FAISS_NPROBE
    print(f"🎯 nprobe beállítva: {faiss_index.nprobe} (recall optimization)")

def set_ivf_nprobe(index, nprobe: int):
    base = index
    while isinstance(base, faiss.IndexPreTransform):
        base = base.index
    if hasattr(base, "nprobe"):
        base.nprobe = nprobe
        return True
    return False

# betöltés után:
ok = set_ivf_nprobe(faiss_index, FAISS_NPROBE)
print(f"🎯 nprobe beállítva belső IVF-re: {ok}, érték: {FAISS_NPROBE}")

# Load chunk ID mapping (.npy, kompatibilis dict és ndarray formátummal)
raw_map = np.load(CHUNK_ID_MAP_PATH, allow_pickle=True)

# Ha dict-et np.save()-eltünk, np.load 0-dim ndarray-ként adhatja vissza → .item()
if isinstance(raw_map, np.ndarray) and raw_map.dtype == object and raw_map.shape == ():
    raw_map = raw_map.item()

if isinstance(raw_map, np.ndarray):
    print(f"✅ Chunk ID mapping betöltve (ndarray): {len(raw_map):,} elem")
elif isinstance(raw_map, dict):
    print(f"✅ Chunk ID mapping betöltve (dict): {len(raw_map):,} elem")
else:
    raise TypeError(f"Ismeretlen chunk_id_map típus: {type(raw_map)}")

def idx_to_chunk_id(i: int):
    if isinstance(raw_map, np.ndarray):
        if 0 <= i < len(raw_map):
            return str(raw_map[i])
        return None
    if isinstance(raw_map, dict):
        # Kulcs lehet int, np.int64 vagy str; próbáljuk sorban
        v = raw_map.get(i)
        if v is None:
            try:
                v = raw_map.get(int(i))
            except Exception:
                pass
        if v is None:
            v = raw_map.get(str(i))
        return v
    return None

# === Query Lista Betöltése ===
print("\n" + "="*60)
print("📋 QUERY LISTA BETÖLTÉSE")
print("="*60)

with open(QUERIES_PATH, 'r', encoding='utf-8') as f:
    queries = [line.strip() for line in f if line.strip()]

print(f"✅ {len(queries)} query betöltve")
print("\nPélda query-k (első 5):")
for i, q in enumerate(queries[:5], 1):
    print(f"  {i}. {q[:70]}{'...' if len(q) > 70 else ''}")

if len(queries) > 5:
    print(f"  ... és még {len(queries) - 5}")

# === Helper Functions ===

def strip_chunk_suffix(chunk_id: str) -> str:
    """Convert chunk_id to doc_id (e.g., 'doc_0' -> 'doc')."""
    if "_" not in chunk_id:
        return chunk_id
    base, suffix = chunk_id.rsplit("_", 1)
    if suffix.isdigit():
        return base
    return chunk_id


def aggregate_chunks_to_docs(chunk_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Aggregate chunk-level scores to document-level (max-score strategy)."""
    doc_scores: Dict[str, float] = {}

    for chunk_id, score in chunk_scores:
        doc_id = strip_chunk_suffix(chunk_id)
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = score

    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


def rrf_fusion(bm25_results: List[Tuple[str, float]],
               dense_results: List[Tuple[str, float]],
               k: int = RRF_K) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion (RRF) - agents.md szerint."""
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

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


print("✅ Helper függvények definiálva")
print("   - strip_chunk_suffix: chunk_id → doc_id")
print("   - aggregate_chunks_to_docs: chunk-level → doc-level (max-score)")
print("   - rrf_fusion: BM25 + FAISS → RRF fusion")

# === Hybrid Retrieval Pipeline ===
print("\n" + "="*60)
print("🔍 HYBRID RETRIEVAL PIPELINE FUTTATÁSA")
print("="*60)
print(f"  Query-k száma: {len(queries)}")
print(f"  Top-K baseline: {TOP_K_BASELINE}")
print(f"  Top-K output: {TOP_K_OUTPUT}")
print(f"  Fusion: RRF (k={RRF_K})")
print()

results = []

for i, query in enumerate(queries, 1):
    print(f"[{i:3d}/{len(queries)}] {query[:60]:<60}", end=" ")

    try:
        # KRITIKUS: Query preprocessing (case-insensitive, agents.md)
        query_processed = query.lower().strip()

        # === BM25 Retrieval ===
        query_tokens = bm25s.tokenize([query_processed])
        bm25_results_raw, bm25_scores = bm25_model.retrieve(
            query_tokens,
            k=TOP_K_BASELINE
        )

        # Extract chunk_ids from corpus (agents.md: corpus structure)
        bm25_chunk_scores = []
        for result_dict, score in zip(bm25_results_raw[0], bm25_scores[0]):
            chunk_id = result_dict.get('text', '')  # CRITICAL: 'text' field contains chunk_id
            if chunk_id:
                bm25_chunk_scores.append((chunk_id, float(score)))

        # Aggregate to document level
        bm25_doc_scores = aggregate_chunks_to_docs(bm25_chunk_scores)

        # === FAISS Retrieval ===
        query_emb = model.encode(
            query_processed,
            prompt_name="query",  # Query prompt (agents.md)
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        distances, indices = faiss_index.search(
            np.expand_dims(query_emb, axis=0),
            TOP_K_BASELINE
        )

        # Map indices to chunk_ids
        dense_chunk_scores = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = idx_to_chunk_id(int(idx))
            if chunk_id:
                dense_chunk_scores.append((chunk_id, float(dist)))

        # Aggregate to document level
        dense_doc_scores = aggregate_chunks_to_docs(dense_chunk_scores)

        # === RRF Fusion ===
        fused_scores = rrf_fusion(bm25_doc_scores, dense_doc_scores, k=RRF_K)
        top_docs = [doc_id for doc_id, _ in fused_scores[:TOP_K_OUTPUT]]

        # Save result
        results.append({
            "query": query,
            "doc_ids": top_docs,
            "num_results": len(top_docs),
        })

        print(f"✅ {len(top_docs)} találat")

    except Exception as e:
        print(f"❌ Hiba: {e}")
        results.append({
            "query": query,
            "doc_ids": [],
            "num_results": 0,
            "error": str(e),
        })

print(f"\n✅ Pipeline futtatása kész: {len(results)} query feldolgozva")

# === Eredmények Mentése ===
print("\n" + "="*60)
print("💾 EREDMÉNYEK MENTÉSE")
print("="*60)

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for result in results:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

print(f"✅ Eredmények mentve: {OUTPUT_PATH}")
print(f"   {len(results)} query × top-{TOP_K_OUTPUT} dokumentum")

# Statisztikák
total_docs_found = sum(r['num_results'] for r in results)
queries_with_results = sum(1 for r in results if r['num_results'] > 0)
queries_with_errors = sum(1 for r in results if 'error' in r)

print("\n📊 STATISZTIKÁK:")
print(f"  Query-k száma: {len(results)}")
print(f"  Query-k eredménnyel: {queries_with_results}")
print(f"  Query-k hiba nélkül: {len(results) - queries_with_errors}")
print(f"  Összes doc: {total_docs_found}")
print(f"  Átlag doc/query: {total_docs_found / len(results):.1f}")

if queries_with_errors > 0:
    print(f"\n⚠️  {queries_with_errors} query-nél hiba történt:")
    for r in results:
        if 'error' in r:
            print(f"     - {r['query'][:50]}: {r['error']}")

print("\n🎉 Hybrid retrieval pipeline sikeresen lefutott!")
print("Következő lépés: futtasd a 'python scripts/qrels_generation.py' szkriptet")
