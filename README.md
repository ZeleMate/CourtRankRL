# CourtRankRL – Magyar bírósági határozatok hibrid visszakeresése RL‑alapú újrarangsorolással

## Áttekintés

Compute‑light, lokálisan futtatható pipeline magyar bírósági határozatokra. A rendszer Doclinggel feldolgozza a DOCX fájlokat, chunkol, BM25S és FAISS indexet épít, hibrid (sparse+dense) visszakeresést végez RRF fúzióval. A lekérdezések kimenete kizárólag azonosítókból álló lista (doc_id), magyar nyelvű kísérőszöveg nélkül.

**Architektúra:**
- **Lokális pipeline:** BM25S + EmbeddingGemma FAISS + hybrid retrieval (M3 MacBook Air 16GB RAM)
- **Cloud-only GRPO:** Reinforcement learning reranking Qwen3-4B-Instruct modellel (RunPod GPU)

Fő komponensek (high‑level)
- Docling feldolgozás és minimál normalizálás
- Chunkolás Docling intelligens szegmentálásával
- BM25S (sparse) index native tokenizer-rel
- FAISS (dense) index EmbeddingGemma-300m modellel (L2-normalized, IP metric)
- Hibrid visszakeresés RRF fúzióval
- GRPO újrarangsorolás (cloud-only, TRL GRPOTrainer, QLoRA adapters)

## Telepítés

1) UV környezet beállítása
- `uv sync` (telepíti a függőségeket és virtuális környezetet)
- `source .venv/bin/activate` (aktiválja a virtuális környezetet, ha szükséges; uv automatikusan kezeli)

2) Környezeti változók (`.env` a projekt gyökerében)
- `HUGGINGFACE_TOKEN=hf_...`  (EmbeddingGemma betöltéshez és query embedding-hez)

Megjegyzés: a projekt minden felhasználói kimenete magyar nyelvű; a query válasz kizárólag azonosítók listája.

## Gyors használat

### 🖥️ Lokális futtatás (CLI)

```bash
# 1. Build pipeline (Docling → chunking → BM25S)
uv run courtrankrl build

# 2. FAISS index generálás (RunPod GPU-n)
# Futtassa: notebooks/gemma_embedding_runpod.ipynb

# 3. Keresés (baseline only, agents.md szerint)
uv run courtrankrl query "családi jogi ügy"

# 4. GRPO slate export (cloud training előkészítés)
uv run courtrankrl train

# 5. GRPO training (RunPod GPU-n)
# Futtassa: notebooks/grpo_train_runpod.ipynb
```

**Megjegyzés:** GRPO reranking csak cloud-on (agents.md spec szerint). Lokális query csak baseline-t ad vissza.

### ☁️ RunPod Cloud GPU futtatás

A projekt **két cloud notebook-ot** tartalmaz RunPod GPU-ra optimalizálva:

#### 1. FAISS Embedding Index (`gemma_embedding_runpod.ipynb`)
```bash
# GPU: A100/H100/RTX 5090 (24GB+ VRAM ajánlott)
# Input: /workspace/chunks.jsonl
# Output: /workspace/faiss_index.bin, /workspace/chunk_id_map.json

# Model: google/embeddinggemma-300m
# Optimalizációk: FP16, Flash Attention 2, PyTorch compile
```

#### 2. GRPO Training (`grpo_train_runpod.ipynb`)
```bash
# GPU: RTX 5090 (24GB VRAM) - optimalizált konfiguráció
# Input: /workspace/training_slates.jsonl (98 query × 20 chunk/slate)
# Output: /workspace/artifacts/grpo_policy/ (LoRA adapters + metrics)

# Model: Qwen/Qwen3-4B-Instruct-2507 (4-bit) + QLoRA (rank=64, alpha=128)
# Training: TRL GRPOTrainer GRPO algoritmus
#   - Loss: dapo (eliminates length bias)
#   - Reward scaling: batch (robust - PPO Lite)
#   - Hardware: batch_size=2, grad_accumulation=2, 6 generations/prompt
#   - Training time: ~45-60 perc (500 steps)
```

**Előnyök RunPod-on:**
- ⚡ **GPU gyorsítás**: 4B parameter model training
- 🔄 **Streaming feldolgozás**: 3M+ chunk biztonságos kezelése
- 📦 **Önálló notebookok**: Környezeti változókból konfigurálható
- 🧠 **Memória optimalizált**: 4-bit quantization, bf16 compute

## Futtatás – Részletes build lépések

### 1. Lokális build pipeline

```bash
uv run courtrankrl build
```

**Mit csinál:**
- Docling parsing: `data/raw/` DOCX → plain text
- Normalizálás és metadata extraction (court, domain, year)
- Intelligens chunking (Docling capabilites)
- BM25S index építés native tokenizer-rel
- Kimenetek: `data/processed/chunks.jsonl`, `data/index/bm25/bm25s_model/`

### 2. Cloud FAISS embedding generálás (kötelező)

```bash
# RunPod GPU-n futtassa: notebooks/gemma_embedding_runpod.ipynb
# Bemenet: chunks.jsonl
# Kimenetek: faiss_index.bin, chunk_id_map.json
# Töltse le lokálisan: data/index/ könyvtárba
```

### 3. Cloud GRPO training (opcionális)

```bash
# 1. Slate export lokálisan
uv run courtrankrl train

# 2. RunPod GPU-n futtassa: notebooks/grpo_train_runpod.ipynb
# Bemenet: training_slates.jsonl
# Kimenetek: grpo_policy/ (LoRA adapters + metrics.json)
```

## Lekérdezés (hibrid baseline)

```bash
# Alapértelmezett keresés (RRF fusion)
uv run courtrankrl query "kártérítés szivattyú ügy"

# Több találat kérése
uv run courtrankrl query "szerződéses jog" --top-k 20

# Példa családi jogi ügyre
uv run courtrankrl query "családi jogi ügy"
```

**HybridRetriever működés:**
1. Query embedding: EmbeddingGemma-300m (MPS acceleration)
2. BM25S sparse retrieval: chunk-level → document-level max-score aggregation
3. FAISS dense retrieval: IndexIVFFlat (100% exact distances, ~9GB RAM for 3M vectors), L2-normalized IP search
4. Fusion: RRF (Reciprocal Rank Fusion) - robusztus, paraméter-mentes algoritmus
5. Kimenet: Top-k document ID lista

**Előfeltétel:** FAISS index létezik (`gemma_embedding_runpod.ipynb`)

**Megjegyzés:** GRPO reranking cloud-only (agents.md spec). Lokális query csak baseline-t ad vissza.

## GRPO újrarangsorolás (cloud-only, agents.md szerint)

### Slate export (lokálisan)
```bash
uv run courtrankrl train
# Output: data/models/grpo_policy/training_slates.jsonl
```

### GRPO training (RunPod GPU-n)
```bash
# Futtassa: notebooks/grpo_train_runpod.ipynb
# Qrels formátum: data/qrels/baseline_qrels.tsv
# - Header: query_id\tdoc_id\trelevance
# - Doc IDs: chunks.jsonl doc_id mezőből (NEM chunk_id!)
# - Relevance: {0, 1, 2}
```

**GRPO konfiguráció (RTX 5090 optimalizált):**
- Model: Qwen/Qwen3-4B-Instruct-2507 (4-bit) + QLoRA (rank=64, alpha=128, 7 target modules)
- Dataset: 98 query (teljes), 20 chunk/slate, teljes chunk szöveg
- Trainer: TRL GRPOTrainer GRPO algoritmus (loss_type="dapo", scale_rewards="batch")
- Reward: nDCG@10 difference + entropy bonus (0.01), clipping [-1.0, 1.0]
- Hardware: RTX 5090 - batch_size=2, grad_accumulation=2, 6 generations, 500 steps
- Training time: ~45-60 perc
- Output: LoRA adapter weights + metrics.json

**Megjegyzés:** Lokális inference nem támogatott (4B model túl nagy 16GB RAM-hoz).

## Artefaktumok és elérési utak

### Lokális artifactok
- Chunks: `data/processed/chunks.jsonl`
- Processed docs: `data/processed/processed_docs.jsonl`
- BM25S index: `data/index/bm25/bm25s_model/` (corpus, vocab, params, indices)
- BM25 stats: `data/index/bm25/bm25_stats.json`
- BM25 chunk IDs: `data/index/bm25/chunk_ids.json`
- Token cache: `data/index/bm25/token_cache/` (token_ids.npy, vocab.json)

### Cloud-ról letöltendő artifactok
- FAISS index: `data/index/faiss_index.bin` (gemma_embedding_runpod.ipynb)
- Chunk ID map: `data/index/chunk_id_map.json` (gemma_embedding_runpod.ipynb)
- GRPO adapters: `data/models/grpo_policy/` (grpo_train_runpod.ipynb)
- GRPO metrics: `data/models/grpo_policy/metrics.json` (grpo_train_runpod.ipynb)

### Qrels
- Format: `data/qrels/baseline_qrels.tsv` (TSV, header, doc_id-k)

## Konfiguráció (`configs/config.py`)

### Retrieval
- **BM25S**: `BM25_K1=1.5`, `BM25_B=0.75`, `BM25_USE_NUMBA`, `BM25_THREADS`
- **Hybrid**: `TOP_K_BASELINE=100`, `TOP_K_RERANKED=20`, `RRF_K=60`
- **FAISS**: `FAISS_NLIST_MIN=64`, `FAISS_NLIST_MAX=1024`

### Memory
- **Batch sizes**: `CHUNK_WRITE_BATCH_SIZE=200`
- **Soft limit**: `MEMORY_SOFT_LIMIT_BYTES=12GB`

### GRPO (RTX 5090 cloud-only)
- **Slate config**: `GRPO_SLATE_SIZE=20` (chunk-based)
- **LoRA config**: `GRPO_LORA_RANK=64`, `GRPO_LORA_ALPHA=128`
- **Training config**: `GRPO_MAX_STEPS=500`, `GRPO_BATCH_SIZE=2`, `GRPO_NUM_GENERATIONS=6`
- **Paths**: `SLATE_EXPORT_PATH`, `QRELS_FILE`

## Hibaelhárítás

### Lokális problémák
- **FAISS index hiányzik**: Futtassa `gemma_embedding_runpod.ipynb` RunPod-on, töltse le az artifactokat
- **MPS acceleration**: M3 MacBook Air automatikusan használja, ha elérhető
- **BM25 index build lassú**: Állítsa `BM25_USE_NUMBA=True` és `BM25_THREADS=-1`
- **Memória hiba**: Csökkentse `CHUNK_WRITE_BATCH_SIZE` vagy `sample_size` értékét

### Cloud problémák
- **CUDA OOM (embedding)**: Csökkentse `BATCH_SIZE` értékét 512→256→128
- **CUDA OOM (GRPO)**: Növelje `GRADIENT_ACCUMULATION_STEPS` 4→8
- **Slow training**: Ellenőrizze Flash Attention 2 aktiválását
- **Qrels format error**: Ellenőrizze TSV header-t és doc_id-kat (nem chunk_id!)

## Szakdolgozati elemzés

```bash
# Adatelemző notebook futtatása
jupyter notebook notebooks/data_analysis.ipynb
```

**Tartalom:**
- Szöveghossz és struktúra elemzés (chunk/document szint)
- Metaadat eloszlások (bíróság, jogterület, év)
- Szókészlet és nyelvi jellemzők (top words, Zipf-törvény)
- FAISS embedding elemzés (norma, dimenzió-szintű stats)
- Professzionális ábrák és táblázatok

## Nyelvi irányelv

- **agents.md és README.md**: Angol
- **Minden más (CLI, notebook output, kommentek)**: Magyar
- **Query output**: Csak doc_id lista, magyarázó szöveg nélkül

---

**Készítette:** Zelenyiánszki Máté  
**Implementáció:** Python 3.11, Hugging Face (EmbeddingGemma, Qwen3), FAISS, BM25S, TRL, Docling  
**Optimalizálva:** M3 MacBook Air 16GB RAM (lokális), RunPod GPU (cloud)
