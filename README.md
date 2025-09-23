# CourtRankRL – Magyar bírósági határozatok hibrid visszakeresése RL‑alapú újrarangsorolással

## Áttekintés

Compute‑light, lokálisan futtatható pipeline magyar bírósági határozatokra. A rendszer Doclinggel feldolgozza a DOCX fájlokat, chunkol, BM25 és FAISS indexet épít, hibrid (sparse+dense) visszakeresést végez RRF fúzióval, és opcionálisan GRPO‑stílusú RL‑lel újrarangsorol. A lekérdezések kimenete kizárólag azonosítókból álló lista (doc_id), magyar nyelvű kísérőszöveg nélkül.

Fő komponensek (high‑level)
- Docling feldolgozás és minimál normalizálás.
- Chunkolás átfedéssel, meta megtartással.
- BM25 (sparse) index és FAISS (dense) index építés.
- Hibrid visszakeresés RRF fúzióval (alapértelmezett).
- RL alapú újrarangsorolás (GRPO) – opcionális, PoC‑barát.

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
# Teljes build pipeline (subset → Docling → chunking → BM25 → EmbeddingGemma FAISS)
python src/cli.py build

# Keresés baseline módban
python src/cli.py query "családi jogi ügy"

# Keresés GRPO reranking-gal (ha már van trained policy)
python src/cli.py query "szerződéses jog" --rerank

# GRPO policy tanítása
python src/cli.py train
```

### ☁️ RunPod Cloud GPU futtatás

A projekt **100%-ban kompatibilis** RunPod cloud GPU-kkal:

```bash
# 1. Notebook feltöltése RunPod-ra
# 2. GPU instance indítása (32GB+ memória ajánlott)
# 3. Jupyter notebook futtatása

# Automatikus elérési utak:
# 📁 Input: /workspace/data/processed/chunks.jsonl
# 💾 Output: /workspace/data/index/faiss_index.bin
# 🗺️ Mapping: /workspace/data/index/chunk_id_map.json

# Részletes útmutató: notebooks/README_embedding.md
```

**Előnyök RunPod-on:**
- ⚡ **GPU gyorsítás**: 32GB+ memória optimalizálva
- 🔄 **Streaming feldolgozás**: 3M+ chunk biztonságos kezelése
- 📦 **Önálló notebook**: Nem függ külső konfigurációktól
- 🧠 **Memória optimalizált**: FP16 + batch védelem

## Futtatás – Részletes build lépések

1) Build pipeline:
- `uv run courtrankrl build`
  - Automatikusan lefuttatja a Docling és BM25 lépéseket.

2) Manuális lépések (opcionális):
- `uv run python src/data_loader/preprocess_documents.py --resume`
  - Bemenet: `data/raw/` alatti DOCX.
  - Kimenet: `data/processed/chunks.jsonl` (chunkok minimál metaadatokkal).

- `uv run python src/data_loader/build_bm25_index.py`
  - Kimenet: `data/index/bm25_index.json`.

3) Embedding generálás (kötelező):
- Használja a `notebooks/qwen_embedding_runpod.ipynb` notebookot
  - Bemenet: `data/processed/chunks.jsonl`
  - Kimenetek: `data/index/faiss_index.bin`, `data/index/chunk_id_map.json`.

## Lekérdezés (hibrid baseline)

- `uv run courtrankrl query "kártérítés szivattyú ügy"`
  - HybridRetriever: BM25 + FAISS, RRF fúzió.
  - Kimenet: dokumentum azonosítók listája (határozat számok).

- Opcionális GRPO reranking:
  - `uv run courtrankrl query "kártérítés szivattyú ügy" --rerank`
  - Kimenet: GRPO-val újrarangsorolt dokumentum azonosítók.

**Fontos:** A lekérdezés előtt futtassa a `qwen_embedding_runpod.ipynb` notebookot az embeddingek és FAISS index generálásához.

Tippek
- A hibrid visszakeresés Qwen3-Embedding-0.6B modellt használja a lekérdezés embeddelésére.
- A Qwen3 használatához GPU/MPS szükséges (M3 MacBook Air optimalizálva).
- A query embedding real-time történik a betöltött Qwen3 modellel.
- A Qwen3 model csak akkor töltődik be, ha van FAISS index.
- M3 MacBook Air: MPS (Metal Performance Shaders) használata a GPU gyorsításhoz.

## RL újrarangsorolás (opcionális PoC)

- Tanítás (qrels szükséges):
  - `uv run courtrankrl train`
  - Megjegyzés: a tréner whitespace‑delimitált qrels fájlt vár. Állítsd a `configs/config.py` `DEV_QRELS_FILE` értékét a megfelelő fájlra, vagy igazítsd a formátumot.
- Használat kereséskor: a `courtrankrl query` automatikusan próbálja betölteni a policy‑t (`data/models/rl_policy.pth`), és ha elérhető, a jelölteket újrarangsorolja.

## Artefaktumok és elérési utak

- Chunks: `data/processed/chunks.jsonl`
- BM25 index: `data/index/bm25_index.json`
- FAISS index: `data/index/faiss_index.bin` (generálva `qwen_embedding_runpod.ipynb`-ban)
- FAISS ID‑map: `data/index/chunk_id_map.json` (generálva `qwen_embedding_runpod.ipynb`-ban)
- RL policy: `data/models/rl_policy.pth`

## Konfiguráció (részletek a `configs/config.py` fájlban)

- Chunkolás: méret, átfedés, per‑dokumentum limit.
- BM25: `BM25_K1`, `BM25_B`.
- Qwen3: `QWEN3_MODEL_NAME`, `QWEN3_DIMENSION`.
- Hybrid: `TOP_K_BASELINE`, `RRF_K`.
- RL: tanulási ráta, epochok, batch méret, rejtett dimenzió.

## Hibaelhárítás

- FAISS index hiányzik: futtassa a `qwen_embedding_runpod.ipynb` notebookot az embeddingek generálásához.
- Memória: növeld fokozatosan a batch méretet; OOM esetén csökkentse a batch size-ot.
- GPU: a Qwen3 embedding generáláshoz GPU szükséges.

## Nyelvi irányelv

- A projekt minden felhasználó felé megjelenő kimenete magyar nyelvű.
- A lekérdezés kimenete kizárólag azonosítókból álló lista (doc_id), magyarázó szöveg nélkül.

—

Készítette: Zelenyiánszki Máté
Implementáció: Python, Hugging Face Transformers, FAISS, PyTorch
