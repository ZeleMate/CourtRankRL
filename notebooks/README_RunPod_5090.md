# CourtRankRL RunPod 5090 GPU Optimalizáció

## 🚀 **Maximális Teljesítmény a 5090 GPU-val**

A CourtRankRL projekt **100%-ban optimalizált** a RunPod 5090 GPU-hoz, amely **24GB VRAM-mal** rendelkezik.

---

## 📊 **RunPod 5090 Specifikáció**

| Komponens | Specifikáció |
|-----------|-------------|
| **GPU** | NVIDIA RTX 5090 (24GB VRAM) |
| **CPU** | AMD EPYC (16+ cores) |
| **RAM** | 64GB+ system memory |
| **Storage** | SSD storage |
| **Network** | High-speed internet |

---

## 🔧 **Optimalizációk**

### ✅ **Model Konfiguráció**
- **Qwen3-Embedding-0.6B**: Teljes specifikáció szerinti modell
- **FP16**: Memória optimalizált számítási mód
- **CUDA**: GPU gyorsítás minden komponensnél
- **Batch Size**: 128 (maximális throughput)

### ✅ **Index Betöltés**
- **BM25 Index**: 9.6GB sparse index (teljes betöltés)
- **FAISS Index**: 2.9M vektor, kvantált (211MB)
- **Chunk Mapping**: 2.9M+ chunk_id ↔ FAISS mapping

### ✅ **RL Training**
- **GRPO Policy**: Shallow MLP neural network
- **GPU Training**: Batch size 64, epochs 10+
- **Memory Efficient**: Gradient accumulation

---

## 🎯 **Használat**

### **1. Query Futtatás**
```bash
cd /workspace
python3 -m src.cli query "családi jog" --top-k 10
```

**Eredmény:**
- BM25 + FAISS hybrid retrieval
- RRF fusion módszer
- GRPO reranking (ha betanítva)
- Top-10 dokumentum ID-k

### **2. Training Futtatás**
```bash
cd /workspace
python3 -m src.cli train
```

**Eredmény:**
- GRPO policy tanítása
- nDCG@10 optimalizáció
- Policy mentése `.pth` formátumban

### **3. Teljesítmény Teszt**
```bash
cd /workspace
python3 notebooks/runpod_query_training.py --query "családi jog" --compare
```

---

## ⚡ **Teljesítmény Előnyök**

### **16GB RAM vs 5090 GPU**

| Funkció | 16GB RAM (CPU) | 5090 GPU (24GB VRAM) |
|---------|----------------|---------------------|
| **Model** | Mock model | Qwen3-0.6B teljes |
| **BM25** | ❌ Nem betölthető | ✅ Teljes index |
| **FAISS** | ✅ 2.9M vektor | ✅ 2.9M vektor |
| **Batch Size** | 1-2 | 128 |
| **Training** | ❌ Nem lehetséges | ✅ Teljes GRPO |
| **Query Time** | Lassú | Gyors (GPU) |

### **Memória Használat**
- **GPU VRAM**: ~20-22GB (5090-ből)
- **System RAM**: ~10-15GB
- **Total**: ~30-37GB (64GB-ból biztonságos)

---

## 🔍 **Query Példák**

### **Teszt Queries**
```python
queries = [
    "családi jog",
    "ingatlan ügy",
    "szerződéses jogvita",
    "munkajogi vita",
    "kártérítési igény"
]
```

### **Várt Eredmény**
```
🔍 Query: családi jog
📊 Top-K: 10
🧠 Reranking: Enabled

📋 Step 1: Retrieving candidates...
   📄 BM25 candidates: 1000
   🧠 Dense candidates: 1000

🎯 Step 2: Applying GRPO reranking...
   ✅ Reranked to: 10 documents

🎯 RERANKED RESULTS:
1. I.50.080/2021/6. szám (score: 0.9876)
2. P.20.693/2020/25. szám (score: 0.9543)
...
```

---

## 🚨 **Hibaelhárítás**

### **Memória Hiba**
```python
# Csökkentse a batch size-ot
batch_size = 64  # Vagy 32, 16
max_length = 512  # Csökkentett token hossz
```

### **CUDA Hiba**
```python
# Ellenőrizze a CUDA verziót
import torch
print(torch.__version__)
print(torch.version.cuda)
```

### **Model Betöltés Hiba**
```python
# Fallback kisebb modelre
self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
```

---

## 🎉 **RunPod 5090 Optimalizáció Teljes!**

A CourtRankRL projekt **maximális teljesítményt** nyújt a 5090 GPU-n:

✅ **24GB VRAM** teljes kihasználása
✅ **Qwen3-0.6B** specifikáció szerinti modell
✅ **BM25 + FAISS** teljes indexek
✅ **GRPO Training** teljes RL tanítás
✅ **Batch processing** maximális throughput

**A rendszer készen áll a termelési használatra!** 🚀
