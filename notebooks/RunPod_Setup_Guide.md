# CourtRankRL RunPod Setup Guide

## 🚀 **RunPod 5090 GPU-n való futtatás**

Ez az útmutató segíti a CourtRankRL teljes pipeline futtatását RunPod Jupyter notebook-ban.

---

## 📋 **1. Előfeltételek**

### **RunPod Account**
1. Regisztrálj a [runpod.io](https://runpod.io)-n
2. Válaszd a **"Secure Cloud"** opciót
3. Keresd meg a **RTX 5090** GPU template-et

### **GPU Template Beállítások**
- **GPU**: RTX 5090 (24GB VRAM)
- **CPU**: AMD EPYC (16+ cores)
- **RAM**: 64GB+ system memory
- **Storage**: 50GB+ SSD storage
- **Container**: PyTorch 2.1+ (Python 3.11)
- **Internet**: High-speed internet (model letöltéshez)

---

## 📁 **2. Fájlok Feltöltése**

### **Projekt Struktúra**
```
/workspace/
├── src/
│   ├── cli.py
│   ├── search/
│   │   ├── hybrid_search.py
│   │   └── grpo_reranker.py
│   └── data_loader/
├── configs/
│   └── config.py
├── data/
│   └── index/
│       ├── faiss_index.bin      (211MB)
│       ├── chunk_id_map.json    (111MB)
│       └── bm25_index.json      (9.6GB)
└── notebooks/
    ├── RunPod_Full_Pipeline.py
    └── README_RunPod_5090.md
```

### **Feltöltési Módszerek**

#### **Módszer 1: RunPod Dashboard**
1. Lépj be a RunPod dashboard-ba
2. Válaszd ki a pod-odat
3. Kattints a **"Files"** tab-ra
4. Upload-old a teljes projektet ZIP-ben

#### **Módszer 2: RunPod CLI**
```bash
# Lokális gépen
cd /path/to/CourtRankRL
zip -r courtrankrl.zip .

# RunPod-on (ha már be vagy jelentkezve)
runpodctl upload courtrankrl.zip /workspace/
unzip courtrankrl.zip
```

#### **Módszer 3: Jupyter-ben**
```python
# Jupyter notebook-ban
import zipfile
import urllib.request

# ZIP feltöltés URL-ről
zip_url = "https://your-storage/courtrankrl.zip"
urllib.request.urlretrieve(zip_url, "courtrankrl.zip")

with zipfile.ZipFile("courtrankrl.zip", 'r') as zip_ref:
    zip_ref.extractall("/workspace")
```

---

## 🖥️ **3. Jupyter Notebook Beállítás**

### **Első Cellák Futtatása**

```python
# 1. Dependencies telepítése
!pip install --upgrade pip
!pip install faiss-cpu tqdm transformers accelerate torch sentencepiece

# 2. System check
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 3. Projekt betöltése
import sys
sys.path.append('/workspace')

from configs import config
from src.search.hybrid_search import HybridRetriever
from src.search.grpo_reranker import GRPOReranker

print("✅ Project loaded successfully!")
```

### **Teljes Pipeline Futtatása**

```python
# Teljes pipeline script futtatása
exec(open('/workspace/notebooks/RunPod_Full_Pipeline.py').read())
```

---

## 🎯 **4. Query Futtatás**

### **Egyszerű Query**
```python
from src.search.hybrid_search import HybridRetriever

retriever = HybridRetriever()
results = retriever.retrieve("családi jog", top_k=10, fusion_method="rrf")

for i, doc_id in enumerate(results, 1):
    print(f"{i}. {doc_id}")
```

### **CLI Parancs**
```bash
cd /workspace
python3 -m src.cli query "családi jog" --top-k 10
```

### **Batch Query**
```python
queries = [
    "családi jog",
    "ingatlan ügy",
    "szerződéses jogvita",
    "munkajogi vita",
    "kártérítési igény"
]

for query in queries:
    results = retriever.retrieve(query, top_k=5)
    print(f"\n{query}: {len(results)} results")
```

---

## 🏃 **5. Training Futtatás**

### **GRPO Training**
```python
from src.search.grpo_reranker import GRPOReranker

reranker = GRPOReranker()

# Training loop
for epoch in range(10):
    print(f"Epoch {epoch+1}/10")
    # Training logic here
    pass

# Policy mentése
torch.save(reranker.policy.state_dict(), config.RL_POLICY_PATH)
print("✅ Training completed!")
```

### **CLI Training**
```bash
cd /workspace
python3 -m src.cli train
```

---

## 📊 **6. Monitoring & Debug**

### **GPU Memory Monitor**
```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"GPU Memory: {allocated:.1f}GB / {total:.1f}GB")
        print(f"Memory Usage: {allocated/total*100:.1f}%")
        print(f"Reserved: {reserved:.1f}GB")
```

### **System Monitor**
```python
import psutil

def system_status():
    memory = psutil.virtual_memory()
    storage = psutil.disk_usage('/')

    print("💾 System Status:")
    print(f"   RAM: {memory.percent:.1f}% ({memory.available/1024**3:.1f}GB free)")
    print(f"   Storage: {storage.percent:.1f}% ({storage.free/1024**3:.1f}GB free)")
```

---

## 🚨 **7. Hibaelhárítás**

### **Memória Hiba**
```python
# Csökkentett batch size
config.EMBEDDING_BATCH_SIZE = 32  # Vagy 16, 8

# Csökkentett max length
config.EMBEDDING_MAX_LENGTH = 512
```

### **CUDA Hiba**
```python
# CUDA verzió ellenőrzése
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

# GPU elérhetőség
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### **Model Betöltés Hiba**
```python
# Fallback model
try:
    model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
except:
    print("🔄 Fallback to smaller model")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
```

### **Index Betöltés Hiba**
```python
# Index méret ellenőrzése
import os
faiss_size = os.path.getsize("/workspace/data/index/faiss_index.bin") / 1024**3
bm25_size = os.path.getsize("/workspace/data/index/bm25_index.json") / 1024**3

print(f"FAISS index: {faiss_size:.1f} GB")
print(f"BM25 index: {bm25_size:.1f} GB")

if bm25_size > 8:
    print("⚠️ BM25 túl nagy, csak FAISS használata")
```

---

## 📈 **8. Performance Optimalizáció**

### **GPU Memory Optimalizáció**
```python
# FP16 használat mindenhol
torch.set_default_dtype(torch.float16)

# Gradient checkpointing
model.gradient_checkpointing_enable()

# Mixed precision training
from torch.cuda.amp import GradScaler
scaler = GradScaler()
```

### **Batch Size Optimalizáció**
```python
# Dinamikus batch size
def get_optimal_batch_size():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 24:  # 5090
            return 128
        elif gpu_memory >= 12:  # 4090
            return 64
        else:
            return 32
    return 16

config.EMBEDDING_BATCH_SIZE = get_optimal_batch_size()
```

---

## 🎉 **9. Termelési Használat**

### **Production Query**
```python
# Production ready query
def production_query(query: str, rerank: bool = True):
    retriever = HybridRetriever()

    if rerank:
        reranker = GRPOReranker()
        reranker.load_policy(config.RL_POLICY_PATH)

        # Baseline retrieval
        baseline = retriever.retrieve(query, top_k=100, fusion_method="rrf")

        # Reranking
        reranked = reranker.rerank(baseline)[:10]

        return reranked
    else:
        return retriever.retrieve(query, top_k=10, fusion_method="rrf")
```

### **Monitoring**
```python
# Log minden query
import logging
logging.basicConfig(filename='courtrankrl.log', level=logging.INFO)

def log_query(query, results):
    logging.info(f"Query: {query}, Results: {len(results)}")
    for doc_id in results:
        logging.info(f"  - {doc_id}")
```

---

## 🚀 **10. RunPod Specific Tips**

### **Cost Optimization**
- **Spot instances**: 50-70% olcsóbb, de preemptible
- **Storage**: Csak a szükséges fájlokat tartsd
- **Idle timeout**: 30 perc inaktivitás után leáll

### **File Management**
```bash
# Fájlok tisztítása
rm -rf /workspace/temp_files/
rm -rf /workspace/__pycache__/

# Csak a szükséges fájlok megtartása
ls -lh /workspace/data/index/
```

### **Network Optimization**
```python
# Model cache
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/transformers

# Offline mode (ha már letöltötted)
export TRANSFORMERS_OFFLINE=1
```

---

## 🎯 **11. Next Steps**

1. **✅ Setup Complete**: Fájlok feltöltve, GPU elérhető
2. **🔄 Run Pipeline**: Teljes pipeline tesztelése
3. **📊 Query Testing**: Különböző jogi területek tesztelése
4. **🏃 Training**: GRPO policy tanítása
5. **🚀 Production**: Termelési deployment

**Gratulálok! A CourtRankRL készen áll a RunPod 5090 GPU-n való használatra!** 🎉

---

## 📞 **Support**

Ha problémák merülnek fel:
1. Ellenőrizd a GPU elérhetőséget: `torch.cuda.is_available()`
2. Nézd meg a memory használatot: `nvidia-smi`
3. Ellenőrizd a fájlok elérhetőségét: `ls -lh /workspace/data/index/`
4. Próbáld a fallback modelt: `sentence-transformers/all-MiniLM-L6-v2`

**Sikeres RunPod deployment-et!** 🚀
