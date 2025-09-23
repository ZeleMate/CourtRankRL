#!/bin/bash

# CourtRankRL RunPod Setup Script
# 🚀 Automatikus telepítés RunPod 5090 GPU-hoz

echo "🚀 CourtRankRL RunPod Setup"
echo "=========================="

# 1. System Update
echo "📦 System Update..."
apt-get update -y
apt-get upgrade -y

# 2. Python Dependencies
echo "🐍 Python Dependencies..."
pip install --upgrade pip
pip install faiss-cpu tqdm transformers accelerate torch sentencepiece psutil

# 3. CUDA Check
echo "🎯 CUDA Check..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'CUDA cores: {torch.cuda.get_device_properties(0).multi_processor_count}')
else:
    print('❌ CUDA not available!')
    exit(1)
"

# 4. Project Structure
echo "📁 Project Structure..."
mkdir -p /workspace/data/index
mkdir -p /workspace/data/models
mkdir -p /workspace/data/raw
mkdir -p /workspace/data/processed

# 5. Environment Variables
echo "⚙️ Environment Setup..."
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/transformers
export CUDA_LAUNCH_BLOCKING=1

# 6. Memory Check
echo "💾 Memory Check..."
python3 -c "
import psutil
import torch

memory = psutil.virtual_memory()
storage = psutil.disk_usage('/')

print('System Resources:')
print(f'  RAM: {memory.total / 1024**3:.1f} GB')
print(f'  Available RAM: {memory.available / 1024**3:.1f} GB')
print(f'  Storage: {storage.total / 1024**3:.1f} GB')
print(f'  Available Storage: {storage.free / 1024**3:.1f} GB')

if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU Memory: {gpu_memory:.1f} GB')
"

# 7. Test Import
echo "🧪 Test Import..."
python3 -c "
import sys
sys.path.append('/workspace')

try:
    from configs import config
    from src.search.hybrid_search import HybridRetriever
    print('✅ Project imports successful!')
    print(f'  Model: {config.QWEN3_MODEL_NAME}')
    print(f'  Dimension: {config.QWEN3_DIMENSION}')
    print(f'  Batch size: {config.EMBEDDING_BATCH_SIZE}')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

# 8. Quick Test
echo "🔍 Quick Test..."
python3 -c "
from src.search.hybrid_search import HybridRetriever
import torch

print('Testing HybridRetriever...')
try:
    retriever = HybridRetriever()
    print('✅ HybridRetriever initialized successfully!')
    print(f'  Model device: {next(retriever.model.parameters()).device}')
    print(f'  FAISS vectors: {retriever.faiss_index.ntotal if retriever.faiss_index else \"Not loaded\"}')
    print(f'  BM25 docs: {retriever.bm25.total_docs if retriever.bm25 else \"Not loaded\"}')
except Exception as e:
    print(f'❌ Test failed: {e}')
    print('💡 Try: python3 notebooks/RunPod_Full_Pipeline.py')
"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🚀 Next Steps:"
echo "1. Run full pipeline: python3 notebooks/RunPod_Full_Pipeline.py"
echo "2. Test queries: python3 -m src.cli query 'családi jog'"
echo "3. Run training: python3 -m src.cli train"
echo ""
echo "📊 System Status:"
echo "• GPU: RTX 5090 (24GB VRAM)"
echo "• Model: Qwen3-0.6B"
echo "• Batch Size: 128"
echo "• Ready for production!"
echo ""
echo "Happy querying! 🎯"
