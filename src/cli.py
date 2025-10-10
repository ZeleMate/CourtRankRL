#!/usr/bin/env python3
"""
CourtRankRL CLI Interface
Agents.md specifikáció alapján implementálva.

Parancsok:
- build: Docling feldolgozás → chunking → BM25 index
- query: hibrid (BM25+FAISS) keresés, opcionális GRPO reranking
- train: GRPO reranker policy tanítása (baseline → feature export → GRPO)
"""

import argparse
import sys
from pathlib import Path
from typing import List

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from src.data_loader.preprocess_documents import main as build_docs
from src.data_loader.build_bm25_index import main as build_bm25
from src.search.hybrid_search import HybridRetriever
from src.search.grpo_reranker import load_qrels, prepare_training_slates

def build_command():
    """Build pipeline: Docling → chunking → BM25."""
    print("=== COURTRANKRL BUILD PIPELINE ===")

    try:
        # 1. lépés: Docling feldolgozás
        print("1. lépés: Docling feldolgozás és normalizálás...")
        print("   (DOCX parsing, minimális normalizálás)")
        build_docs()

        # 2. lépés: Chunkolás
        print("2. lépés: Intelligens chunkolás...")
        print("   (Docling dönti el a chunk méretet és átfedést)")

        # 3. lépés: BM25 indexelés
        print("3. lépés: BM25S index (bm25s.tokenize + cache)...")
        build_bm25()

        print("\nBUILD PIPELINE KÉSZ!")
        print("Generált artifaktumok:")
        print(f"   Chunks: {config.CHUNKS_JSONL}")
        print(f"   BM25 Index: {config.BM25_INDEX_PATH}")

        print("\nKész a lekérdezésre! Használat: uv run courtrankrl query \"kérdés\"")
        print("Megjegyzés: FAISS index generálása szükséges a gemma_embedding_runpod.ipynb futtatásával")

    except Exception as e:
        print(f"\nBUILD SIKERTELEN: {e}")
        print("Ellenőrizd a hibaüzenetet és próbáld újra.")
        sys.exit(1)

def query_command(query: str, top_k: int = 10):
    """Query pipeline: embed query → BM25 + dense → RRF fusion → doc IDs (baseline only)."""
    print("=== COURTRANKRL QUERY PIPELINE ===")
    print(f"Lekérdezés: {query}")
    print(f"Top-K: {top_k}")
    print(f"Fusion: RRF (Reciprocal Rank Fusion)")

    try:
        # Retriever inicializálása
        retriever = HybridRetriever()
        
        print("Megjegyzés: GRPO reranking csak cloud-on (agents.md specifikáció szerint)")

        # Hibrid baseline retrieval (agents.md 3. lépés)
        print("Hibrid baseline retrieval...")
        doc_ids = retriever.retrieve(query, top_k=top_k)

        print(f"   Találatok száma: {len(doc_ids)}")

        print("\nBASELINE EREDMÉNYEK:")
        for idx, doc_id in enumerate(doc_ids, start=1):
            print(f"{idx}. {doc_id}")

        print(f"\nQuery sikeresen befejezve!")

    except Exception as e:
        print(f"\nQUERY SIKERTELEN: {e}")
        print("Ellenőrizd, hogy az indexek elkészültek: uv run courtrankrl build")
        sys.exit(1)

def train_command():
    """Train GRPO reranker: load qrels → baseline candidates → features → GRPO training."""
    print("=== COURTRANKRL GRPO TRAINING PIPELINE ===")
    print("Agents.md spec: 5) RL Reranking (GRPO-style)")
    print("Cél: a baseline javítása megerősítéses tanulással")

    try:
        # 1. lépés: Qrels betöltése
        print("1. lépés: Qrels betöltése...")
        qrels = load_qrels(config.QRELS_FILE)

        if not qrels:
            print(f"Nincs qrels adat: {config.QRELS_FILE}")
            return

        print(f"   Betöltött query-k: {len(qrels)}")

        # 2. lépés: Retriever inicializálása
        print("2. lépés: Retriever inicializálása...")
        retriever = HybridRetriever()

        # 3. lépés: Slate export cloud traininghez
        print("3. lépés: Slate export cloud traininghez...")
        slates = prepare_training_slates(retriever, qrels, config.SLATE_EXPORT_PATH)

        print("4. lépés: Cloud training notebook futtatása...")
        print(f"   Notebook: {config.GRPO_TRAIN_NOTEBOOK}")
        print("   RunPod-on futtassa: grpo_train_runpod.ipynb")
        print("   Input: training_slates.jsonl (kész)")

        print("\nGRPO TRAINING ELŐKÉSZÍTVE!")
        print("Következő lépések:")
        print("   1. Másolja a slates fájlt RunPod-ra")
        print(f"   2. Futtassa: {config.GRPO_TRAIN_NOTEBOOK}")
        print("   3. Töltse le az adapter fájlokat")
        print("   4. Helyezze el: data/models/grpo_policy/")

        print("\nArtifactumok:")
        print(f"   Slates: {config.SLATE_EXPORT_PATH}")
        print(f"   Adapter: {config.GRPO_ADAPTER_PATH} (RunPod-ról)")
        print(f"   Tokenizer: {config.GRPO_TOKENIZER_PATH} (RunPod-ról)")
        print(f"   Metrics: {config.GRPO_METRICS_PATH} (RunPod-ról)")

    except Exception as e:
        print(f"\nGRPO TRAINING ELŐKÉSZÍTÉS SIKERTELEN: {e}")
        print("Ellenőrizd, hogy a qrels fájl létezik és az indexek elkészültek.")
        sys.exit(1)

def main():
    """Main CLI entry point for CourtRankRL."""
    parser = argparse.ArgumentParser(
        description="CourtRankRL - Magyar bírósági döntések retrieval rendszer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Példák használatra:

  # Build pipeline futtatása
  uv run courtrankrl build

  # Keresés (RRF fusion)
  uv run courtrankrl query "családi jogi ügy"

  # Több találat kérése
  uv run courtrankrl query "szerződéses jog" --top-k 20

  # GRPO slate export (cloud training előkészítés)
  uv run courtrankrl train

AJÁNLOTT WORKFLOW (szakdolgozat):
  1. uv run courtrankrl build
  2. Generate FAISS index: gemma_embedding_runpod.ipynb
  3. Baseline eval: notebooks/baseline_evaluation.ipynb
  4. GRPO training: grpo_train_runpod.ipynb (cloud)
  
Megjegyzés: RRF (Reciprocal Rank Fusion) baseline paraméter-mentesen működik
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Elérhető parancsok')

    # Build command
    build_parser = subparsers.add_parser(
        'build',
        help='Build pipeline: Docling → chunking → BM25 indexing'
    )

    # Query command
    query_parser = subparsers.add_parser(
        'query',
        help='Keresés RRF fúzióval'
    )
    query_parser.add_argument(
        'query',
        help='Keresési lekérdezés'
    )
    query_parser.add_argument(
        '--top-k', type=int, default=10,
        help='Visszaadandó dokumentumok száma (alap: 10)'
    )

    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='GRPO reranker policy tanítása'
    )

    # Parse arguments
    args = parser.parse_args()

    # Route to appropriate command
    if args.command == 'build':
        build_command()
    elif args.command == 'query':
        query_command(args.query, args.top_k)
    elif args.command == 'train':
        train_command()
    else:
        parser.print_help()
        print("\nKezdéshez: uv run courtrankrl build")

if __name__ == '__main__':
    main()
