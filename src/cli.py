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
from src.data_loader.export_training_slates import main as export_slates
from src.search.hybrid_search import HybridRetriever
from src.search.grpo_reranker import GRPOReranker

def build_command():
    """Build pipeline: Docling → chunking → BM25."""
    print("=== COURTRANKRL BUILD PIPELINE ===")

    try:
        # Step 1: Ingestion with Docling
        print("📄 Step 1: Docling ingestion and normalization...")
        print("   (DOCX parsing, minimal normalization)")
        build_docs()

        # Step 2: Chunking
        print("✂️  Step 2: Intelligent chunking...")
        print("   (Docling decides chunk size and overlap)")

        # Step 3: BM25 indexing
        print("🔍 Step 3: BM25S index (bm25s.tokenize + cache)...")
        build_bm25()

        print("\n✅ BUILD PIPELINE COMPLETE!")
        print("📊 Generated artifacts:")
        print(f"   📄 Chunks: {config.CHUNKS_JSONL}")
        print(f"   🔍 BM25 Index: {config.BM25_INDEX_PATH}")

        print("\n🚀 Ready for queries! Use: uv run courtrankrl query \"your question\"")
        print("📝 Note: FAISS index and embeddings should be generated using gemma_embedding_runpod.ipynb")

    except Exception as e:
        print(f"\n❌ BUILD FAILED: {e}")
        print("💡 Check the error message above and try again.")
        sys.exit(1)

def export_slate_command():
    """Export training slates for GRPO cloud training."""
    print("=== COURTRANKRL SLATE EXPORT ===")
    print("🎯 Training slates exportálása GRPO cloud traininghez")
    print("📁 Kimenet: data/models/grpo_policy/training_slates.jsonl")

    try:
        export_slates()
        print("\n✅ SLATE EXPORT KÉSZ!")
        print("📊 Exportált fájlok:")
        print(f"   📄 Training slates: {config.GRPO_SLATE_EXPORT_PATH}")
        print("\n🚀 Készen a cloud trainingre!")
        print("💡 Futtasd: notebooks/grpo_train_runpod.ipynb a Runpod-on")

    except Exception as e:
        print(f"\n❌ SLATE EXPORT FAILED: {e}")
        print("💡 Győződj meg róla, hogy a qrels és chunks fájlok léteznek.")
        sys.exit(1)

def query_command(query: str, top_k: int = 10, rerank: bool = True):
    """Query pipeline: embed query → BM25 + dense → fusion → RL reranking → doc IDs."""
    print("=== COURTRANKRL QUERY PIPELINE ===")
    print(f"🔍 Lekérdezés: {query}")
    print(f"📊 Top-K: {top_k}")
    print(f"🧠 Reranking: {'bekapcsolva' if rerank else 'kikapcsolva'}")

    try:
        # Initialize retriever
        retriever = HybridRetriever()

        if rerank:
            # Step 1: Get candidates for reranking (agents.md step 4)
            print("📋 Step 1: Retrieving candidates for reranking...")
            retriever.retrieve_candidates(query, top_k=config.TOP_K_BASELINE)
            bm25_results = retriever.get_last_doc_scores("bm25")
            dense_results = retriever.get_last_doc_scores("dense")

            print(f"   📄 BM25 jelöltek: {len(bm25_results)}")
            print(f"   🧠 Dense jelöltek: {len(dense_results)}")

            # Step 2: Apply GRPO reranking (agents.md step 5)
            print("🎯 Step 2: Applying GRPO reranking...")
            try:
                reranker = GRPOReranker()
                # Load chunks data for context
                chunks_data = {}
                if config.CHUNKS_JSONL.exists():
                    import json
                    with open(config.CHUNKS_JSONL, 'r', encoding='utf-8') as f:
                        for line in f:
                            chunk = json.loads(line.strip())
                            chunks_data[chunk['chunk_id']] = chunk

                reranked = reranker.rerank(retriever, bm25_results, dense_results, query, chunks_data)[:top_k]
                print(f"   ✅ Újrendezett lista: {len(reranked)} dokumentum")

                print("\n🎯 RERANKELT EREDMÉNYEK:")
                for idx, (doc_id, score) in enumerate(reranked, start=1):
                    print(f"{idx}. {doc_id} (GRPO pont: {score:.4f})")

            except Exception as e:
                print(f"⚠️  GRPO reranker unavailable ({e}), falling back to baseline...")
                rerank = False

        if not rerank:
            # Step 1: Hybrid baseline retrieval (agents.md step 4)
            print("📋 Step 1: Hybrid baseline retrieval...")
            doc_ids = retriever.retrieve(query, top_k=top_k, fusion_method="rrf")

            print(f"   📄 Találatok száma: {len(doc_ids)}")

            print("\n🔍 BASELINE EREDMÉNYEK:")
            for idx, doc_id in enumerate(doc_ids, start=1):
                print(f"{idx}. {doc_id}")

        print(f"\n✅ Query completed successfully!")

    except Exception as e:
        print(f"\n❌ QUERY FAILED: {e}")
        print("💡 Make sure indexes are built: uv run courtrankrl build")
        sys.exit(1)

def train_command():
    """Train GRPO reranker using cloud notebook."""
    print("=== COURTRANKRL GRPO TRAINING PIPELINE ===")
    print("Agents.md spec: 5) RL Reranking (GRPO-style)")
    print("🎯 Cél: a baseline javítása megerősítéses tanulással")
    print("☁️  Training a cloud GPU-n (Runpod) keresztül")

    try:
        # Check prerequisites
        if not config.BASELINE_QRELS_FILE.exists():
            print(f"❌ Qrels fájl nem található: {config.BASELINE_QRELS_FILE}")
            print("💡 Hozz létre qrels fájlt: data/qrels/baseline_qrels.tsv")
            sys.exit(1)

        if not config.CHUNKS_JSONL.exists():
            print(f"❌ Chunks fájl nem található: {config.CHUNKS_JSONL}")
            print("💡 Futtasd előbb: uv run courtrankrl build")
            sys.exit(1)

        print("📋 Prerequisites check: ✅")
        print("📚 Tanító adatok előkészítése...")

        # Export training slates
        export_slates()

        print("\n✅ TRAINING DATA ELŐKÉSZÍTVE!")
        print("📊 Exportált fájlok:")
        print(f"   📄 Training slates: {config.GRPO_SLATE_EXPORT_PATH}")

        print("\n🚀 KÖVETKEZŐ LÉPÉSEK:")
        print("1. Másold át a training slates fájlt a Runpod workspace-be")
        print("2. Futtasd a notebook-ot: notebooks/grpo_train_runpod.ipynb")
        print("3. Másold vissza az artifacts könyvtárat: data/models/grpo_policy/")
        print("4. Teszteld a query parancsot --rerank opcióval")

        print("\n💡 Runpod workflow:")
        print("   - Environment variables: HF_TOKEN beállítva")
        print("   - Mixed precision: bf16 engedélyezve")
        print("   - Artifacts sync: /workspace/artifacts/grpo_policy/ → local data/models/grpo_policy/")

    except Exception as e:
        print(f"\n❌ GRPO TRAINING PREP FAILED: {e}")
        print("💡 Check prerequisites and try again.")
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

  # Keresés baseline módban
  uv run courtrankrl query "családi jogi ügy" --no-rerank

  # Keresés GRPO reranking-gal
  uv run courtrankrl query "szerződéses jog" --top-k 5

  # Training slates exportálása
  uv run courtrankrl export-slate

  # GRPO cloud training előkészítése
  uv run courtrankrl train

Használat előtt:
  1. uv run courtrankrl build
  2. Generate FAISS index using gemma_embedding_runpod.ipynb
  3. Export training slates: uv run courtrankrl export-slate
  4. Cloud GRPO training: notebooks/grpo_train_runpod.ipynb
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
        help='Keresés a rendszerben dokumentum azonosítókkal'
    )
    query_parser.add_argument(
        'query',
        help='Keresési lekérdezés'
    )
    query_parser.add_argument(
        '--top-k', type=int, default=10,
        help='Visszaadandó dokumentumok száma (alap: 10)'
    )
    query_parser.add_argument(
        '--fusion-method', choices=['rrf', 'zscore'], default='rrf',
        help='Fusion method: rrf vagy zscore (alap: rrf)'
    )
    query_parser.add_argument(
        '--no-rerank', action='store_true',
        help='GRPO reranking kikapcsolása (csak baseline keresés)'
    )

    # Export slate command
    export_slate_parser = subparsers.add_parser(
        'export-slate',
        help='Training slates exportálása GRPO cloud traininghez'
    )

    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='GRPO reranker cloud training előkészítése'
    )

    # Parse arguments
    args = parser.parse_args()

    # Route to appropriate command
    if args.command == 'build':
        build_command()
    elif args.command == 'query':
        query_command(args.query, args.top_k, not args.no_rerank)
    elif args.command == 'export-slate':
        export_slate_command()
    elif args.command == 'train':
        train_command()
    else:
        parser.print_help()
        print("\n💡 Kezdéshez: uv run courtrankrl build")

if __name__ == '__main__':
    main()
