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
                reranker.load_policy(config.RL_POLICY_PATH)
                reranked = reranker.rerank(retriever, bm25_results, dense_results)[:top_k]
                print(f"   ✅ Újrendezett lista: {len(reranked)} dokumentum")

                print("\n🎯 RERANKELT EREDMÉNYEK:")
                for idx, (doc_id, score) in enumerate(reranked, start=1):
                    print(f"{idx}. {doc_id} (sztochasztikus pont: {score:.4f})")

            except Exception as e:
                print(f"⚠️  Reranker unavailable ({e}), falling back to baseline...")
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
    """Train GRPO reranker: load qrels → baseline candidates → features → GRPO training."""
    print("=== COURTRANKRL GRPO TRAINING PIPELINE ===")
    print("Agents.md spec: 5) RL Reranking (GRPO-style)")
    print("🎯 Cél: a baseline javítása megerősítéses tanulással")

    try:
        from src.search.grpo_reranker import main as train_rl

        print("📚 Tanító adatok betöltése...")
        print("🎮 GRPO reranker inicializálása...")
        print("🏃 Tanítás indítása...")

        train_rl()

        print("\n✅ GRPO TANÍTÁS KÉSZ!")
        print("📊 Eredmények:")
        print(f"   🧠 Policy mentve: {config.RL_POLICY_PATH}")
        print("   📈 Használd kereséskor a --rerank kapcsolóval!")

    except Exception as e:
        print(f"\n❌ GRPO TRAINING FAILED: {e}")
        print("💡 Make sure qrels file exists and indexes are built.")
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

  # GRPO policy tanítása
  uv run courtrankrl train

Használat előtt:
  1. uv run courtrankrl build
  2. Generate FAISS index using gemma_embedding_runpod.ipynb
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
        query_command(args.query, args.top_k, not args.no_rerank)
    elif args.command == 'train':
        train_command()
    else:
        parser.print_help()
        print("\n💡 Kezdéshez: uv run courtrankrl build")

if __name__ == '__main__':
    main()
