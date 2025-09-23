#!/usr/bin/env python3
"""
CourtRankRL CLI Interface
Agents.md specifikáció alapján implementálva.

Parancsok:
- build: subset selection → Docling feldolgozás → chunking → BM25 → EmbeddingGemma FAISS
- query: hybrid keresés + GRPO reranking (opcionális)
- train: GRPO reranker policy tanítása
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
    print("Agents.md spec: 1→2→3")

    try:
        # Step 1: Ingestion with Docling
        print("📄 Step 1: Docling ingestion and normalization...")
        print("   (DOCX parsing, minimal normalization)")
        build_docs()

        # Step 2: Chunking
        print("✂️  Step 2: Intelligent chunking...")
        print("   (Docling decides chunk size and overlap)")

        # Step 3: BM25 indexing
        print("🔍 Step 3: BM25 sparse indexing...")
        print("   (Tokenization via simple split(), postings, doc_len/avg_len, idf cache)")
        build_bm25()

        print("\n✅ BUILD PIPELINE COMPLETE!")
        print("📊 Generated artifacts:")
        print(f"   📄 Chunks: {config.CHUNKS_JSONL}")
        print(f"   🔍 BM25 Index: {config.BM25_INDEX_PATH}")

        print("\n🚀 Ready for queries! Use: uv run courtrankrl query \"your question\"")
        print("📝 Note: FAISS index and embeddings should be generated using qwen_embedding_runpod.ipynb")

    except Exception as e:
        print(f"\n❌ BUILD FAILED: {e}")
        print("💡 Check the error message above and try again.")
        sys.exit(1)

def query_command(query: str, top_k: int = 10, rerank: bool = True):
    """Query pipeline: embed query → BM25 + dense → fusion → RL reranking → doc IDs."""
    print("=== COURTRANKRL QUERY PIPELINE ===")
    print(f"🔍 Query: {query}")
    print(f"📊 Top-K: {top_k}")
    print(f"🧠 Reranking: {'Enabled' if rerank else 'Disabled'}")

    try:
        # Initialize retriever
        retriever = HybridRetriever()

        if rerank:
            # Step 1: Get candidates for reranking (agents.md step 4)
            print("📋 Step 1: Retrieving candidates for reranking...")
            bm25_results, dense_results = retriever.retrieve_candidates(query, top_k=config.TOP_K_BASELINE)

            print(f"   📄 BM25 candidates: {len(bm25_results)}")
            print(f"   🧠 Dense candidates: {len(dense_results)}")

            # Step 2: Apply GRPO reranking (agents.md step 5)
            print("🎯 Step 2: Applying GRPO reranking...")
            try:
                reranker = GRPOReranker()
                reranker.load_policy(config.RL_POLICY_PATH)
                reranked_results = reranker.rerank(bm25_results, dense_results)[:top_k]

                print(f"   ✅ Reranked to: {len(reranked_results)} documents")

                # Step 3: Output reranked results
                print("\n🎯 RERANKED RESULTS (Document IDs):")
                for i, (doc_id, score) in enumerate(reranked_results, 1):
                    print(f"{i}. {doc_id}")

            except Exception as e:
                print(f"⚠️  Reranker unavailable ({e}), falling back to baseline...")
                rerank = False

        if not rerank:
            # Step 1: Hybrid baseline retrieval (agents.md step 4)
            print("📋 Step 1: Hybrid baseline retrieval...")
            doc_ids = retriever.retrieve(query, top_k=top_k, fusion_method="rrf")

            print(f"   📄 Retrieved documents: {len(doc_ids)}")

            # Step 2: Output baseline results
            print("\n🔍 BASELINE RESULTS (Document IDs):")
            for i, doc_id in enumerate(doc_ids, 1):
                print(f"{i}. {doc_id}")

        print(f"\n✅ Query completed successfully!")

    except Exception as e:
        print(f"\n❌ QUERY FAILED: {e}")
        print("💡 Make sure indexes are built: uv run courtrankrl build")
        sys.exit(1)

def train_command():
    """Train GRPO reranker: load qrels → baseline candidates → features → GRPO training."""
    print("=== COURTRANKRL GRPO TRAINING PIPELINE ===")
    print("Agents.md spec: 5) RL Reranking (GRPO-style)")
    print("🎯 Goal: Improve ranking quality via reinforcement learning")

    try:
        from src.search.grpo_reranker import main as train_rl

        print("📚 Loading training data...")
        print("🎮 Initializing GRPO reranker...")
        print("🏃 Starting training...")

        train_rl()

        print("\n✅ GRPO TRAINING COMPLETE!")
        print("📊 Training results:")
        print(f"   🧠 Policy saved: {config.RL_POLICY_PATH}")
        print("   📈 Ready for improved reranking!")
        print("\n🚀 Use with queries: uv run courtrankrl query \"question\" --rerank")

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

  # Keresés alap (baseline) módban
  uv run courtrankrl query "családi jogi ügy" --no-rerank

  # Keresés GRPO reranking-gal
  uv run courtrankrl query "szerződéses jog" --top-k 5

  # GRPO policy tanítása
  uv run courtrankrl train

Használat előtt:
  1. uv run courtrankrl build
  2. Generate FAISS index using qwen_embedding_runpod.ipynb
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
