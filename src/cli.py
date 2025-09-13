#!/usr/bin/env python3
"""
CLI Interface for CourtRankRL
Implements build, query, and train commands as per specification.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from configs import config
from src.data_loader.preprocess_documents import main as build_docs
from src.data_loader.build_bm25_index import main as build_bm25
from src.embedding.create_embeddings_gemini_api import main as build_dense
from src.search.hybrid_search import HybridRetriever
from src.search.grpo_reranker import GRPOReranker

def build_command():
    """Build command: parse → clean → chunk → embed → build indexes."""
    print("=== COURTRANKRL BUILD COMMAND ===")

    try:
        # Step 1: Parse and chunk documents
        print("Step 1: Parsing documents and creating chunks...")
        build_docs()

        # Step 2: Build BM25 sparse index
        print("Step 2: Building BM25 sparse index...")
        build_bm25()

        # Step 3: Build FAISS dense index
        print("Step 3: Building FAISS dense index...")
        build_dense()

        print("✅ Build complete!")
        print(f"   - Chunks: {config.CHUNKS_JSONL}")
        print(f"   - BM25 Index: {config.BM25_INDEX_PATH}")
        print(f"   - FAISS Index: {config.FAISS_INDEX_PATH}")

    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)

def query_command(query: str, top_k: int = 10, rerank: bool = True):
    """Query command: embed query → retrieve candidates → fuse → return ranked chunks."""
    print(f"=== COURTRANKRL QUERY COMMAND ===")
    print(f"Query: {query}")
    print(f"Top-K: {top_k}")

    try:
        # Initialize retriever
        retriever = HybridRetriever()

        # Retrieve baseline results
        print("Retrieving candidates...")
        baseline_results = retriever.retrieve(query, top_k=top_k)

        print(f"Baseline results: {len(baseline_results)} chunks")

        if rerank:
            # Load and apply reranker
            reranker = GRPOReranker()
            reranker.load_policy(config.RL_POLICY_PATH)

            # Prepare candidates for reranking
            candidates = retriever.retrieve_candidates(query, top_k=config.TOP_K_BASELINE)
            reranked_results = reranker.rerank(candidates)[:top_k]

            print(f"Reranked results: {len(reranked_results)} chunks")

            # Print reranked results
            print("\n=== RERANKED RESULTS ===")
            for i, (chunk_id, score) in enumerate(reranked_results, 1):
                print(f"{i:2d}. {chunk_id} (score: {score:.4f})")

        else:
            # Print baseline results
            print("\n=== BASELINE RESULTS ===")
            for i, (chunk_id, score) in enumerate(baseline_results, 1):
                print(f"{i:2d}. {chunk_id} (score: {score:.4f})")

    except Exception as e:
        print(f"❌ Query failed: {e}")
        sys.exit(1)

def train_command():
    """Train RL command: load qrels → retrieve baseline → compute features → train policy."""
    print("=== COURTRANKRL TRAIN RL COMMAND ===")

    try:
        from src.search.grpo_reranker import main as train_rl
        train_rl()
        print("✅ RL training complete!")
        print(f"   - Policy saved: {config.RL_POLICY_PATH}")

    except Exception as e:
        print(f"❌ RL training failed: {e}")
        sys.exit(1)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="CourtRankRL CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Build command
    subparsers.add_parser('build', help='Build indexes from documents')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('--top-k', type=int, default=10,
                             help='Number of results to return')
    query_parser.add_argument('--no-rerank', action='store_true',
                             help='Disable RL reranking')

    # Train command
    subparsers.add_parser('train', help='Train RL reranking policy')

    args = parser.parse_args()

    if args.command == 'build':
        build_command()
    elif args.command == 'query':
        query_command(args.query, args.top_k, not args.no_rerank)
    elif args.command == 'train':
        train_command()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
