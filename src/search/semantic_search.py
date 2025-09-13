#!/usr/bin/env python3
"""
Query Interface for CourtRankRL
Simple interface to test the retrieval system.
"""

from configs import config
from src.search.hybrid_search import HybridRetriever
from src.search.grpo_reranker import GRPOReranker

def main():
    """Simple query interface."""
    print("CourtRankRL Query Interface")
    print("===========================")

    # Initialize components
    retriever = HybridRetriever()
    reranker = GRPOReranker()
    reranker.load_policy(config.RL_POLICY_PATH)

    # Test query
    query = "családi jogi ügy"

    print(f"Query: {query}")

    # Get baseline results
    baseline = retriever.retrieve(query, top_k=5)

    # Get reranked results
    candidates = retriever.retrieve_candidates(query, top_k=config.TOP_K_BASELINE)
    reranked = reranker.rerank(candidates)[:5]

    print("\n=== BASELINE RESULTS ===")
    for i, (chunk_id, score) in enumerate(baseline, 1):
        print(f"{i}. {chunk_id} (score: {score:.4f})")

    print("\n=== RERANKED RESULTS ===")
    for i, (chunk_id, score) in enumerate(reranked, 1):
        print(f"{i}. {chunk_id} (score: {score:.4f})")

if __name__ == '__main__':
    main() 