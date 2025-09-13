#!/usr/bin/env python3
"""
GRPO-style RL Reranker for CourtRankRL
Trains a policy network to improve ranking quality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config

class RankingPolicy(nn.Module):
    """Neural network policy for ranking."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output ranking score 0-1
        )

    def forward(self, x):
        return self.network(x)

class GRPOReranker:
    """GRPO-style reranker for improving search rankings."""

    def __init__(self):
        self.policy = RankingPolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.RL_LEARNING_RATE)
        self.device = torch.device('cpu')  # Keep it simple, no GPU

    def extract_features(self, candidates: List[Tuple[str, float, str]],
                        query: str) -> List[np.ndarray]:
        """
        Extract features for each candidate:
        [dense_similarity, bm25_score, rank_difference]
        """
        features = []

        # Separate BM25 and dense scores
        bm25_scores = {}
        dense_scores = {}

        for chunk_id, score, method in candidates:
            if method == 'bm25':
                bm25_scores[chunk_id] = score
            elif method == 'dense':
                dense_scores[chunk_id] = score

        # Get all unique chunk IDs
        all_chunk_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

        for chunk_id in all_chunk_ids:
            dense_score = dense_scores.get(chunk_id, 0.0)
            bm25_score = bm25_scores.get(chunk_id, 0.0)

            # Simple rank difference (placeholder - would need full ranking)
            rank_diff = abs(dense_score - bm25_score)

            features.append(np.array([dense_score, bm25_score, rank_diff]))

        return features

    def calculate_ndcg(self, predicted_ranks: List[int], true_relevance: List[int],
                       k: int = 10) -> float:
        """Calculate NDCG@k."""
        if not true_relevance:
            return 0.0

        # DCG
        dcg = 0.0
        for i in range(min(k, len(predicted_ranks))):
            if i < len(true_relevance):
                rel = true_relevance[predicted_ranks[i]]
                dcg += rel / np.log2(i + 2)

        # IDCG (ideal DCG)
        sorted_rel = sorted(true_relevance, reverse=True)
        idcg = 0.0
        for i in range(min(k, len(sorted_rel))):
            idcg += sorted_rel[i] / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def train_step(self, candidates: List[Tuple[str, float, str]],
                  qrel_data: Dict[str, int], query: str):
        """Single GRPO training step."""
        if not candidates:
            return

        # Extract features
        features = self.extract_features(candidates, query)

        if not features:
            return

        # Convert to tensor
        features_tensor = torch.FloatTensor(features)

        # Get policy scores
        with torch.no_grad():
            policy_scores = self.policy(features_tensor).squeeze()

        # Create ranking based on policy scores
        ranked_indices = torch.argsort(policy_scores, descending=True).numpy()

        # Calculate reward (NDCG improvement)
        true_relevance = [qrel_data.get(chunk_id, 0) for chunk_id, _, _ in candidates]

        current_ndcg = self.calculate_ndcg(list(range(len(candidates))), true_relevance)
        policy_ndcg = self.calculate_ndcg(ranked_indices.tolist(), true_relevance)

        reward = policy_ndcg - current_ndcg

        # GRPO-style update (simplified)
        if reward > 0:
            # Positive reward - reinforce good ranking
            loss = -torch.mean(policy_scores[ranked_indices[:10]])  # Encourage top rankings
        else:
            # Negative reward - discourage bad ranking
            loss = torch.mean(policy_scores[ranked_indices[:10]])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return reward

    def rerank(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float]]:
        """Rerank candidates using trained policy."""
        if not candidates:
            return []

        features = self.extract_features(candidates, "")

        if not features:
            return [(chunk_id, score) for chunk_id, score, _ in candidates]

        features_tensor = torch.FloatTensor(features)

        with torch.no_grad():
            policy_scores = self.policy(features_tensor).squeeze().numpy()

        # Create new ranking
        ranked_indices = np.argsort(policy_scores)[::-1]  # Descending

        reranked = []
        for idx in ranked_indices:
            chunk_id, _, _ = candidates[idx]
            new_score = policy_scores[idx].item()
            reranked.append((chunk_id, new_score))

        return reranked

    def save_policy(self, path: Path):
        """Save trained policy."""
        torch.save(self.policy.state_dict(), path)
        print(f"Policy saved to {path}")

    def load_policy(self, path: Path):
        """Load trained policy."""
        if path.exists():
            self.policy.load_state_dict(torch.load(path))
            print(f"Policy loaded from {path}")

def load_qrels(qrels_file: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels file for training."""
    qrels = {}

    if not qrels_file.exists():
        return qrels

    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                chunk_id = parts[2]
                relevance = int(parts[3])

                if query_id not in qrels:
                    qrels[query_id] = {}

                qrels[query_id][chunk_id] = relevance

    return qrels

def main():
    """Train GRPO reranker."""
    if not config.DEV_QRELS_FILE.exists():
        print(f"Error: Qrels file not found: {config.DEV_QRELS_FILE}")
        return

    # Load qrels
    qrels = load_qrels(config.DEV_QRELS_FILE)

    if not qrels:
        print("No qrels data found")
        return

    # Initialize reranker
    reranker = GRPOReranker()

    # Training loop (simplified)
    print("Training GRPO reranker...")

    for epoch in range(config.RL_EPOCHS):
        total_reward = 0.0
        num_queries = 0

        for query_id, relevance_data in qrels.items():
            # Simulate candidates (in real scenario, this would come from retrieval)
            candidates = [(chunk_id, random.random(), 'simulated')
                         for chunk_id in relevance_data.keys()]

            if candidates:
                reward = reranker.train_step(candidates, relevance_data, query_id)
                if reward is not None:
                    total_reward += reward
                    num_queries += 1

        avg_reward = total_reward / num_queries if num_queries > 0 else 0
        print(f"Epoch {epoch + 1}/{config.RL_EPOCHS}, Avg Reward: {avg_reward:.4f}")

    # Save trained policy
    reranker.save_policy(config.RL_POLICY_PATH)

if __name__ == '__main__':
    main()
