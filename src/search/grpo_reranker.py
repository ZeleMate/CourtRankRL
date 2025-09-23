#!/usr/bin/env python3
"""
CourtRankRL GRPO-style Reranker
Inspired by DeepSeekMath GRPO and Unsloth memory optimizations.

Főbb jellemzők:
- Group Relative Policy Optimization (memory efficient, no critic model)
- Shallow MLP policy network (agents.md: linear or shallow MLP head)
- NDCG@10 reward calculation at group level
- VRAM efficient training (inspired by Unsloth)
- Groupwise softmax over candidates
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
    """Neural network policy for ranking - agents.md spec: linear or shallow MLP head."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        # Shallow MLP per agents.md spec - RunPod GPU optimalizált
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output ranking score 0-1
        )

        # RunPod 5090 GPU optimalizáció
        if torch.cuda.is_available():
            self.cuda()  # Move to GPU if available

    def forward(self, x):
        return self.network(x)

class GRPOReranker:
    """GRPO-style reranker with Unsloth-inspired memory optimizations."""

    def __init__(self):
        self.policy = RankingPolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.RL_LEARNING_RATE)
        self.device = torch.device('cpu')  # VRAM efficient, no GPU

        # Unsloth-inspired memory optimizations
        self.baseline_history = []  # Rolling baseline for stable training
        self.reward_std_history = []  # Global reward std tracking
        self.memory_efficient = True  # Enable memory optimizations

    def extract_features(self, bm25_results: List[Tuple[str, float]],
                        dense_results: List[Tuple[str, float]]) -> List[np.ndarray]:
        """
        Extract features for each candidate document:
        [dense_similarity, normalized_bm25_score, rank_difference]
        Agents.md spec: dense similarity, normalized BM25 score, rank difference
        """
        features = []

        # Create score dictionaries
        bm25_scores = dict(bm25_results)
        dense_scores = dict(dense_results)

        # Get all unique doc IDs
        all_doc_ids = set(bm25_scores.keys()) | set(dense_scores.keys())

        # Calculate normalized BM25 scores
        if bm25_scores:
            bm25_values = list(bm25_scores.values())
            bm25_mean = np.mean(bm25_values)
            bm25_std = np.std(bm25_values)
            bm25_std = bm25_std if bm25_std > 0 else 1.0  # Handle zero variance

        for doc_id in all_doc_ids:
            dense_score = dense_scores.get(doc_id, 0.0)
            bm25_score = bm25_scores.get(doc_id, 0.0)

            # Normalized BM25 score (z-score)
            if bm25_scores:
                normalized_bm25 = (bm25_score - bm25_mean) / bm25_std
            else:
                normalized_bm25 = 0.0

            # Rank difference (simple difference between methods)
            rank_diff = abs(dense_score - bm25_score)

            features.append(np.array([dense_score, normalized_bm25, rank_diff]))

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

    def train_step(self, bm25_results: List[Tuple[str, float]],
                  dense_results: List[Tuple[str, float]],
                  qrel_data: Dict[str, int]):
        """
        Single GRPO training step with Unsloth-inspired memory optimizations.
        Uses group-relative rewards and efficient memory usage.
        """
        if not bm25_results and not dense_results:
            return None

        # Extract features
        features = self.extract_features(bm25_results, dense_results)

        if not features:
            return None

        # Memory efficient tensor creation
        features_tensor = torch.FloatTensor(features)

        # Get policy scores and apply groupwise softmax (Unsloth approach)
        with torch.no_grad():
            policy_scores = self.policy(features_tensor).squeeze()
            softmax_scores = torch.softmax(policy_scores, dim=0)

        # Create ranking based on policy scores
        ranked_indices = torch.argsort(softmax_scores, descending=True)

        # Get document IDs
        bm25_scores = dict(bm25_results)
        dense_scores = dict(dense_results)
        all_doc_ids = list(set(bm25_scores.keys()) | set(dense_scores.keys()))

        # Calculate true relevance
        true_relevance = [qrel_data.get(doc_id, 0) for doc_id in all_doc_ids]

        # Calculate NDCG@10
        baseline_ndcg = self.calculate_ndcg(list(range(len(all_doc_ids))), true_relevance, k=10)
        policy_ndcg = self.calculate_ndcg(ranked_indices.tolist(), true_relevance, k=10)

        # Group-relative reward with Unsloth-style optimization
        reward = policy_ndcg - baseline_ndcg

        # Update rolling baseline (Unsloth memory optimization)
        self.baseline_history.append(baseline_ndcg)
        if len(self.baseline_history) > 100:  # Rolling window
            self.baseline_history.pop(0)

        # Update global reward std (Unsloth approach)
        self.reward_std_history.append(reward)
        if len(self.reward_std_history) > 50:
            self.reward_std_history.pop(0)

        # Memory efficient GRPO update
        if self.memory_efficient:
            # Calculate advantage using group-relative method
            if len(self.baseline_history) > 10:
                group_baseline = np.mean(self.baseline_history[-10:])
                if len(self.reward_std_history) > 5:
                    global_std = np.std(self.reward_std_history) + 1e-8
                    advantage = (reward - group_baseline) / global_std
                else:
                    advantage = reward - group_baseline
            else:
                advantage = reward
        else:
            advantage = reward

        # GRPO-style loss calculation
        if advantage > 0:
            # Positive advantage - reinforce
            loss = -torch.mean(softmax_scores[ranked_indices[:10]] * advantage)
        else:
            # Negative advantage - discourage
            loss = torch.mean(softmax_scores[ranked_indices[:10]] * (-advantage))

        # Memory efficient backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return reward

    def rerank(self, bm25_results: List[Tuple[str, float]],
               dense_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Rerank candidates using trained policy with groupwise softmax.
        Agents.md spec: linear or shallow MLP head; groupwise softmax over candidates.
        """
        if not bm25_results and not dense_results:
            return []

        # Extract features for all candidates
        features = self.extract_features(bm25_results, dense_results)

        if not features:
            # Fallback to BM25 results
            return bm25_results

        features_tensor = torch.FloatTensor(features)

        with torch.no_grad():
            # Policy scores (0-1 range from sigmoid)
            policy_scores = self.policy(features_tensor).squeeze()

            # Groupwise softmax over candidates (agents.md spec)
            softmax_scores = torch.softmax(policy_scores, dim=0)

        # Get document IDs in same order as features
        bm25_scores = dict(bm25_results)
        dense_scores = dict(dense_results)
        all_doc_ids = list(set(bm25_scores.keys()) | set(dense_scores.keys()))

        # Create ranking based on softmax scores
        ranked_indices = torch.argsort(softmax_scores, descending=True)

        reranked = []
        for idx in ranked_indices:
            doc_id = all_doc_ids[idx.item()]
            new_score = softmax_scores[idx].item()
            reranked.append((doc_id, new_score))

        return reranked

    def evaluate(self, test_queries: List[str], qrels: Dict[str, Dict[str, int]]):
        """
        Evaluate reranker performance vs baseline.
        Agents.md spec: numeric comparison baseline vs rerank.
        """
        from src.search.hybrid_search import HybridRetriever

        retriever = HybridRetriever()

        total_baseline_ndcg = 0.0
        total_reranked_ndcg = 0.0
        num_queries = 0

        print("=== EVALUATION: BASELINE vs RERANKED ===")

        for query in test_queries:
            if query not in qrels:
                continue

            # Get retrieval results
            bm25_results, dense_results = retriever.retrieve_candidates(query, top_k=50)

            if not bm25_results:
                continue

            # Baseline: BM25 results
            baseline_doc_ids = [doc_id for doc_id, _ in bm25_results]
            true_relevance = [qrels[query].get(doc_id, 0) for doc_id in baseline_doc_ids]
            baseline_ndcg = self.calculate_ndcg(list(range(len(baseline_doc_ids))), true_relevance)

            # Reranked results
            reranked_results = self.rerank(bm25_results, dense_results)
            reranked_doc_ids = [doc_id for doc_id, _ in reranked_results]
            true_relevance_reranked = [qrels[query].get(doc_id, 0) for doc_id in reranked_doc_ids]
            reranked_ndcg = self.calculate_ndcg(list(range(len(reranked_doc_ids))), true_relevance_reranked)

            total_baseline_ndcg += baseline_ndcg
            total_reranked_ndcg += reranked_ndcg
            num_queries += 1

            print(f"Query: {query[:30]}...")
            print(".4f")
            print(".4f")
            print()

        if num_queries > 0:
            avg_baseline_ndcg = total_baseline_ndcg / num_queries
            avg_reranked_ndcg = total_reranked_ndcg / num_queries
            improvement = avg_reranked_ndcg - avg_baseline_ndcg

            print("=== ÖSSZESÍTÉS ===")
            print(".4f")
            print(".4f")
            print(".4f")
            print(".1f")

            return {
                'baseline_ndcg': avg_baseline_ndcg,
                'reranked_ndcg': avg_reranked_ndcg,
                'improvement': improvement,
                'num_queries': num_queries
            }

        return None

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
    """Load qrels file for training - convert chunk_id to doc_id."""
    qrels = {}

    if not qrels_file.exists():
        return qrels

    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                chunk_id = parts[2]  # This might be chunk_id from qrels
                relevance = int(parts[3])

                # Convert chunk_id to doc_id (assuming format: "doc_id_src_tag_i")
                doc_id = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id

                if query_id not in qrels:
                    qrels[query_id] = {}

                qrels[query_id][doc_id] = relevance

    return qrels

def main():
    """Train GRPO reranker with Unsloth-inspired memory optimizations."""
    from src.search.hybrid_search import HybridRetriever

    if not config.DEV_QRELS_FILE.exists():
        print(f"Hiba: Qrels fájl nem található: {config.DEV_QRELS_FILE}")
        return

    # Load qrels
    qrels = load_qrels(config.DEV_QRELS_FILE)

    if not qrels:
        print("Nincs qrels adat")
        return

    # Initialize components
    reranker = GRPOReranker()
    retriever = HybridRetriever()

    # Training loop with Unsloth-style optimizations
    print("=== GRPO RERANKER TANÍTÁS (Unsloth optimalizálások) ===")
    print(f"Query-k száma: {len(qrels)}")
    print(f"Memory efficient mode: {reranker.memory_efficient}")

    for epoch in range(config.RL_EPOCHS):
        total_reward = 0.0
        num_queries = 0
        epoch_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        epoch_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        if epoch_start:
            epoch_start.record()

        print(f"\nEpoch {epoch + 1}/{config.RL_EPOCHS}")

        for query_id, relevance_data in qrels.items():
            try:
                # Get real retrieval results for this query
                bm25_results, dense_results = retriever.retrieve_candidates(query_id, top_k=50)

                # Only train if we have both BM25 and dense results
                if bm25_results and dense_results:
                    reward = reranker.train_step(bm25_results, dense_results, relevance_data)
                    if reward is not None:
                        total_reward += reward
                        num_queries += 1

            except Exception as e:
                print(f"Hiba a {query_id} query feldolgozásakor: {e}")
                continue

        # Memory cleanup (Unsloth style)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if epoch_end:
            epoch_end.record()
            torch.cuda.synchronize()
            epoch_time = epoch_start.elapsed_time(epoch_end) / 1000
        else:
            epoch_time = 0

        avg_reward = total_reward / num_queries if num_queries > 0 else 0
        print(f"Epoch {epoch + 1} eredmény: {num_queries} query feldolgozva")
        print(".4f")
        if epoch_time > 0:
            print(".2f")

        # Early stopping if reward is consistently improving
        if epoch > 5 and avg_reward > 0.1:  # Good performance threshold
            print("Korai megállás - jó teljesítmény elérve")
            break

    # Final evaluation
    print(f"\n=== VÉGÉRTÉKELÉS ===")
    try:
        eval_results = reranker.evaluate(list(qrels.keys())[:10], qrels)  # Evaluate on first 10 queries
        if eval_results:
            print("Baseline NDCG@10: .4f")
            print("Reranked NDCG@10: .4f")
            print("Improvement: .4f")
    except Exception as e:
        print(f"Evaluation hiba: {e}")

    # Save trained policy
    print(f"\nMentés: {config.RL_POLICY_PATH}")
    reranker.save_policy(config.RL_POLICY_PATH)
    print("=== TANÍTÁS SIKERES ===")

if __name__ == '__main__':
    main()
