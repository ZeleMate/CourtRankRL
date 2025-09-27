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

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Projekt gyökér hozzáadása az import úthoz
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs import config
from src.search.hybrid_search import HybridRetriever

class RankingPolicy(nn.Module):
    """Shallow MLP policy (agents.md specifikáció)."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = config.RL_HIDDEN_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        return self.network(x)

class GRPOReranker:
    """GRPO-style reranker with Unsloth-inspired memory optimizations."""

    def __init__(self):
        self.policy = RankingPolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.RL_LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        self.baseline_history: List[float] = []
        self.reward_std_history: List[float] = []
        self.memory_efficient = True

    def extract_features(
        self,
        doc_ids: List[str],
        bm25_scores: Dict[str, float],
        dense_scores: Dict[str, float],
        chunk_groups: Dict[str, List[Tuple[str, float]]],
        retriever: HybridRetriever,
    ) -> np.ndarray:
        features: List[np.ndarray] = []

        bm25_vals = list(bm25_scores.values())
        bm25_mean = np.mean(bm25_vals) if bm25_vals else 0.0
        bm25_std = np.std(bm25_vals) if bm25_vals else 1.0
        if bm25_std == 0:
            bm25_std = 1.0

        for doc_id in doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            dense_score = dense_scores.get(doc_id, 0.0)
            normalized_bm25 = (bm25_score - bm25_mean) / bm25_std

            # Rang különbség a top listákból
            bm25_rank = self._get_rank(doc_id, bm25_scores)
            dense_rank = self._get_rank(doc_id, dense_scores)
            rank_diff = abs(bm25_rank - dense_rank)

            # Token hossz feature – átlag chunk hossz
            chunk_list = chunk_groups.get(doc_id, [])
            if chunk_list and retriever.bm25:
                lengths = []
                for chunk_id, _ in chunk_list:
                    try:
                        parts = chunk_id.split("_")
                        lengths.append(int(parts[-1]))
                    except Exception:
                        continue
                token_len = np.mean(lengths) if lengths else retriever.bm25.avg_doc_length
            else:
                token_len = retriever.bm25.avg_doc_length if retriever.bm25 else 0.0

            features.append(
                np.array([dense_score, normalized_bm25, rank_diff, token_len], dtype=np.float32)
            )

        return np.stack(features) if features else np.zeros((0, 4), dtype=np.float32)

    @staticmethod
    def _get_rank(doc_id: str, scores: Dict[str, float]) -> int:
        if not scores:
            return 0
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        for idx, (candidate, _) in enumerate(ranked, start=1):
            if candidate == doc_id:
                return idx
        return len(ranked) + 1

    def calculate_ndcg(
        self,
        predicted_ranks: Sequence[int],
        true_relevance: Sequence[float],
        k: int = 10,
    ) -> float:
        """Calculate NDCG@k."""
        if not true_relevance:
            return 0.0

        dcg = 0.0
        for i in range(min(k, len(predicted_ranks))):
            if i < len(true_relevance):
                rel = true_relevance[predicted_ranks[i]]
                dcg += rel / np.log2(i + 2)

        sorted_rel = sorted(true_relevance, reverse=True)
        idcg = 0.0
        for i in range(min(k, len(sorted_rel))):
            idcg += sorted_rel[i] / np.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def train_step(
        self,
        retriever: HybridRetriever,
        query_id: str,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        qrel_data: Dict[str, int],
    ):
        """
        Single GRPO training step with Unsloth-inspired memory optimizations.
        Uses group-relative rewards and efficient memory usage.
        """
        if not bm25_results and not dense_results:
            return None

        bm25_scores = dict(bm25_results)
        dense_scores = dict(dense_results)

        all_doc_ids_sorted = sorted(
            set(bm25_scores.keys()) | set(dense_scores.keys()),
            key=lambda doc: bm25_scores.get(doc, 0.0) + dense_scores.get(doc, 0.0),
            reverse=True,
        )

        features = self.extract_features(
            all_doc_ids_sorted,
            bm25_scores,
            dense_scores,
            getattr(retriever, "bm25_chunk_groups", {}),
            retriever,
        )

        if features.size == 0:
            return None

        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            policy_scores = self.policy(features_tensor).squeeze()
            softmax_scores = torch.softmax(policy_scores, dim=0)

        ranked_indices = torch.argsort(softmax_scores, descending=True)

        all_doc_ids = all_doc_ids_sorted
        true_relevance = [qrel_data.get(doc_id, 0) for doc_id in all_doc_ids]

        baseline_ndcg = self.calculate_ndcg(
            list(range(len(all_doc_ids))), true_relevance, k=10
        )
        policy_ndcg = self.calculate_ndcg(ranked_indices.tolist(), true_relevance, k=10)

        reward = policy_ndcg - baseline_ndcg

        self.baseline_history.append(baseline_ndcg)
        if len(self.baseline_history) > 100:
            self.baseline_history.pop(0)

        self.reward_std_history.append(reward)
        if len(self.reward_std_history) > 50:
            self.reward_std_history.pop(0)

        advantage = reward
        if self.memory_efficient and len(self.baseline_history) > 5:
            group_baseline = np.mean(self.baseline_history[-10:]) if len(self.baseline_history) >= 10 else np.mean(self.baseline_history)
            advantage = reward - group_baseline
            if len(self.reward_std_history) > 5:
                global_std = np.std(self.reward_std_history) + 1e-8
                advantage /= global_std

        top_indices = ranked_indices[: min(10, len(ranked_indices))]
        selected_scores = softmax_scores[top_indices]

        advantage_tensor = torch.tensor(float(advantage), dtype=torch.float32, device=self.device)
        loss = -advantage_tensor * torch.mean(selected_scores)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return reward

    def rerank(
        self,
        retriever: HybridRetriever,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        if not bm25_results and not dense_results:
            return []

        bm25_scores = dict(bm25_results)
        dense_scores = dict(dense_results)
        all_doc_ids = sorted(
            set(bm25_scores.keys()) | set(dense_scores.keys()),
            key=lambda doc: bm25_scores.get(doc, 0.0) + dense_scores.get(doc, 0.0),
            reverse=True,
        )

        features = self.extract_features(
            all_doc_ids,
            bm25_scores,
            dense_scores,
            getattr(retriever, "bm25_chunk_groups", {}),
            retriever,
        )
        if features.size == 0:
            return bm25_results

        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            policy_scores = self.policy(features_tensor).squeeze()
            softmax_scores = torch.softmax(policy_scores, dim=0)

        ranked_indices = torch.argsort(softmax_scores, descending=True)
        reranked = []
        for idx in ranked_indices:
            doc_id = all_doc_ids[int(idx)]
            reranked.append((doc_id, float(softmax_scores[idx].item())))
        return reranked

    def evaluate(self, retriever: HybridRetriever, test_queries: List[str], qrels: Dict[str, Dict[str, int]]):
        """
        Evaluate reranker performance vs baseline.
        Agents.md spec: numeric comparison baseline vs rerank.
        """
        total_baseline_ndcg = 0.0
        total_reranked_ndcg = 0.0
        num_queries = 0

        print("=== EVALUATION: BASELINE vs RERANKED ===")

        for query in test_queries:
            if query not in qrels:
                continue

            retriever.retrieve_candidates(query, top_k=50)
            bm25_results = retriever.get_last_doc_scores("bm25")
            dense_results = retriever.get_last_doc_scores("dense")

            if not bm25_results:
                continue

            # Baseline: BM25 results
            baseline_doc_ids = [doc_id for doc_id, _ in bm25_results]
            true_relevance = [qrels[query].get(doc_id, 0) for doc_id in baseline_doc_ids]
            baseline_ndcg = self.calculate_ndcg(list(range(len(baseline_doc_ids))), true_relevance)

            reranked_results = self.rerank(retriever, bm25_results, dense_results)
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
                retriever.retrieve_candidates(query_id, top_k=50)
                bm25_results = retriever.get_last_doc_scores("bm25")
                dense_results = retriever.get_last_doc_scores("dense")

                if bm25_results:
                    reward = reranker.train_step(
                        retriever,
                        query_id,
                        bm25_results,
                        dense_results,
                        relevance_data,
                    )
                    if reward is not None:
                        total_reward += reward
                        num_queries += 1

            except Exception as e:
                print(f"Hiba a {query_id} query feldolgozásakor: {e}")
                continue

        # Memory cleanup (Unsloth style)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if epoch_end and epoch_start:
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
        eval_results = reranker.evaluate(retriever, list(qrels.keys())[:10], qrels)
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
