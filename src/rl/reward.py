"""
Module for handling expert evaluations and calculating rewards for the RL agent.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from ...configs import config
import os

def calculate_ndcg(relevance_scores: List[float], k: int | None = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).

    Args:
        relevance_scores: List of relevance scores for the ranked items (higher is better).
        k: Position cutoff. If None, calculate for the full list.

    Returns:
        NDCG score.
    """
    k = k if k is not None else len(relevance_scores)
    if k == 0:
        return 0.0
    k = min(k, len(relevance_scores))
    scores = np.array(relevance_scores[:k])
    ideal_scores = np.sort(scores)[::-1] # Descending sort for ideal ranking

    def dcg(scores_arr):
        gains = 2**scores_arr - 1
        discounts = np.log2(np.arange(len(scores_arr)) + 2)
        return np.sum(gains / discounts)

    dcg_val = dcg(scores)
    idcg_val = dcg(ideal_scores)

    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def load_expert_evaluations(filepath: str = config.EXPERT_EVAL_PATH) -> pd.DataFrame:
    """Load expert evaluations from a CSV file."""
    if not os.path.exists(filepath):
        # Create an empty DataFrame with expected columns if file doesn't exist
        print(f"Evaluation file not found at {filepath}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['query', 'doc_original_idx', 'relevance_score'])
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading evaluations from {filepath}: {e}")
        return pd.DataFrame(columns=['query', 'doc_original_idx', 'relevance_score'])

def save_expert_evaluations(eval_df: pd.DataFrame, filepath: str = config.EXPERT_EVAL_PATH):
    """Save expert evaluations to a CSV file."""
    try:
        eval_df.to_csv(filepath, index=False)
        print(f"Evaluations saved to {filepath}")
    except Exception as e:
        print(f"Error saving evaluations to {filepath}: {e}")


def get_relevance_scores_for_ranking(query: str, ranked_list: List[Tuple[int, str, float]],
                                     eval_df: pd.DataFrame) -> List[float]:
    """
    Retrieve expert relevance scores for a given query and ranked list.

    Args:
        query: The query string.
        ranked_list: The list of ranked documents [(original_idx, text, initial_score), ...].
        eval_df: DataFrame containing expert evaluations ('query', 'doc_original_idx', 'relevance_score').

    Returns:
        List of relevance scores corresponding to the order in ranked_list.
        Returns zeros if evaluations are missing.
    """
    query_evals = eval_df[eval_df['query'] == query]
    if query_evals.empty:
        print(f"Warning: No evaluations found for query: '{query}'")
        return [0.0] * len(ranked_list)

    # Create a mapping from doc_original_idx to relevance_score
    score_map = pd.Series(query_evals.relevance_score.values, index=query_evals.doc_original_idx).to_dict()

    # Get scores for the ranked list, defaulting to 0 if not found
    relevance_scores = [float(score_map.get(doc[0], 0.0)) for doc in ranked_list]
    return relevance_scores


def compute_reward_from_evaluations(query: str, ranked_list: List[Tuple[int, str, float]],
                                    eval_df: pd.DataFrame, k: int = config.FINAL_TOP_K,
                                    baseline_ranking: List[Tuple[int, str, float]] | None = None) -> float:
    """
    Compute the reward for the RL agent based on the NDCG of the ranking.
    Can be absolute NDCG or improvement over a baseline.

    Args:
        query: The query string.
        ranked_list: The ranked list produced by the agent.
        eval_df: DataFrame with expert evaluations.
        k: NDCG cutoff.
        baseline_ranking: Optional baseline ranking (e.g., initial semantic ranking) to compare against.

    Returns:
        Reward value.
    """
    relevance_scores = get_relevance_scores_for_ranking(query, ranked_list, eval_df)
    ndcg_score = calculate_ndcg(relevance_scores, k)

    if baseline_ranking:
        baseline_relevance = get_relevance_scores_for_ranking(query, baseline_ranking, eval_df)
        baseline_ndcg = calculate_ndcg(baseline_relevance, k)
        reward = ndcg_score - baseline_ndcg # Reward is the improvement in NDCG
    else:
        reward = ndcg_score # Reward is the absolute NDCG

    return reward

