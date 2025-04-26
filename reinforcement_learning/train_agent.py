"""
Script to train the RL ranking agent using expert evaluations.
"""
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.embedding import load_embedding_model
from src.data_loader.legal_docs import load_documents_from_folder
from src.search.semantic_search import SemanticSearch
from src.reinforcement_learning.agent import RLAgent
from src.reinforcement_learning.environment import RankingEnv # Needed for state construction logic
from src.reinforcement_learning.reward_models.reward import load_expert_evaluations, compute_reward_from_evaluations, get_relevance_scores_for_ranking, calculate_ndcg
from configs import config
from tqdm import tqdm # Progress bar

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL Ranking Agent")
    parser.add_argument("--eval_path", type=str, default=config.EXPERT_EVAL_PATH,
                        help="Path to expert evaluation data CSV")
    parser.add_argument("--doc_path", type=str, default=config.RAW_DATA_PATH,
                        help="Path to folder containing documents (needed for search engine)")
    parser.add_argument("--model", type=str, default=config.EMBEDDING_MODEL_NAME,
                        help="Embedding model name")
    parser.add_argument("--iterations", type=int, default=config.MAX_TRAINING_ITERATIONS,
                        help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=config.TRAINING_BATCH_SIZE,
                        help="Batch size for agent updates")
    # Add other training-specific arguments (e.g., save frequency)
    return parser.parse_args()

def collect_experience_batch(queries: List[str], search_engine: SemanticSearch,
                             rl_agent: RLAgent, eval_df: pd.DataFrame, batch_size: int) -> List[Dict[str, Any]]:
    """
    Collect a batch of experiences (state, action, reward) using the current policy.
    """
    batch = []
    collected = 0
    attempts = 0
    max_attempts = batch_size * 5 # Limit attempts to avoid infinite loop if queries lack evals

    while collected < batch_size and attempts < max_attempts:
        attempts += 1
        query = random.choice(queries)

        # Check if evaluations exist for this query
        query_evals = eval_df[eval_df['query'] == query]
        if query_evals.empty:
            continue # Skip query if no evaluations available

        # --- Perform search and get state (similar to main.py/run_search) ---
        initial_candidates = search_engine.search_candidates(query, config.INITIAL_TOP_K)
        if not initial_candidates or len(initial_candidates) < config.INITIAL_TOP_K:
             # print(f"Skipping query '{query[:30]}...' due to insufficient candidates.")
             continue # Skip if not enough candidates (or handle padding robustly)

        # Construct state vector
        query_embedding = search_engine.model.encode([query])[0].astype(np.float32)
        candidate_embeddings = np.zeros((config.INITIAL_TOP_K, query_embedding.shape[0]), dtype=np.float32)
        valid_indices = [i for i, doc in enumerate(initial_candidates) if doc[0] != -1]
        texts_to_encode = [initial_candidates[i][1] for i in valid_indices]
        if texts_to_encode:
            embeddings = search_engine.model.encode(texts_to_encode).astype(np.float32)
            if len(valid_indices) == embeddings.shape[0]:
                 candidate_embeddings[valid_indices, :] = embeddings
            else:
                 print(f"Warning: Embedding count mismatch for query '{query[:30]}...'")
                 continue # Skip if state construction fails

        state_parts = [query_embedding] + list(candidate_embeddings)
        state = np.concatenate(state_parts).astype(np.float32)
        # Ensure state shape matches agent's expected input dim
        expected_len = config.POLICY_NETWORK_PARAMS['input_dim']
        if state.shape[0] != expected_len:
            # print(f"Warning: State shape mismatch for query '{query[:30]}...'. Adjusting...")
            if state.shape[0] < expected_len:
                padded_state = np.zeros(expected_len, dtype=np.float32)
                padded_state[:state.shape[0]] = state
                state = padded_state
            else:
                state = state[:expected_len]
        # --- Get action (scores) from agent ---
        action_scores = rl_agent.select_action(state)

        # --- Determine the ranking based on scores ---
        scored_candidates = list(zip(initial_candidates, action_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        ranked_list = [item[0] for item in scored_candidates] # List of (original_idx, text, initial_score)

        # --- Calculate reward using expert evaluations ---
        # Use the initial semantic ranking as baseline for reward calculation
        reward = compute_reward_from_evaluations(query, ranked_list, eval_df,
                                                 k=config.FINAL_TOP_K,
                                                 baseline_ranking=initial_candidates)

        # Store experience
        experience = {
            "state": state,
            "action": action_scores, # Store the scores that led to the ranking
            "reward": reward,
            "query": query, # For debugging/analysis
            "ranked_list_indices": [doc[0] for doc in ranked_list] # Store ranked indices
        }
        batch.append(experience)
        collected += 1

    if collected < batch_size:
        print(f"Warning: Collected only {collected}/{batch_size} experiences.")

    return batch


def main():
    args = parse_args()

    print("Loading expert evaluations...")
    eval_df = load_expert_evaluations(args.eval_path)
    if eval_df.empty:
        print("Error: No evaluation data found. Cannot train.")
        return
    queries_with_evals = eval_df['query'].unique().tolist()
    if not queries_with_evals:
        print("Error: No queries found in evaluation data.")
        return
    print(f"Found evaluations for {len(queries_with_evals)} unique queries.")

    print("Loading embedding model and documents...")
    model = load_embedding_model(args.model)
    documents = load_documents_from_folder(args.doc_path)
    if not documents:
        print("Error: No documents found.")
        return

    print("Initializing search engine...")
    search_engine = SemanticSearch(model, documents)

    print("Initializing RL agent...")
    agent_input_dim = config.POLICY_NETWORK_PARAMS['input_dim']
    agent_output_dim = config.INITIAL_TOP_K
    rl_agent = RLAgent(input_dim=agent_input_dim,
                       output_dim=agent_output_dim,
                       hidden_dim=config.POLICY_NETWORK_PARAMS['hidden_dim'])
    rl_agent.load() # Load existing agent if available

    print(f"Starting RL training for {args.iterations} iterations...")
    for iteration in tqdm(range(args.iterations), desc="Training Iterations"):
        # 1. Collect a batch of experiences using the current policy
        experience_batch = collect_experience_batch(queries_with_evals, search_engine, rl_agent, eval_df, args.batch_size)

        if not experience_batch:
            print(f"Iteration {iteration+1}: No experiences collected, skipping update.")
            continue

        # 2. Update the agent using the collected batch
        # The update logic inside agent.update() depends on the chosen algorithm (GRPO, PG)
        rl_agent.update(experience_batch)

        # Optional: Log metrics (e.g., average reward in batch)
        avg_reward = np.mean([exp['reward'] for exp in experience_batch])
        tqdm.write(f"Iteration {iteration+1}/{args.iterations} - Avg Batch Reward (NDCG Improvement): {avg_reward:.4f}")

        # Optional: Save agent periodically
        if (iteration + 1) % 50 == 0: # Save every 50 iterations
            rl_agent.save()

    # Save final agent
    rl_agent.save()
    print("Training finished. Final RL agent saved.")

if __name__ == "__main__":
    main()
