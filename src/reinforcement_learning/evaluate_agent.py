"""
Agent kiértékelése: Reinforcement Learning alapú rangsoroló teljesítményének mérése.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.embedding import load_embedding_model
from src.data_loader.legal_docs import load_documents_from_folder
from src.search.semantic_search import SemanticSearch
from src.reinforcement_learning.agent import RLAgent
from src.reinforcement_learning.reward_models.reward import load_expert_evaluations, get_relevance_scores_for_ranking, calculate_ndcg
from configs import config

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RL Ranking Agent")
    parser.add_argument("--eval_path", type=str, default=config.EXPERT_EVAL_PATH,
                        help="Path to expert evaluation data CSV")
    parser.add_argument("--doc_path", type=str, default=config.RAW_DATA_PATH,
                        help="Path to folder containing documents")
    parser.add_argument("--model", type=str, default=config.EMBEDDING_MODEL_NAME,
                        help="Embedding model name")
    parser.add_argument("--top_k", type=int, default=config.FINAL_TOP_K,
                        help="NDCG@k értékeléshez a k")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Loading expert evaluations...")
    eval_df = load_expert_evaluations(args.eval_path)
    if eval_df.empty:
        print("Error: No evaluation data found. Cannot evaluate.")
        return
    queries = eval_df['query'].unique()
    print(f"Found {len(queries)} queries for evaluation.")

    print("Loading embedding model and documents...")
    model = load_embedding_model(args.model)
    documents = load_documents_from_folder(args.doc_path)
    if not documents:
        print("Error: No documents found.")
        return
    search_engine = SemanticSearch(model, documents)

    print("Loading RL agent...")
    agent_input_dim = config.POLICY_NETWORK_PARAMS['input_dim']
    agent_output_dim = config.INITIAL_TOP_K
    rl_agent = RLAgent(input_dim=agent_input_dim,
                       output_dim=agent_output_dim,
                       hidden_dim=config.POLICY_NETWORK_PARAMS['hidden_dim'])
    rl_agent.load()

    results = []
    for query in tqdm(queries, desc="Evaluating queries"):
        initial_candidates = search_engine.search_candidates(query, config.INITIAL_TOP_K)
        if not initial_candidates:
            continue
        query_embedding = search_engine.model.encode([query])[0].astype(np.float32)
        candidate_embeddings = np.zeros((config.INITIAL_TOP_K, query_embedding.shape[0]), dtype=np.float32)
        valid_indices = [i for i, doc in enumerate(initial_candidates) if doc[0] != -1]
        texts_to_encode = [initial_candidates[i][1] for i in valid_indices]
        if texts_to_encode:
            embeddings = search_engine.model.encode(texts_to_encode).astype(np.float32)
            if len(valid_indices) == embeddings.shape[0]:
                candidate_embeddings[valid_indices, :] = embeddings
            else:
                continue
        state_parts = [query_embedding] + list(candidate_embeddings)
        state = np.concatenate(state_parts).astype(np.float32)
        expected_len = config.POLICY_NETWORK_PARAMS['input_dim']
        if state.shape[0] != expected_len:
            if state.shape[0] < expected_len:
                padded_state = np.zeros(expected_len, dtype=np.float32)
                padded_state[:state.shape[0]] = state
                state = padded_state
            else:
                state = state[:expected_len]
        action_scores = rl_agent.select_action(state)
        scored_candidates = list(zip(initial_candidates, action_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        rl_ranked_list = [item[0] for item in scored_candidates]
        initial_relevance = get_relevance_scores_for_ranking(query, initial_candidates, eval_df)
        rl_relevance = get_relevance_scores_for_ranking(query, rl_ranked_list, eval_df)
        k = args.top_k
        initial_ndcg = calculate_ndcg(initial_relevance, k)
        rl_ndcg = calculate_ndcg(rl_relevance, k)
        results.append({
            "query": query,
            f"initial_ndcg@{k}": initial_ndcg,
            f"rl_ndcg@{k}": rl_ndcg,
            "improvement": rl_ndcg - initial_ndcg
        })
    if not results:
        print("No queries were successfully evaluated.")
        return
    results_df = pd.DataFrame(results)
    print("\n--- Evaluation Summary ---")
    print(results_df.describe())
    print("\nAverage Improvement:", results_df['improvement'].mean())

if __name__ == "__main__":
    main()
