"""
Main module for running the Semantic Search + RL Ranking system.
Use command-line arguments to specify mode (search, train, etc.).
"""

import os
import sys
import argparse

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.embedding import load_embedding_model
from src.data_loader.legal_docs import load_documents_from_folder
from src.search.semantic_search import SemanticSearch
# Graph components are used in scripts/populate_graph.py or potentially for context enrichment
# from src.graph.graph_db import GraphDBConnector
# from src.graph.extractor import GraphExtractor
from src.rl.agent import RLAgent
from src.rl.environment import RankingEnv
from src.rl.reward import load_expert_evaluations, compute_reward_from_evaluations
from configs import config
import pandas as pd # Import pandas for handling evaluations
import numpy as np
from tqdm import tqdm # For progress bar

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Legal Semantic Search with RL Ranking")
    parser.add_argument("mode", choices=['search', 'evaluate'], # Add 'train' mode handled by train_rl_agent.py
                        help="Operation mode: 'search' for interactive search, 'evaluate' for batch evaluation.")
    parser.add_argument("--query", type=str, default="Jogellenes felmondás munkahelyen",
                        help="Search query (for 'search' mode)")
    parser.add_argument("--doc_path", type=str, default=config.RAW_DATA_PATH,
                        help="Path to folder containing legal documents")
    parser.add_argument("--model", type=str, default=config.EMBEDDING_MODEL_NAME,
                        help="Embedding model name")
    parser.add_argument("--top_k", type=int, default=config.FINAL_TOP_K,
                        help="Number of final documents to display")
    # Add other relevant arguments if needed
    return parser.parse_args()

def run_search(args, documents, search_engine, rl_agent):
    """Handles the interactive search mode."""
    query = args.query
    print(f"Query: {query}")

    # 1. Initial Candidate Retrieval
    initial_candidates = search_engine.search_candidates(query, config.INITIAL_TOP_K)
    if not initial_candidates:
        print("No candidates found.")
        return

    print(f"\nInitial {len(initial_candidates)} candidates (Semantic Search):")
    for i, (idx, doc, score) in enumerate(initial_candidates):
        print(f"  {i+1}. (ID: {idx}, Score: {score:.4f}) {doc[:100]}...")

    # 2. Prepare State for RL Agent
    # This requires constructing the state vector as defined in RankingEnv
    query_embedding = search_engine.model.encode([query])[0].astype(np.float32)
    candidate_embeddings = np.zeros((config.INITIAL_TOP_K, query_embedding.shape[0]), dtype=np.float32)
    valid_indices = [i for i, doc in enumerate(initial_candidates) if doc[0] != -1] # Assuming -1 is invalid index
    if valid_indices:
        texts_to_encode = [initial_candidates[i][1] for i in valid_indices]
        if texts_to_encode:
            embeddings = search_engine.model.encode(texts_to_encode).astype(np.float32)
            if len(valid_indices) == embeddings.shape[0]:
                 candidate_embeddings[valid_indices, :] = embeddings
            else:
                 print("Warning: Embedding count mismatch during state preparation.")

    state_parts = [query_embedding] + list(candidate_embeddings)
    state = np.concatenate(state_parts).astype(np.float32)
    # Ensure state shape matches agent's expected input dim
    expected_len = config.POLICY_NETWORK_PARAMS['input_dim']
    if state.shape[0] != expected_len:
        print(f"Warning: State shape mismatch in search. Expected {expected_len}, got {state.shape[0]}. Adjusting...")
        if state.shape[0] < expected_len:
            padded_state = np.zeros(expected_len, dtype=np.float32)
            padded_state[:state.shape[0]] = state
            state = padded_state
        else:
            state = state[:expected_len]


    # 3. RL Agent Re-ranking
    action_scores = rl_agent.select_action(state)

    # Combine candidates with scores and sort
    scored_candidates = list(zip(initial_candidates, action_scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    final_ranked_list = [item[0] for item in scored_candidates] # List of (original_idx, text, initial_score)

    # 4. Display Final Results
    print(f"\nFinal Top {args.top_k} Results (RL Re-ranked):")
    for i, (idx, doc, initial_score) in enumerate(final_ranked_list[:args.top_k]):
         rl_score = scored_candidates[i][1] # Get the score assigned by RL agent
         print(f"  {i+1}. (ID: {idx}, RL Score: {rl_score:.4f}, Initial Score: {initial_score:.4f}) {doc[:150]}...")

    # 5. (Optional) Graph Context Enrichment
    # Here you could query the graph database for context about the top results
    # e.g., graph_connector.get_neighbors(NODE_JUDGMENT, top_result_id)

def get_relevance_scores_for_ranking(query, ranked_documents, eval_df):
    """
    Get relevance scores for a list of ranked documents based on expert evaluations.
    
    Args:
        query: The search query
        ranked_documents: List of (doc_id, doc_text, score) tuples
        eval_df: DataFrame containing expert evaluations
        
    Returns:
        List of relevance scores corresponding to the ranked documents
    """
    relevance_scores = []
    
    # Filter evaluation data for this query
    query_evals = eval_df[eval_df['query'] == query]
    
    for doc_id, _, _ in ranked_documents:
        # Find the relevance score in evaluation data
        doc_eval = query_evals[query_evals['doc_id'] == doc_id]
        if not doc_eval.empty:
            # Assuming there's a 'relevance' column with scores
            relevance_scores.append(float(doc_eval['relevance'].iloc[0]))
        else:
            # If document not evaluated, assume zero relevance
            relevance_scores.append(0.0)
    
    return relevance_scores

def calculate_ndcg(relevance_scores, k):
    """
    Calculate Normalized Discounted Cumulative Gain.
    
    Args:
        relevance_scores: List of relevance scores for documents
        k: Number of documents to consider (NDCG@k)
        
    Returns:
        NDCG value between 0.0 and 1.0
    """
    # Ensure we only consider top k items
    relevance_scores = relevance_scores[:k]
    
    # If no relevant documents, return 0
    if not relevance_scores or sum(relevance_scores) == 0:
        return 0.0
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = relevance_scores[0] + sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores[1:]))
    
    # Calculate IDCG (Ideal DCG - when documents are sorted by relevance)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = ideal_relevance[0] + sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[1:]))
    
    # Return normalized score
    return dcg / idcg if idcg > 0 else 0.0

def run_evaluation(args, documents, search_engine, rl_agent, eval_df):
    """Handles batch evaluation mode."""
    print("Running evaluation...")
    # Use queries from the evaluation dataset
    queries = eval_df['query'].unique()
    results = []

    for query in queries:
        # --- Perform search and re-ranking as in run_search ---
        initial_candidates = search_engine.search_candidates(query, config.INITIAL_TOP_K)
        if not initial_candidates: continue

        # Prepare state (simplified, reuse logic from run_search or RankingEnv.reset)
        query_embedding = search_engine.model.encode([query])[0].astype(np.float32)
        # ... (construct full state vector including candidate embeddings) ...
        # This state construction needs to be robustly implemented
        state = np.zeros(config.POLICY_NETWORK_PARAMS['input_dim'], dtype=np.float32) # Placeholder state

        action_scores = rl_agent.select_action(state)
        scored_candidates = list(zip(initial_candidates, action_scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        rl_ranked_list = [item[0] for item in scored_candidates]

        # --- Calculate Metrics ---
        # Get relevance scores from expert evaluations
        initial_relevance = get_relevance_scores_for_ranking(query, initial_candidates, eval_df)
        rl_relevance = get_relevance_scores_for_ranking(query, rl_ranked_list, eval_df)

        # Calculate NDCG@k for both rankings
        k = args.top_k
        initial_ndcg = calculate_ndcg(initial_relevance, k)
        rl_ndcg = calculate_ndcg(rl_relevance, k)

        results.append({
            "query": query,
            f"initial_ndcg@{k}": initial_ndcg,
            f"rl_ndcg@{k}": rl_ndcg,
            "improvement": rl_ndcg - initial_ndcg
        })
        print(f"Query: {query[:50]}... | Initial NDCG@{k}: {initial_ndcg:.4f} | RL NDCG@{k}: {rl_ndcg:.4f}")

    # --- Aggregate and Print Results ---
    results_df = pd.DataFrame(results)
    print("\n--- Evaluation Summary ---")
    print(results_df.describe())
    print("\nAverage Improvement:", results_df['improvement'].mean())


def main():
    """Main function."""
    args = parse_args()

    # Load embedding model
    print(f"Loading embedding model: {args.model}")
    model = load_embedding_model(args.model)

    # Load documents
    print(f"Loading documents from: {args.doc_path}")
    documents = load_documents_from_folder(args.doc_path)
    if not documents:
        print(f"Error: No documents found in {args.doc_path}. Exiting.")
        return
    print(f"Loaded {len(documents)} documents.")

    # Initialize semantic search engine
    print("Initializing semantic search engine...")
    search_engine = SemanticSearch(model, documents)
    print("Semantic search engine initialized.")

    # Initialize RL agent
    print("Initializing RL agent...")
    agent_input_dim = config.POLICY_NETWORK_PARAMS['input_dim']
    agent_output_dim = config.INITIAL_TOP_K # Agent outputs scores for K candidates
    rl_agent = RLAgent(input_dim=agent_input_dim,
                       output_dim=agent_output_dim,
                       hidden_dim=config.POLICY_NETWORK_PARAMS['hidden_dim'])
    rl_agent.load() # Load pre-trained weights if available
    print("RL agent initialized.")

    # Initialize Graph Connector (optional, if used for context)
    # print("Initializing Graph DB connector...")
    # graph_connector = GraphDBConnector()

    if args.mode == 'search':
        run_search(args, documents, search_engine, rl_agent)
    elif args.mode == 'evaluate':
        # Load expert evaluations needed for evaluation mode
        print(f"Loading expert evaluations from: {config.EXPERT_EVAL_PATH}")
        eval_df = load_expert_evaluations()
        if eval_df.empty:
             print("Warning: Evaluation data is empty. Cannot run evaluation.")
        else:
             run_evaluation(args, documents, search_engine, rl_agent, eval_df)

    # Close graph connection if opened
    # graph_connector.close()

if __name__ == "__main__":
    main()