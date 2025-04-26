"""
Module for the reinforcement learning environment focused on ranking search results.
"""

import gym
from gym import spaces
import numpy as np
from typing import List, Dict, Tuple, Any
from ..models.embedding import EmbeddingModel
from ..search.semantic_search import SemanticSearch

class RankingEnv(gym.Env):
    """
    RL environment for learning to re-rank legal document search results.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, model: EmbeddingModel, documents: List[str], search_engine: SemanticSearch, initial_top_k: int):
        """
        Initialize environment.

        Args:
            model: Embedding model instance.
            documents: List of all document strings in the corpus.
            search_engine: SemanticSearch instance for candidate retrieval.
            initial_top_k: The number of initial candidates to retrieve and re-rank.
        """
        super().__init__()
        self.model = model
        self.documents = documents
        self.search_engine = search_engine
        self.initial_top_k = initial_top_k

        # --- State Space ---
        # Represents the query and the initial list of candidate documents.
        # Example: Concatenation of query embedding + K document embeddings.
        # Needs careful design based on what the policy network expects.
        embedding_dim = self.model.encode(["test"])[0].shape[0]
        state_dim = embedding_dim * (1 + self.initial_top_k) # Query + K docs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, # Embeddings can have negative values
            shape=(state_dim,),
            dtype=np.float32
        )

        # --- Action Space ---
        # Represents the re-ranked order of the initial_top_k documents.
        # Option 1: Output scores for each doc (requires sorting post-action). Size K.
        # Option 2: Output a permutation directly (complex). Size K! or requires special handling.
        # Option 3: Pairwise preferences (complex). Size K*(K-1)/2.
        # Let's use Option 1: Output scores for simplicity in definition.
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, # Scores can be anything
            shape=(self.initial_top_k,),
            dtype=np.float32
        )

        # Internal state
        self.current_query: str = ""
        self.current_query_embedding: np.ndarray | None = None
        self.candidate_docs: List[Tuple[int, str, float]] = [] # (original_idx, text, initial_score)
        self.candidate_embeddings: np.ndarray | None = None

    def reset(self, query: str | None = None) -> np.ndarray:
        """
        Reset the environment with a new query and initial candidate set.

        Args:
            query: The search query string. If None, a default/random query might be used.

        Returns:
            Initial observation (state).
        """
        if query is None:
            # Replace with a more robust way to get queries for training/evaluation
            query = "Jogellenes elbocsátás"
        self.current_query = query
        self.current_query_embedding = self.model.encode([query])[0].astype(np.float32)

        # Get initial candidates from semantic search
        self.candidate_docs = self.search_engine.search_candidates(query, self.initial_top_k)

        # Ensure we have K candidates, pad if necessary (important for fixed-size state/action)
        if len(self.candidate_docs) < self.initial_top_k:
            # Handle padding: Add dummy docs/embeddings or adjust state/action space dynamically (more complex)
            # For now, let's assume we always get K or handle errors if not.
             print(f"Warning: Got {len(self.candidate_docs)} candidates, expected {self.initial_top_k}. Padding/error handling needed.")
             # Simple padding example (not robust):
             num_missing = self.initial_top_k - len(self.candidate_docs)
             dummy_doc = (-1, "", 0.0)
             self.candidate_docs.extend([dummy_doc] * num_missing)


        # Get embeddings for candidate docs (handle potential missing docs if padding)
        candidate_texts = [doc[1] for doc in self.candidate_docs if doc[0] != -1]
        if candidate_texts:
             embeddings = self.model.encode(candidate_texts).astype(np.float32)
             # Need to map embeddings back to the padded list structure
             self.candidate_embeddings = np.zeros((self.initial_top_k, self.current_query_embedding.shape[0]), dtype=np.float32)
             valid_indices = [i for i, doc in enumerate(self.candidate_docs) if doc[0] != -1]
             if len(valid_indices) == embeddings.shape[0]:
                 self.candidate_embeddings[valid_indices, :] = embeddings
             else:
                 print("Error: Embedding count mismatch after encoding candidates.")
                 # Fallback to zeros or raise error
        else:
             self.candidate_embeddings = np.zeros((self.initial_top_k, self.current_query_embedding.shape[0]), dtype=np.float32)


        # Construct the state
        state = self._get_state()
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step by applying the re-ranking action.
        The reward is determined externally based on expert evaluation of the ranking produced by 'action'.

        Args:
            action: The action from the agent (e.g., scores for each candidate doc).

        Returns:
            observation: The current state (doesn't change within an episode for ranking).
            reward: The reward for the ranking produced by the action (needs external calculation).
            done: True, as ranking is typically a one-step process per query.
            info: Dictionary containing the ranked list based on the action.
        """
        # 'action' contains scores for each of the K candidate documents.
        # Higher scores should mean higher rank.
        if len(action) != self.initial_top_k:
            raise ValueError(f"Action length {len(action)} does not match initial_top_k {self.initial_top_k}")

        # Create the ranked list based on scores
        scored_candidates = list(zip(self.candidate_docs, action))
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # The final ranked list (containing original_idx, text, initial_score)
        ranked_list = [item[0] for item in scored_candidates]

        # The reward is NOT calculated here. It must be provided externally
        # after expert evaluation of 'ranked_list' for the 'current_query'.
        reward = 0.0 # Placeholder - must be calculated externally

        done = True # Ranking is a single step per query
        info = {
            "query": self.current_query,
            "initial_candidates": self.candidate_docs,
            "ranked_list": ranked_list, # List of (original_idx, text, initial_score) tuples in the new order
            "action_scores": action
        }

        # State doesn't change after the action in this setup
        state = self._get_state()

        return state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Construct the state vector from query and candidate embeddings."""
        if self.current_query_embedding is None or self.candidate_embeddings is None:
             # Return a zero vector or handle error appropriately
             embedding_dim = self.observation_space.shape[0] // (1 + self.initial_top_k)
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Concatenate query embedding and all candidate embeddings
        state_parts = [self.current_query_embedding] + list(self.candidate_embeddings)
        state = np.concatenate(state_parts).astype(np.float32)

        # Ensure state shape matches observation space
        if state.shape[0] != self.observation_space.shape[0]:
             # Handle potential shape mismatch due to padding/errors
             print(f"Warning: State shape mismatch. Expected {self.observation_space.shape[0]}, got {state.shape[0]}.")
             # Attempt to fix or raise error - e.g., pad/truncate state
             expected_len = self.observation_space.shape[0]
             if state.shape[0] < expected_len:
                 padded_state = np.zeros(expected_len, dtype=np.float32)
                 padded_state[:state.shape[0]] = state
                 state = padded_state
             else:
                 state = state[:expected_len]

        return state

    def render(self, mode='human'):
        """Render the environment state (optional)."""
        if mode == 'human':
            print(f"Current Query: {self.current_query}")
            print("Candidates:")
            for i, (idx, doc, score) in enumerate(self.candidate_docs):
                print(f"  {i+1}. (ID: {idx}, Score: {score:.4f}) {doc[:100]}...")
        else:
            super().render(mode=mode)

    def close(self):
        """Clean up environment resources."""
        pass