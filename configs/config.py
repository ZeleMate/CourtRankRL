"""
Configuration settings for the project.
"""
import os # Import os to potentially read API key from environment

# Model configuration
# Choose appropriate model like 'SZTAKI-HLT/hubert-base-cc', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', or a specific LegalBERT
# Using OpenAI's API
EMBEDDING_MODEL_NAME = "openai/text-embedding-3-large"
# Update dimension based on the chosen model
EMBEDDING_DIMENSION = 3072  # Dimension for text-embedding-3-large

# OpenAI API Key - IMPORTANT: Set this as an environment variable 'OPENAI_API_KEY'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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") # Example, better to just read from env

# Search configuration
INITIAL_TOP_K = 20 # Number of candidates retrieved by semantic search for RL re-ranking
FINAL_TOP_K = 5    # Number of results shown to the user after re-ranking
# SIMILARITY_THRESHOLD = 0.7 # May not be needed if using vector search index directly

# Graph Database configuration
GRAPH_DB_URI = "bolt://localhost:7687" # Example for Neo4j
GRAPH_DB_USER = "neo4j"
GRAPH_DB_PASSWORD = "password" # Use environment variables in production

# RL configuration
RL_ALGORITHM = "GRPO" # or "PolicyGradient", "PPO", etc.
POLICY_NETWORK_PARAMS = {
    "input_dim": EMBEDDING_DIMENSION * (INITIAL_TOP_K + 1), # Updated dimension used here
    "hidden_dim": 256,
    "output_dim": INITIAL_TOP_K # Example: scores for each doc
}
LEARNING_RATE = 0.0001 # Adjusted learning rate for policy networks
DISCOUNT_FACTOR = 0.99
EPOCHS_PER_UPDATE = 5
TRAINING_BATCH_SIZE = 32 # Number of query-ranking pairs per update
MAX_TRAINING_ITERATIONS = 1000 # Total training iterations

# Data paths
RAW_DATA_PATH = "/Users/zelenyianszkimate/Downloads/BHGY-k" # Updated path
PROCESSED_DATA_PATH = "data/processed"
MODEL_SAVE_PATH = "models"
GRAPH_SCHEMA_PATH = "configs/graph_schema.json" # Optional: if schema is defined in a file
EXPERT_EVAL_PATH = "data/expert_evaluations.csv" # Path to store/load expert feedback
RL_AGENT_SAVE_PATH = "models/rl_agent.pt"