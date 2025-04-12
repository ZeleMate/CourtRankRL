"""
Configuration settings for the project.
"""

# Model configuration
# Choose appropriate model like 'SZTAKI-HLT/hubert-base-cc', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', or a specific LegalBERT
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Update dimension based on the chosen model
EMBEDDING_DIMENSION = 384  # Example for paraphrase-multilingual-MiniLM-L12-v2

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
    "input_dim": EMBEDDING_DIMENSION * (INITIAL_TOP_K + 1), # Example: query emb + K doc embs
    "hidden_dim": 256,
    "output_dim": INITIAL_TOP_K # Example: scores for each doc
}
LEARNING_RATE = 0.0001 # Adjusted learning rate for policy networks
DISCOUNT_FACTOR = 0.99
EPOCHS_PER_UPDATE = 5
TRAINING_BATCH_SIZE = 32 # Number of query-ranking pairs per update
MAX_TRAINING_ITERATIONS = 1000 # Total training iterations

# Data paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
MODEL_SAVE_PATH = "models"
GRAPH_SCHEMA_PATH = "configs/graph_schema.json" # Optional: if schema is defined in a file
EXPERT_EVAL_PATH = "data/expert_evaluations.csv" # Path to store/load expert feedback
RL_AGENT_SAVE_PATH = "models/rl_agent.pt"