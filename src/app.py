"""
Flask frontend for the Semantic Search + Reinforcement Learning project.
"""

import sys
import os
import numpy as np
from flask import Flask, request, render_template, jsonify

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.embedding import load_embedding_model
from src.search.semantic_search import SemanticSearch
from src.rl.agent import RLAgent
from configs import config

# --- Initialization ---
app = Flask(__name__)

# Load models and data (consider loading lazily or using a dedicated setup)
print("Loading components for Flask app...")
model = load_embedding_model(config.EMBEDDING_MODEL_NAME)
# Load documents - replace with actual loading
documents = [
    "A munkáltató rendkívüli felmondással megszüntette a munkaviszonyt.",
    "A munkavállaló bírósághoz fordult az elbocsátása miatt.",
    "A bíróság elutasította a keresetet, mert az eljárás jogszerű volt.",
    "A munkáltató nem tartotta be a felmondási időt.",
    "A per tárgya a végkielégítés mértéke volt.",
    "A kollektív szerződés szabályozza a felmondási védelmet.",
    "A jogellenes munkaviszony megszüntetés következményei.",
    "A próbaidő alatti azonnali hatályú felmondás feltételei.",
    "A munkavállaló kártérítési igénye jogellenes elbocsátás esetén.",
    "A munkáltató bizonyítási kötelezettsége felmondáskor."
]
if not documents:
    print("Error: No documents loaded for Flask app.")
    # Handle error appropriately

search_engine = SemanticSearch(model, documents)
rl_agent = RLAgent(input_dim=config.POLICY_NETWORK_PARAMS['input_dim'],
                   output_dim=config.INITIAL_TOP_K,
                   hidden_dim=config.POLICY_NETWORK_PARAMS['hidden_dim'])
rl_agent.load()
print("Components loaded.")

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Render the main search page."""
    # Assumes an 'index.html' template exists in a 'templates' folder
    return render_template('index.html', initial_query="Jogellenes felmondás munkahelyen")

@app.route('/search', methods=['POST'])
def search():
    """Handle the search request and return ranked results."""
    query = request.form.get('query')
    final_k = int(request.form.get('top_k', config.FINAL_TOP_K))

    if not query:
        return jsonify({"error": "Missing query parameter"}), 400

    try:
        # 1. Initial Candidate Retrieval
        initial_candidates = search_engine.search_candidates(query, config.INITIAL_TOP_K)

        if not initial_candidates:
            return jsonify({"results": [], "message": "No semantic candidates found."})

        # Pad if necessary (ensure robust handling)
        padded_candidates = initial_candidates[:]
        if len(padded_candidates) < config.INITIAL_TOP_K:
            num_missing = config.INITIAL_TOP_K - len(padded_candidates)
            dummy_doc = (-1, "", 0.0)
            padded_candidates.extend([dummy_doc] * num_missing)

        # 2. Prepare State for RL Agent
        query_embedding = search_engine.model.encode([query])[0].astype(np.float32)
        candidate_embeddings = np.zeros((config.INITIAL_TOP_K, query_embedding.shape[0]), dtype=np.float32)
        valid_indices = [i for i, doc in enumerate(padded_candidates) if doc[0] != -1]
        texts_to_encode = [padded_candidates[i][1] for i in valid_indices]

        if texts_to_encode:
            embeddings = search_engine.model.encode(texts_to_encode).astype(np.float32)
            if len(valid_indices) == embeddings.shape[0]:
                candidate_embeddings[valid_indices, :] = embeddings
            else:
                 raise ValueError("Embedding count mismatch during state preparation.")

        state_parts = [query_embedding] + list(candidate_embeddings)
        state = np.concatenate(state_parts).astype(np.float32)

        # Ensure state shape matches agent's expected input dim
        expected_len = config.POLICY_NETWORK_PARAMS['input_dim']
        if state.shape[0] != expected_len:
            if state.shape[0] < expected_len:
                padded_state = np.zeros(expected_len, dtype=np.float32)
                padded_state[:state.shape[0]] = state
                state = padded_state
            else:
                state = state[:expected_len]

        # 3. RL Agent Re-ranking
        action_scores = rl_agent.select_action(state)

        # Combine candidates with scores and sort
        scored_candidates = list(zip(padded_candidates, action_scores))
        valid_scored_candidates = [sc for sc in scored_candidates if sc[0][0] != -1]
        valid_scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Prepare results for JSON response
        results_json = []
        for i, (candidate_data, rl_score) in enumerate(valid_scored_candidates[:final_k]):
            original_idx, doc_text, initial_score = candidate_data
            results_json.append({
                "rank": i + 1,
                "id": original_idx,
                "text_preview": doc_text[:200] + "...", # Send preview
                "rl_score": float(rl_score),
                "initial_semantic_score": float(initial_score)
                # Add full text or link if needed
            })

        return jsonify({"results": results_json})

    except Exception as e:
        print(f"Error during search: {e}") # Log the error server-side
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

# --- Run Application ---
if __name__ == '__main__':
    # Use debug=True only for development
    app.run(debug=True, host='0.0.0.0', port=5001)