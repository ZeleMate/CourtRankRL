"""
Script to train the RL ranking agent using expert evaluations.
This script now uses the self-contained RankingEnv and loads all data from Azure.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import io
import logging

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.reinforcement_learning.agent import RLAgent
from src.reinforcement_learning.environment import RankingEnv
from src.reinforcement_learning.reward_models.reward import compute_ndcg
from configs import config
from tqdm import tqdm

# --- Beállítások ---
TRAINING_ITERATIONS = 1000
BATCH_SIZE = 32
INITIAL_TOP_K = 20 # Ennek meg kell egyeznie a RankingEnv-ben használt értékkel

# Loggolás beállítása
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_expert_evaluations() -> pd.DataFrame:
    """Loads expert evaluation data from a local CSV file."""
    eval_path = config.EXPERT_EVALUATIONS_CSV
    logging.info(f"Szakértői értékelések betöltése innen: {eval_path}")
    try:
        df = pd.read_csv(eval_path)
        logging.info(f"Sikeresen betöltve {len(df)} értékelés.")
        return df
    except FileNotFoundError:
        logging.error(f"Hiba: Az értékelő fájl nem található itt: {eval_path}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Hiba az értékelő adatok betöltésekor: {e}", exc_info=True)
        return pd.DataFrame()

def get_relevance_scores(ranked_doc_ids: List[str], eval_df: pd.DataFrame, query: str) -> List[float]:
    """Retrieves relevance scores for a ranked list of document IDs."""
    query_evals = eval_df[eval_df['query'] == query].set_index('doc_id')
    return [query_evals.loc[doc_id, 'relevance'] if doc_id in query_evals.index else 0.0 for doc_id in ranked_doc_ids]

def main():
    """Main training loop for the RL agent."""
    logging.info("RL ÜGYNÖK TANÍTÁSÁNAK INDÍTÁSA")

    # 1. Erőforrások inicializálása
    try:
        env = RankingEnv(initial_top_k=INITIAL_TOP_K)
        
        agent_input_dim = env.observation_space.shape[0]
        agent_output_dim = env.action_space.shape[0]
        agent = RLAgent(input_dim=agent_input_dim, output_dim=agent_output_dim)
        # Próbáljuk betölteni a korábban mentett modellt
        agent.load()

    except Exception as e:
        logging.error(f"Hiba az inicializálás során: {e}", exc_info=True)
        sys.exit(1)

    # 2. Szakértői értékelések betöltése
    eval_df = load_expert_evaluations()
    if eval_df.empty:
        logging.error("Nincsenek szakértői értékelések, a tanítás leáll.")
        sys.exit(1)
    
    queries_with_evals = eval_df['query'].unique().tolist()
    logging.info(f"Tanítás {len(queries_with_evals)} egyedi lekérdezés alapján.")

    # 3. Tanítási ciklus
    for iteration in tqdm(range(TRAINING_ITERATIONS), desc="Training Iterations"):
        experience_batch: List[Dict[str, Any]] = []
        
        # Tapasztalatgyűjtés a batch-hez
        for _ in range(BATCH_SIZE):
            query = np.random.choice(queries_with_evals)
            state, info = env.reset(query=query)
            
            # Akció választása az ágenstől
            action = agent.select_action(state)
            
            # Lépés a környezetben (a jutalom itt még placeholder)
            next_state, _, done, _, step_info = env.step(action)
            
            # Jutalom számítása
            reranked_results = step_info['reranked_results']
            reranked_doc_ids = [res.doc_id for res in reranked_results]
            
            relevance_scores = get_relevance_scores(reranked_doc_ids, eval_df, query)
            reward = compute_ndcg(np.array(relevance_scores))

            experience_batch.append({
                "state": state,
                "action": action,
                "reward": reward
            })

        if not experience_batch:
            logging.warning("Nem sikerült tapasztalatot gyűjteni, a frissítés kimarad.")
            continue
            
        # Ágens frissítése a gyűjtött tapasztalatok alapján
        agent.update(experience_batch)

        if (iteration + 1) % 10 == 0:
            avg_reward = np.mean([exp['reward'] for exp in experience_batch])
            tqdm.write(f"Iteráció {iteration+1}/{TRAINING_ITERATIONS} | Átlagos jutalom (NDCG): {avg_reward:.4f}")

        # Modell mentése időnként
        if (iteration + 1) % 100 == 0:
            agent.save()
            logging.info(f"Ügynök mentve a(z) {iteration+1}. iterációnál.")

    # Végső modell mentése
    agent.save()
    logging.info("Tanítás befejezve. A végső modell mentve.")

if __name__ == "__main__":
    main()
