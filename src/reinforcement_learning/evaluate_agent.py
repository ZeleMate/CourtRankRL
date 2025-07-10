"""
Agent kiértékelése: A reinforcement learning alapú rangsoroló teljesítményének mérése
a szakértői értékelések alapján. A rendszer az Azure-ból tölti be az adatokat.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import io
import logging

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.reinforcement_learning.agent import RLAgent
from src.reinforcement_learning.environment import RankingEnv
from src.reinforcement_learning.reward_models.reward import compute_ndcg
from src.utils.azure_blob_storage import AzureBlobStorage
from configs import config

# --- Beállítások ---
EVALUATION_TOP_K = 10 # NDCG@k

# Loggolás beállítása
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_expert_evaluations_from_azure(blob_storage: AzureBlobStorage) -> pd.DataFrame:
    """Loads expert evaluation data from a CSV in Azure Blob Storage."""
    logging.info(f"Szakértői értékelések letöltése: {config.BLOB_EXPERT_EVALUATIONS_CSV}")
    try:
        data = blob_storage.download_data(config.BLOB_EXPERT_EVALUATIONS_CSV)
        df = pd.read_csv(io.BytesIO(data))
        logging.info(f"✅ Sikeresen betöltve {len(df)} értékelés.")
        return df
    except Exception as e:
        logging.error(f"Hiba a szakértői értékelések letöltésekor: {e}", exc_info=True)
        return pd.DataFrame()

def get_relevance_scores(ranked_doc_ids: list[str], eval_df: pd.DataFrame, query: str) -> list[float]:
    """Retrieves relevance scores for a ranked list of document IDs."""
    query_evals = eval_df[eval_df['query'] == query].set_index('doc_id')
    return [query_evals.loc[doc_id, 'relevance'] if doc_id in query_evals.index else 0.0 for doc_id in ranked_doc_ids]

def main():
    """Main evaluation loop for the RL agent."""
    logging.info("🚀 RL ÜGYNÖK KIÉRTÉKELÉSÉNEK INDÍTÁSA")

    # 1. Erőforrások inicializálása
    try:
        blob_storage = AzureBlobStorage(container_name=config.AZURE_CONTAINER_NAME)
        # Az env inicializálja a keresőt is, amire szükségünk van a jelöltekhez
        env = RankingEnv(initial_top_k=EVALUATION_TOP_K)
        
        agent_input_dim = env.observation_space.shape[0]
        agent_output_dim = env.action_space.shape[0]
        agent = RLAgent(input_dim=agent_input_dim, output_dim=agent_output_dim)
        agent.load() # Betöltjük a tanított ügynököt

    except Exception as e:
        logging.error(f"Hiba az inicializálás során: {e}", exc_info=True)
        sys.exit(1)

    # 2. Szakértői értékelések betöltése
    eval_df = load_expert_evaluations_from_azure(blob_storage)
    if eval_df.empty:
        logging.error("Nincsenek szakértői értékelések, a kiértékelés leáll.")
        sys.exit(1)
        
    queries_for_eval = eval_df['query'].unique().tolist()
    logging.info(f"Kiértékelés {len(queries_for_eval)} egyedi lekérdezés alapján.")

    # 3. Kiértékelési ciklus
    results_data = []
    for query in tqdm(queries_for_eval, desc="Lekérdezések kiértékelése"):
        # Kezdeti állapot és jelöltek lekérése
        state, info = env.reset(query=query)
        initial_candidates = info['candidates']
        
        # Kezdeti (szemantikus) rangsor NDCG-je
        initial_doc_ids = [res.doc_id for res in initial_candidates]
        initial_relevance = get_relevance_scores(initial_doc_ids, eval_df, query)
        initial_ndcg = compute_ndcg(np.array(initial_relevance), k=EVALUATION_TOP_K)
        
        # Ügynök általi újra-rangsorolás
        action = agent.select_action(state)
        
        # A step metódus elvégzi a rangsorolást
        _, _, _, _, step_info = env.step(action)
        reranked_results = step_info['reranked_results']
        reranked_doc_ids = [res.doc_id for res in reranked_results]

        # Újra-rangsorolt lista NDCG-je
        reranked_relevance = get_relevance_scores(reranked_doc_ids, eval_df, query)
        reranked_ndcg = compute_ndcg(np.array(reranked_relevance), k=EVALUATION_TOP_K)
        
        results_data.append({
            "query": query,
            f"initial_ndcg@{EVALUATION_TOP_K}": initial_ndcg,
            f"reranked_ndcg@{EVALUATION_TOP_K}": reranked_ndcg,
            "improvement": reranked_ndcg - initial_ndcg
        })

    if not results_data:
        logging.warning("Nem sikerült egyetlen lekérdezést sem kiértékelni.")
        return

    # 4. Eredmények összesítése és megjelenítése
    results_df = pd.DataFrame(results_data)
    print("\n--- KIÉRTÉKELÉSI ÖSSZEFOGLALÓ ---")
    print(results_df.to_string())
    
    print("\n--- STATISZTIKÁK ---")
    print(results_df.describe())
    
    avg_improvement = results_df['improvement'].mean()
    print(f"\nÁtlagos NDCG javulás: {avg_improvement:+.4f}")
    
    logging.info("🎉 Kiértékelés befejezve.")

if __name__ == "__main__":
    main()
