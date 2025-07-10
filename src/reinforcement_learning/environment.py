"""
Module for the reinforcement learning environment focused on ranking search results.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Tuple, Any
import logging

# Abszolút importok használata
from src.search.semantic_search import HybridSearch, SearchResult

class RankingEnv(gym.Env):
    """
    Megerősítéses tanulási (RL) környezet a keresési eredmények
    újrarangsorolásának tanításához. A környezet a HybridSearch komponenst
    használja a jelöltek lekéréséhez és az állapotok reprezentálásához.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_top_k: int):
        """
        Inicializálja a környezetet.

        Args:
            initial_top_k: A kezdetben lekérendő és újrarangsorolandó jelöltek száma.
        """
        super().__init__()
        logging.info("RankingEnv inicializálása...")
        
        try:
            self.searcher = HybridSearch()
        except Exception as e:
            logging.error(f"Hiba a HybridSearch inicializálása közben: {e}", exc_info=True)
            raise

        self.initial_top_k = initial_top_k
        self.embedding_dim = self.searcher.embedding_model.model.get_sentence_embedding_dimension()
        
        # --- Állapottér (State Space) ---
        # Lekérdezés embedding + K db jelölt dokumentum embeddingjének összefűzése.
        state_dim = self.embedding_dim * (1 + self.initial_top_k)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # --- Akciótér (Action Space) ---
        # Minden dokumentumhoz egy pontszámot rendel, ami alapján az újrarangsorolás történik.
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.initial_top_k,),
            dtype=np.float32
        )

        # Belső állapot
        self.current_query: str = ""
        self.current_query_embedding: np.ndarray = np.zeros(self.embedding_dim, dtype=np.float32)
        self.candidate_results: List[SearchResult] = []
        self.candidate_embeddings: np.ndarray = np.zeros((self.initial_top_k, self.embedding_dim), dtype=np.float32)
        logging.info("RankingEnv sikeresen inicializálva.")

    def reset(self, query: str = "kártérítés", *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Visszaállítja a környezetet egy új lekérdezéssel.
        """
        super().reset(seed=seed)
        self.current_query = query
        
        # Lekérdezés embedding generálása
        self.current_query_embedding = self.searcher.embedding_model.embed_query(query)
        
        # Jelöltek lekérése a hibrid keresőtől
        # A rangsorolási feladathoz a szemantikai pontszám a legfontosabb kiindulási alap, ezért alpha=1.0
        self.candidate_results = self.searcher.search(query, top_k=self.initial_top_k, alpha=1.0)

        # Jelöltek feltöltése, ha kevesebb van a vártnál
        if len(self.candidate_results) < self.initial_top_k:
            num_missing = self.initial_top_k - len(self.candidate_results)
            padding_result = SearchResult(doc_id="PAD", score=0.0, rank=-1)
            self.candidate_results.extend([padding_result] * num_missing)

        # Jelölt dokumentumok embeddingjeinek összegyűjtése
        candidate_doc_ids = [res.doc_id for res in self.candidate_results]
        self.candidate_embeddings = self._get_doc_embeddings(candidate_doc_ids)

        state = self._get_state()
        info = {"current_query": self.current_query, "candidates": self.candidate_results}
        return state, info

    def _get_doc_embeddings(self, doc_ids: List[str]) -> np.ndarray:
        """Visszaadja a dokumentumok embeddingjeit a FAISS indexből."""
        embeddings = np.zeros((len(doc_ids), self.embedding_dim), dtype=np.float32)
        
        # Fordított leképezés: doc_id -> faiss_index
        if not hasattr(self, 'doc_id_to_faiss_id'):
            self.doc_id_to_faiss_id = {v: k for k, v in self.searcher.id_map.items()}

        for i, doc_id in enumerate(doc_ids):
            faiss_id = self.doc_id_to_faiss_id.get(doc_id)
            if faiss_id is not None:
                # A reconstruct metódus visszaadja a vektort az indexből
                embeddings[i] = self.searcher.faiss_index.reconstruct(faiss_id)
        
        return embeddings

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Végrehajt egy lépést a környezetben.
        """
        if len(action) != self.initial_top_k:
            raise ValueError(f"Az akció mérete ({len(action)}) nem egyezik a jelöltek számával ({self.initial_top_k})")

        # Újrarangsorolás az ágens által adott pontszámok alapján
        scored_candidates = list(zip(self.candidate_results, action))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        reranked_results = [item[0] for item in scored_candidates]

        # A jutalom számítása külsőleg történik. Itt egy placeholder.
        reward = 0.0

        done = True
        truncated = False
        info = {
            "query": self.current_query,
            "reranked_results": reranked_results,
            "action_scores": action
        }
        
        state = self._get_state()
        return state, reward, done, truncated, info

    def _get_state(self) -> np.ndarray:
        """
        Összeállítja az állapotvektort a lekérdezés és a jelöltek embeddingjeiből.
        """
        state_parts = [self.current_query_embedding]
        state_parts.extend([emb for emb in self.candidate_embeddings])
        return np.concatenate(state_parts).astype(np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        logging.info("RankingEnv lezárása.")