"""
Module for the reinforcement learning environment focused on ranking search results.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Tuple, Any
import logging # logging importálása

# Abszolút importok használata, feltételezve, hogy a projekt gyökere a PYTHONPATH-on van
from models.embedding import OpenAIEmbeddingModel
from search.semantic_search import SemanticSearch

class RankingEnv(gym.Env):
    """
    Megerősítéses tanulási (RL) környezet jogi dokumentumok keresési eredményeinek
    újrarangsorolásának tanításához.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, model: OpenAIEmbeddingModel, documents: List[str], search_engine: SemanticSearch, initial_top_k: int):
        """
        Inicializálja a környezetet.

        Args:
            model: Beágyazási modell példánya (OpenAIEmbeddingModel).
            documents: A korpuszban található összes dokumentum szövegének listája.
            search_engine: SemanticSearch példány a jelöltek lekéréséhez.
            initial_top_k: A kezdetben lekérendő és újrarangsorolandó jelöltek száma.
        """
        super().__init__()
        self.model = model
        self.documents = documents # Ezt a teljes dokumentumlistát valószínűleg nem itt kellene tárolni, ha nagy.
                                 # A search_engine felelőssége lehet a dokumentumokhoz való hozzáférés.
                                 # Jelenleg a SemanticSearch placeholder is megkapja.
        self.search_engine = search_engine
        self.initial_top_k = initial_top_k
        self.embedding_dim = model.get_dimension() # Embedding dimenzió lekérése és tárolása

        # --- Állapottér (State Space) ---
        # A lekérdezést és a kezdeti jelölt dokumentumok listáját reprezentálja.
        # Pl.: Lekérdezés embedding + K dokumentum embedding összefűzése.
        # Óvatos tervezést igényel a policy hálózat elvárásai alapján.
        # embedding_dim = self.model.encode(["test"])[0].shape[0] # Régi mód
        state_dim = self.embedding_dim * (1 + self.initial_top_k) # Query + K docs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, # Embeddings can have negative values
            shape=(state_dim,),
            dtype=np.float32
        )

        # --- Akciótér (Action Space) ---
        # A `initial_top_k` dokumentum újrarangsorolt sorrendjét reprezentálja.
        # 1. opció: Minden dokumentumhoz egy pontszám (akció utáni rendezést igényel). Méret: K.
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

    def reset(self, query: str | None = None, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Visszaállítja a környezetet egy új lekérdezéssel és kezdeti jelölt halmazzal.
        Ez a metódus előkészíti a környezetet egy új epizódra.

        A következő lépéseket hajtja végre:
        1. Beállítja az aktuális lekérdezést és generálja annak embeddingjét.
        2. Lekéri a kezdeti jelölt dokumentumokat a search_engine segítségével.
        3. Feltölti a jelölteket, ha kevesebb érkezik vissza, mint `initial_top_k`.
        4. Generál embeddingeket az érvényes jelölt dokumentumokhoz.
        5. Összeállítja és visszaadja a kezdeti állapot megfigyelést és egy info szótárat.

        Args:
            query: A keresési lekérdezés szövege. Ha None, egy alapértelmezett lekérdezés ("Jogellenes elbocsátás") kerül használatra.
            seed: Opcionális seed a véletlenszám-generátorhoz, amit a `super().reset()` kap meg.
                  Ez segíti a környezet reprodukálhatóságát.
            options: Opcionális, környezet-specifikus beállításokat tartalmazó szótár (jelenleg nincs használatban).

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - observation (np.ndarray): A környezet kezdeti állapota, tipikusan a lekérdezés embedding
                  és a jelölt dokumentumok embeddingjeinek összefűzése. Alakját a `self.observation_space` határozza meg.
                - info (Dict[str, Any]): Egy szótár, ami kiegészítő információkat tartalmaz a reset folyamatról,
                  mint például az aktuális lekérdezés és a jelölt dokumentumok száma. Példa:
                  `{"current_query": self.current_query, "candidate_docs_count": len(self.candidate_docs)}`
        """
        super().reset(seed=seed) # Call super().reset() with seed

        if query is None:
            # Replace with a more robust way to get queries for training/evaluation
            query = "Jogellenes elbocsátás"
        self.current_query = query
        
        query_embedding_result = self.model.encode([query])
        # query_embedding_result is a NumPy array, potentially with a NaN vector
        if query_embedding_result.shape[0] > 0 and not np.all(np.isnan(query_embedding_result[0])):
            self.current_query_embedding = query_embedding_result[0].astype(np.float32)
        else:
            self.current_query_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            logging.warning(f"Query embedding for '{query}' failed or resulted in NaN. Using zero vector.") # logging használata

        self.candidate_docs = self.search_engine.search_candidates(query, self.initial_top_k)

        if len(self.candidate_docs) < self.initial_top_k:
            logging.warning(f"Got {len(self.candidate_docs)} candidates, expected {self.initial_top_k}. Padding.") # logging használata
            num_missing = self.initial_top_k - len(self.candidate_docs)
            dummy_doc = (-1, "", 0.0)
            self.candidate_docs.extend([dummy_doc] * num_missing)

        candidate_texts = [doc[1] for doc in self.candidate_docs if doc[0] != -1 and doc[1]] # Csak valós, nem üres szövegek
        
        if candidate_texts:
            candidate_embeddings_result = self.model.encode(candidate_texts)
            # Most feltételezzük, hogy candidate_embeddings_result annyi sort ad vissza, ahány candidate_texts elem van,
            # és a sikertelenek helyén NaN vektorok vannak.
            self.candidate_embeddings = np.zeros((self.initial_top_k, self.embedding_dim), dtype=np.float32)
            
            valid_text_indices_in_cand_docs = [i for i, doc_tuple in enumerate(self.candidate_docs) if doc_tuple[0] != -1 and doc_tuple[1]]
            num_actual_embeddings_generated = candidate_embeddings_result.shape[0]

            # Iterálunk a generált embeddingeken (ami annyi, ahány valid candidate_text volt)
            for i in range(num_actual_embeddings_generated):
                original_padded_idx = valid_text_indices_in_cand_docs[i] # Index a self.candidate_docs-ban (ami már paddelt lehet)
                if not np.all(np.isnan(candidate_embeddings_result[i])):
                    self.candidate_embeddings[original_padded_idx, :] = candidate_embeddings_result[i].astype(np.float32)
                else:
                    # Ha NaN vektort kaptunk, a self.candidate_embeddings már nullákkal van inicializálva arra a pozícióra
                    logging.warning(f"Candidate embedding for doc '{self.candidate_docs[original_padded_idx][1][:30]}...' resulted in NaN. Using zero vector for this candidate.") # logging használata
        else: # Nincsenek érvényes candidate_texts
            self.candidate_embeddings = np.zeros((self.initial_top_k, self.embedding_dim), dtype=np.float32)
            if self.initial_top_k > 0:
                 logging.warning("No valid candidate texts to embed. Using zero vectors for all candidate embeddings.") # logging használata

        state = self._get_state()
        # info should typically be a dictionary
        info = {"current_query": self.current_query, "candidate_docs_count": len(self.candidate_docs)}
        return state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Egy lépést tesz a környezetben az újrarangsorolási akció alkalmazásával.
        A jutalmat (reward) külsőleg kell meghatározni a `action` által előállított rangsor szakértői értékelése alapján.

        Args:
            action: Az ágenstől kapott akció (pl. pontszámok minden jelölt dokumentumhoz).

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]:
                - observation (np.ndarray): Az aktuális állapot (rangsorolás esetén egy epizódon belül nem változik).
                - reward (float): A jutalom az akció által előállított rangsorért (külső számítást igényel).
                - done (bool): Igaz, mivel a rangsorolás tipikusan egylépéses folyamat lekérdezésenként.
                - truncated (bool): Tipikusan hamis rangsorolásnál, hacsak nincs lépéslimit (itt nincs).
                - info (Dict[str, Any]): Szótár, ami az akció alapján előállított rangsorolt listát tartalmazza.
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
        truncated = False # Typically false for ranking, unless there's a step limit not used here
        info = {
            "query": self.current_query,
            "initial_candidates": self.candidate_docs,
            "ranked_list": ranked_list, # List of (original_idx, text, initial_score) tuples in the new order
            "action_scores": action
        }

        # State doesn't change after the action in this setup
        state = self._get_state()

        return state, reward, done, truncated, info

    def _get_state(self) -> np.ndarray:
        """
        Összeállítja az állapotvektort a lekérdezés és a jelölt dokumentumok embeddingjeiből.
        """
        # embedding_dim_val is now self.embedding_dim, defined in __init__
        
        current_query_embedding_safe = self.current_query_embedding
        if current_query_embedding_safe is None:
            logging.warning("self.current_query_embedding is None in _get_state. Using zero vector.") # logging használata
            current_query_embedding_safe = np.zeros(self.embedding_dim, dtype=np.float32)

        candidate_embeddings_safe = self.candidate_embeddings
        if candidate_embeddings_safe is None:
            logging.warning("self.candidate_embeddings is None in _get_state. Using zero array.") # logging használata
            candidate_embeddings_safe = np.zeros((self.initial_top_k, self.embedding_dim), dtype=np.float32)
        
        # Ensure candidate_embeddings_safe has the correct second dimension if it's not None
        # This check might be less critical if self.embedding_dim is consistently used.
        if candidate_embeddings_safe.shape[1] != self.embedding_dim:
             logging.warning(f"Mismatch in embedding dimensions in _get_state. Expected: {self.embedding_dim}, Candidates got: {candidate_embeddings_safe.shape[1]}. Adjusting candidates.") # logging használata
             adjusted_candidates = np.zeros((candidate_embeddings_safe.shape[0], self.embedding_dim), dtype=np.float32)
             min_dim = min(candidate_embeddings_safe.shape[1], self.embedding_dim)
             adjusted_candidates[:, :min_dim] = candidate_embeddings_safe[:, :min_dim]
             candidate_embeddings_safe = adjusted_candidates

        # state_parts = [current_query_embedding_safe] + [cand_emb for cand_emb in candidate_embeddings_safe]
        # Biztosítjuk, hogy a candidate_embeddings_safe sorai legyenek a listában
        state_parts = [current_query_embedding_safe]
        for i in range(candidate_embeddings_safe.shape[0]):
            state_parts.append(candidate_embeddings_safe[i])
        
        try:
            state = np.concatenate(state_parts).astype(np.float32)
        except ValueError as e:
            logging.error(f"Error during state concatenation: {e}. State parts shapes:") # logging használata
            logging.error(f"Query embedding shape: {current_query_embedding_safe.shape}") # logging használata
            for i, p_part in enumerate(candidate_embeddings_safe):
                logging.error(f"Candidate {i} embedding shape: {p_part.shape}") # logging használata
            if hasattr(self.observation_space, 'shape') and self.observation_space.shape is not None:
                state = np.zeros(self.observation_space.shape, dtype=np.float32)
                logging.warning("Falling back to zero state due to concatenation error.") # logging használata
            else: 
                raise e

        if hasattr(self.observation_space, 'shape') and self.observation_space.shape is not None:
            expected_len = self.observation_space.shape[0]
            if state.shape[0] != expected_len:
                logging.warning(f"State shape mismatch. Expected {expected_len}, got {state.shape[0]}. Adjusting state.") # logging használata
                if state.shape[0] < expected_len:
                    padded_state = np.zeros(expected_len, dtype=np.float32)
                    padded_state[:state.shape[0]] = state
                    state = padded_state
                else:
                    state = state[:expected_len]
        else:
            logging.warning("self.observation_space.shape is None in _get_state. Cannot verify/adjust state shape.") # logging használata

        return state

    def render(self, mode='human'):
        """
        Megjeleníti a környezet állapotát (opcionális).

        Args:
            mode (str): A megjelenítési mód. Jelenleg csak a 'human' támogatott.
        """
        if mode == 'human':
            # A render metódusban a print elfogadható, mivel ez a felhasználói felület része
            print(f"Current Query: {self.current_query}")
            print("Candidates:")
            for i, (idx, doc, score) in enumerate(self.candidate_docs):
                print(f"  {i+1}. (ID: {idx}, Score: {score:.4f}) {doc[:100]}...")
        else:
            super().render() # Gymnasium's Env.render() might not take mode, or handles it. Let's try without for now.

    def close(self):
        """
        Felszabadítja a környezet erőforrásait.
        """
        pass