from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import faiss
import json
import sys
import logging
import networkx as nx
from dataclasses import dataclass, field
from collections import defaultdict
import math
import io
import pickle
from pathlib import Path
import pandas as pd

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs import config
from src.embedding.create_embeddings_gemini_api import get_embedding_model

@dataclass
class SearchResult:
    """Keresési eredmény reprezentálása."""
    doc_id: str
    score: float
    rank: int
    semantic_score: float = 0.0
    graph_score: float = 0.0
    temporal_score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    text: Optional[str] = None
    metadata: Dict[str, any] = field(default_factory=dict)

# Az Azure SDK naplózási szintjének beállítása, hogy ne legyen túl beszédes
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

class HybridSearch:
    """
    Hibrid keresési motor, amely szemantikus és gráf-alapú keresést kombinál.
    Az adatokat a lokális fájlrendszerből tölti be inicializáláskor.
    """
    
    def __init__(self):
        """Inicializálja a keresőt és betölti az összes szükséges adatot a lokális fájlrendszerből."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Hibrid kereső motor inicializálása...")

        try:
            self._load_all_data()
            self._precompute_metrics()
            # Az embedding modell betöltése a végén, hogy a többi komponens gyorsan betöltődjön
            self.embedding_model = get_embedding_model()
        except Exception as e:
            self.logger.error(f"Kritikus hiba a kereső inicializálása közben: {e}", exc_info=True)
            raise

        self.logger.info("Hibrid kereső motor sikeresen inicializálva.")
    
    def _load_all_data(self):
        """Minden szükséges adat (index, gráf, metaadatok) betöltése a lokális fájlrendszerből."""
        self.logger.info("Adatok betöltése a lokális fájlrendszerből...")
        
        # FAISS index
        self.logger.info(f"FAISS index betöltése: {config.FAISS_INDEX_PATH}")
        self.faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        
        # FAISS ID -> doc_id leképezés
        self.logger.info(f"ID leképezés betöltése: {config.FAISS_DOC_ID_MAP_PATH}")
        with open(config.FAISS_DOC_ID_MAP_PATH, 'r') as f:
            self.id_map = {int(k): v for k, v in json.load(f).items()}
        
        # Gráf
        self.logger.info(f"Gráf betöltése: {config.GRAPH_PATH}")
        with open(config.GRAPH_PATH, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Dokumentum szövegek és metaadatok
        self.logger.info(f"Dokumentum metaadatok betöltése: {config.CLEANED_DOCUMENTS_PARQUET}")
        df_docs = pd.read_parquet(config.CLEANED_DOCUMENTS_PARQUET)
        
        # Ellenőrizzük és kezeljük a duplikált 'doc_id'-kat, ami a .to_dict('index') hibáját okozza
        if df_docs['doc_id'].duplicated().any():
            num_duplicates = df_docs['doc_id'].duplicated().sum()
            self.logger.warning(
                f"Figyelem: {num_duplicates} duplikált 'doc_id' található a metaadatokban. "
                "Ezek problémát okoznak a szótárrá alakításkor. "
                "Csak az első egyedi előfordulásokat tartjuk meg."
            )
            df_docs = df_docs.drop_duplicates(subset=['doc_id'], keep='first')

        self.doc_metadata = df_docs.set_index('doc_id').to_dict('index')

        self.logger.info("✅ Minden adat sikeresen betöltve.")

    def _precompute_metrics(self):
        """Gráf metrikák előszámítása a gyorsabb keresés érdekében."""
        self.logger.info("Gráf metrikák előszámítása...")
        self.pagerank = nx.pagerank(self.graph)
        
        # Normalizálás, hogy a pontszámok 0 és 1 között legyenek, robusztus módon.
        pr_values = np.array(list(self.pagerank.values()))
        max_pr = np.max(pr_values) if len(pr_values) > 0 else 0.0

        if max_pr > 0:
            normalized_values = pr_values / max_pr
        else:
            normalized_values = pr_values # Már eleve 0-k

        self.normalized_pagerank = dict(zip(self.pagerank.keys(), normalized_values))
        self.logger.info("✅ Metrikák előszámítva.")

    def search(self, query: str, top_k: int = 10, alpha: float = 0.6) -> List[SearchResult]:
        """
        Keresés végrehajtása a megadott lekérdezésre.
        
        Args:
            query (str): A felhasználói keresési szöveg.
            top_k (int): A visszaadandó találatok száma.
            alpha (float): A szemantikus (alpha) és gráf (1-alpha) pontszámok súlya.
        
        Returns:
            List[SearchResult]: A találati lista.
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Az alpha súlynak 0 és 1 között kell lennie.")

        self.logger.info(f"Keresés indítása: '{query}' (top_k={top_k}, alpha={alpha})")

        # 1. Szemantikus keresés
        query_embedding = self.embedding_model.embed_query(query)
        distances, indices = self.faiss_index.search(np.array([query_embedding]), top_k * 3) # Több jelöltet kérünk le

        semantic_scores = defaultdict(float)
        for i, idx in enumerate(indices[0]):
            if idx != -1: # Érvényes index
                doc_id = self.id_map.get(int(idx))
                if doc_id:
                                        # A távolság (L2) pontszámmá alakítása (magasabb pont a jobb)
                    semantic_scores[doc_id] = 1.0 / (1.0 + distances[0][i])
        
        # Szemantikus pontszámok normalizálása
        max_sem_score = max(semantic_scores.values()) if semantic_scores else 1.0
        normalized_semantic_scores = {doc_id: score / max_sem_score for doc_id, score in semantic_scores.items()}
        
        # 2. Gráf-alapú pontszámítás
        graph_scores = {
            doc_id: self.normalized_pagerank.get(doc_id, 0.0)
            for doc_id in normalized_semantic_scores.keys()
        }
        
        # 3. Hibrid pontszámítás
        hybrid_scores = {}
        for doc_id in normalized_semantic_scores:
            sem_score = normalized_semantic_scores.get(doc_id, 0.0)
            graph_score = graph_scores.get(doc_id, 0.0)
            hybrid_scores[doc_id] = alpha * sem_score + (1 - alpha) * graph_score

        # 4. Rendezés és eredmények összeállítása
        sorted_docs = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results = []
        for i, (doc_id, score) in enumerate(sorted_docs):
            doc_meta = self.doc_metadata.get(doc_id, {})
            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                rank=i + 1,
                semantic_score=normalized_semantic_scores.get(doc_id, 0.0),
                graph_score=graph_scores.get(doc_id, 0.0),
                components={
                    'semantic_weight': alpha,
                    'graph_weight': 1 - alpha
                },
                text=doc_meta.get('text'),
                metadata=doc_meta
            ))
        
        self.logger.info(f"Keresés befejezve, {len(results)} találat.")
        return results

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Keresőmotor inicializálása... (ez eltarthat egy ideig)")
    searcher = HybridSearch()
    
    print("\nKeresőmotor készen áll.")
    print("Teszt lekérdezés: 'kártérítés'")
    
    try:
        search_results = searcher.search("kártérítés", top_k=5)

        print("\n--- Keresési Eredmények ---")
        for res in search_results:
            print(f"Rank {res.rank}: {res.doc_id} (Score: {res.score:.4f})")
            print(f"  - Szemantikus pontszám: {res.semantic_score:.4f}")
            print(f"  - Gráf pontszám (PageRank): {res.graph_score:.4f}")
            # Szöveg-részlet megjelenítése
            text_snippet = (res.text[:200] + '...') if res.text and len(res.text) > 200 else res.text
            print(f"  - Szöveg: {text_snippet}")
            print("-" * 10)

    except Exception as e:
        print(f"\nHiba a tesztkeresés során: {e}") 