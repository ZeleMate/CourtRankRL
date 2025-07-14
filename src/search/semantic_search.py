from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import faiss
import json
import sys
import logging
import networkx as nx
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
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
from src.utils.azure_blob_storage import AzureBlobStorage
# A modellek importálása mostantól a src.models-ból történik
from src.models.embedding import get_embedding_model

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

class HybridSearch:
    """
    Hibrid keresési motor, amely szemantikus és gráf-alapú keresést kombinál.
    Az adatokat az Azure Blob Storage-ból tölti be inicializáláskor.
    """
    
    def __init__(self):
        """Inicializálja a keresőt és betölti az összes szükséges adatot az Azure-ból."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Hibrid kereső motor inicializálása...")

        try:
            self.blob_storage = AzureBlobStorage(container_name=config.AZURE_CONTAINER_NAME)
            self._load_all_data()
            self._precompute_metrics()
            # Az embedding modell betöltése a végén, hogy a többi komponens gyorsan betöltődjön
            self.embedding_model = get_embedding_model()
        except Exception as e:
            self.logger.error(f"Kritikus hiba a kereső inicializálása közben: {e}", exc_info=True)
            raise

        self.logger.info("✅ Hibrid kereső motor sikeresen inicializálva.")
    
    def _load_all_data(self):
        """Minden szükséges adat (index, gráf, metaadatok) betöltése a Blob Storage-ból."""
        self.logger.info("Adatok betöltése az Azure Blob Storage-ból...")
        
        # FAISS index
        self.logger.info(f"FAISS index letöltése: {config.BLOB_FAISS_INDEX}")
        index_data = self.blob_storage.download_data(config.BLOB_FAISS_INDEX)
        self.faiss_index = faiss.read_index(faiss.PyCallbackIOReader(index_data))
        
        # FAISS ID -> doc_id leképezés
        self.logger.info(f"ID leképezés letöltése: {config.BLOB_FAISS_DOC_ID_MAP}")
        map_data = self.blob_storage.download_data(config.BLOB_FAISS_DOC_ID_MAP)
        # A JSON kulcsok stringek, de a FAISS int-eket ad vissza, ezért konvertálunk
        self.id_map = {int(k): v for k, v in json.loads(map_data).items()}
        
        # Gráf
        self.logger.info(f"Gráf letöltése: {config.BLOB_GRAPH}")
        graph_data = self.blob_storage.download_data(config.BLOB_GRAPH)
        self.graph = pickle.load(io.BytesIO(graph_data))
        
        # Dokumentum szövegek és metaadatok
        self.logger.info(f"Dokumentum metaadatok letöltése: {config.BLOB_CLEANED_DOCUMENTS_PARQUET}")
        docs_data = self.blob_storage.download_data(config.BLOB_CLEANED_DOCUMENTS_PARQUET)
        df_docs = pd.read_parquet(io.BytesIO(docs_data))
        self.doc_metadata = df_docs.set_index('doc_id').to_dict('index')

        self.logger.info("✅ Minden adat sikeresen betöltve.")

    def _precompute_metrics(self):
        """Gráf metrikák előszámítása a gyorsabb keresés érdekében."""
        self.logger.info("Gráf metrikák előszámítása...")
        self.pagerank = nx.pagerank(self.graph)
        # Normalizálás, hogy a pontszámok 0 és 1 között legyenek
        self.scaler = MinMaxScaler()
        pr_values = np.array(list(self.pagerank.values())).reshape(-1, 1)
        self.normalized_pagerank = dict(zip(self.pagerank.keys(), self.scaler.fit_transform(pr_values).flatten()))
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
    
    print("\n✅ Keresőmotor készen áll.")
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