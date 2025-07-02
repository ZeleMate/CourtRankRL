from typing import List, Tuple, Dict, Optional, Set
# Importáljuk az újonnan létrehozott EmbeddingModel-t
# Feltételezzük, hogy a 'models' könyvtár a python path-on van,
# vagy a projekt gyökeréből futtatjuk a kódot.
from models.embedding import OpenAIEmbeddingModel
import numpy as np
import faiss
import json
import os
import sys
import logging
import networkx as nx
from dataclasses import dataclass
from collections import defaultdict, Counter
from sklearn.preprocessing import MinMaxScaler
import math

# Projekt gyökérkönyvtárának hozzáadása a Python útvonalhoz
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from configs import config

@dataclass
class SearchResult:
    """Keresési eredmény reprezentálása."""
    doc_id: str
    semantic_score: float
    graph_score: float
    temporal_score: float
    authority_score: float
    diversity_score: float
    hybrid_score: float
    rank: int
    connections: Dict[str, List[str]] = None

@dataclass 
class QueryExpansion:
    """Query expansion eredmények."""
    original_terms: List[str]
    expanded_terms: List[str]
    legal_concepts: List[str]
    related_statutes: List[str]

class AdvancedHybridSearch:
    """Fejlett hibrid keresési motor: többrétegű pontszámítás és intelligens re-ranking."""
    
    def __init__(self, faiss_index_path: str, graph_path: str, embeddings_path: str, 
                 document_metadata_path: Optional[str] = None, enable_chunked_loading: bool = True):
        """
        Fejlett hibrid keresési motor inicializálása.
        ÚJDONSÁG: Chunked embedding támogatás a szemantikai kereshetőség megőrzésével.
        """
        self.logger = logging.getLogger(__name__)
        
        # FAISS index betöltése
        self.logger.info(f"FAISS index betöltése: {faiss_index_path}")
        self.faiss_index = faiss.read_index(faiss_index_path)
        
        # Gráf betöltése
        self.logger.info(f"Gráf betöltése: {graph_path}")
        self.graph = self._load_graph(graph_path)
        
        # ===== CHUNKED EMBEDDING METADATA BETÖLTÉSE =====
        self.embedding_metadata = self._load_embedding_metadata_smart(
            embeddings_path, enable_chunked_loading
        )
        
        # Dokumentum metaadatok (opcionális)
        self.document_metadata = {}
        if document_metadata_path and os.path.exists(document_metadata_path):
            with open(document_metadata_path, 'r', encoding='utf-8') as f:
                self.document_metadata = json.load(f)
        
        # Dokumentum ID-k listája
        self.doc_ids = self.embedding_metadata.get('doc_ids', [])
        
        # Fejlett gráf metrikák előszámítása
        self.logger.info("Fejlett gráf metrikák előszámítása...")
        self._precompute_advanced_metrics()
        
        # Jogi fogalmak szótára (statikus vagy betölthető)
        self._build_legal_concept_dictionary()
        
        self.logger.info("Fejlett hibrid keresési motor inicializálva.")
    
    def _load_graph(self, graph_path: str) -> nx.DiGraph:
        """Gráf betöltése GraphML vagy JSON formátumból."""
        try:
            if graph_path.endswith('.graphml'):
                return nx.read_graphml(graph_path)
            elif graph_path.endswith('.json'):
                with open(graph_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                return nx.node_link_graph(graph_data)
            else:
                raise ValueError(f"Nem támogatott gráf formátum: {graph_path}")
        except Exception as e:
            self.logger.error(f"Hiba a gráf betöltése során: {e}")
            raise
    
    def _load_embedding_metadata_smart(self, embeddings_path: str, enable_chunked: bool) -> Dict:
        """
        Intelligens embedding metadata betöltés chunked támogatással.
        
        1. Elsőként chunked parquet fájlokat keres
        2. Ha nincs, fallback az egyesített JSON-ra
        3. Biztosítja a szemantikai kereshetőség megőrzését
        """
        # ===== 1. CHUNKED PARQUET KERESÉSE =====
        if enable_chunked:
            embeddings_dir = os.path.dirname(embeddings_path)
            chunked_parquet_pattern = os.path.join(embeddings_dir, "*_with_embeddings.parquet")
            
            # Chunk parquet fájlok keresése
            import glob
            chunk_parquet_files = glob.glob(chunked_parquet_pattern)
            
            if chunk_parquet_files:
                self.logger.info(f"🎯 CHUNKED PARQUET MÓD: {len(chunk_parquet_files)} embedding chunk található")
                return self._load_from_chunked_parquet(chunk_parquet_files)
        
        # ===== 2. UNIFIED JSON FALLBACK =====
        if os.path.exists(embeddings_path):
            self.logger.info(f"📄 UNIFIED JSON MÓD: Fallback embedding JSON betöltése")
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # ===== 3. HIBA ESETÉN =====
        raise FileNotFoundError(
            f"Nincs elérhető embedding metadata! "
            f"Sem chunked parquet ({chunked_parquet_pattern}), sem unified JSON ({embeddings_path})"
        )
    
    def _load_from_chunked_parquet(self, chunk_files: List[str]) -> Dict:
        """
        Chunked parquet fájlokból embedding metadata konstruálása.
        
        KRITIKUS: Biztosítja, hogy a doc_id sorrend konzisztens legyen a FAISS index-szel!
        """
        import pandas as pd
        
        all_doc_ids = []
        all_texts = []
        all_metadata = []
        
        # Chunk fájlok rendezett betöltése (konzisztens sorrend biztosítása)
        sorted_chunk_files = sorted(chunk_files)
        
        for chunk_file in sorted_chunk_files:
            try:
                self.logger.info(f"Chunk parquet betöltése: {os.path.basename(chunk_file)}")
                df_chunk = pd.read_parquet(chunk_file)
                
                # Kötelező oszlopok ellenőrzése
                required_cols = ['doc_id', 'text', 'embedding']
                missing_cols = [col for col in required_cols if col not in df_chunk.columns]
                if missing_cols:
                    raise ValueError(f"Hiányzó oszlopok {chunk_file}-ban: {missing_cols}")
                
                # Adatok hozzáadása
                all_doc_ids.extend(df_chunk['doc_id'].tolist())
                all_texts.extend(df_chunk['text'].tolist())
                
                # Metadata JSON parse-olása
                for _, row in df_chunk.iterrows():
                    metadata_dict = {
                        'doc_id': row['doc_id'],
                        'text': row['text']
                    }
                    
                    # Metadata JSON hozzáadása (ha van)
                    if 'metadata_json' in row and pd.notna(row['metadata_json']):
                        try:
                            parsed_metadata = json.loads(str(row['metadata_json']))
                            metadata_dict.update(parsed_metadata)
                        except:
                            pass  # Hibás JSON esetén alapértelmezett metadata
                    
                    # További oszlopok hozzáadása
                    for col in df_chunk.columns:
                        if col not in ['doc_id', 'text', 'embedding', 'metadata_json']:
                            metadata_dict[col] = row[col] if pd.notna(row[col]) else None
                    
                    all_metadata.append(metadata_dict)
                
            except Exception as e:
                self.logger.error(f"Hiba chunk parquet betöltésében ({chunk_file}): {e}")
                continue
        
        # ===== METADATA STRUKTÚRA KONSTRUKCIÓ =====
        embedding_metadata = {
            'doc_ids': all_doc_ids,
            'texts': all_texts,
            'metadata': all_metadata,
            'source_type': 'chunked_parquet',
            'chunk_count': len(sorted_chunk_files),
            'total_documents': len(all_doc_ids)
        }
        
        self.logger.info(f"✅ Chunked parquet metadata betöltve:")
        self.logger.info(f"  📁 Chunk fájlok: {len(sorted_chunk_files)}")
        self.logger.info(f"  📄 Dokumentumok: {len(all_doc_ids):,}")
        self.logger.info(f"  🔍 Szemantikai kereshetőség: MEGŐRIZVE")
        
        return embedding_metadata
    
    def _precompute_advanced_metrics(self):
        """Fejlett gráf metrikák előszámítása teljesítmény optimalizáláshoz."""
        # Alapvető centralitás metrikák
        self.logger.info("PageRank számítása...")
        self.pagerank = nx.pagerank(self.graph, alpha=0.85, max_iter=100)
        
        self.logger.info("Fokszám centralitás számítása...")
        self.degree_centrality = nx.degree_centrality(self.graph)
        
        # Dokumentum csomópontok
        doc_nodes = {n for n, data in self.graph.nodes(data=True) 
                    if data.get('type') == 'dokumentum'}
        
        # Közöttiség centralitás (mintavételezéssel nagy gráfokhoz)
        if len(doc_nodes) > 1000:
            sample_nodes = list(doc_nodes)[:1000]
            self.betweenness_centrality = nx.betweenness_centrality_subset(
                self.graph, sample_nodes, sample_nodes
            )
        else:
            self.betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        # Authority scores (HITS algoritmus)
        self.logger.info("HITS authority scores számítása...")
        try:
            hits_scores = nx.hits(self.graph, max_iter=100)
            self.authority_scores = hits_scores[1]  # Authority scores
            self.hub_scores = hits_scores[0]        # Hub scores
        except:
            self.authority_scores = {node: 0.0 for node in self.graph.nodes()}
            self.hub_scores = {node: 0.0 for node in self.graph.nodes()}
        
        # Dokumentum évek szerinti csoportosítás (temporális releváncia)
        self.logger.info("Temporális metrikák számítása...")
        self.temporal_weights = self._compute_temporal_weights()
        
        # Jogi terület alapú klaszterek
        self.legal_area_clusters = self._compute_legal_area_clusters()
        
        self.logger.info("Fejlett gráf metrikák előszámítása befejezve.")
    
    def _compute_temporal_weights(self) -> Dict[str, float]:
        """Temporális súlyok számítása dokumentum évek alapján."""
        temporal_weights = {}
        current_year = 2024  # Aktuális év
        
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'dokumentum':
                doc_year = data.get('ev')
                if doc_year and isinstance(doc_year, int):
                    # Exponenciális lecsengés: újabb dokumentumok nagyobb súlyt kapnak
                    age = max(0, current_year - doc_year)
                    temporal_weight = math.exp(-age / 10)  # 10 éves felezési idő
                    temporal_weights[node_id] = temporal_weight
                else:
                    temporal_weights[node_id] = 0.5  # Alapértelmezett súly
        
        return temporal_weights
    
    def _compute_legal_area_clusters(self) -> Dict[str, Set[str]]:
        """Jogi területek szerint csoportosítás."""
        clusters = defaultdict(set)
        
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'dokumentum':
                legal_area = data.get('jogterulet')
                if legal_area:
                    clusters[legal_area].add(node_id)
        
        return dict(clusters)
    
    def _build_legal_concept_dictionary(self):
        """Jogi fogalmak szótárának felépítése a gráfból."""
        self.legal_concepts = set()
        self.statute_references = set()
        
        for node_id, data in self.graph.nodes(data=True):
            if data.get('type') == 'jogszabaly':
                reference = data.get('reference', '')
                if reference:
                    self.statute_references.add(reference.lower())
                    # Jogszabály címek feldolgozása
                    words = reference.lower().split()
                    self.legal_concepts.update(words)
    
    def expand_query(self, query: str) -> QueryExpansion:
        """
        Intelligens query expansion jogi kontextusban.
        """
        query_lower = query.lower()
        original_terms = query_lower.split()
        
        expanded_terms = set(original_terms)
        legal_concepts = []
        related_statutes = []
        
        # Jogi fogalmak keresése
        for concept in self.legal_concepts:
            if any(term in concept or concept in term for term in original_terms):
                legal_concepts.append(concept)
                expanded_terms.add(concept)
        
        # Kapcsolódó jogszabályok keresése
        for statute in self.statute_references:
            if any(term in statute for term in original_terms):
                related_statutes.append(statute)
        
        # Szinonimák és jogi terminológia (ez bővíthető külső szótárral)
        legal_synonyms = {
            'károk': ['kártérítés', 'sérelem', 'veszteség'],
            'szerződés': ['megállapodás', 'szerződéses', 'kontraktus'],
            'per': ['eljárás', 'jogvita', 'peres'],
            'bíróság': ['bírói', 'igazságszolgáltatás', 'ítélkezés'],
            'jogerős': ['végleges', 'jogerőre', 'hatályos']
        }
        
        for term in original_terms:
            if term in legal_synonyms:
                expanded_terms.update(legal_synonyms[term])
        
        return QueryExpansion(
            original_terms=original_terms,
            expanded_terms=list(expanded_terms),
            legal_concepts=legal_concepts[:10],  # Top 10
            related_statutes=related_statutes[:5]  # Top 5
        )
    
    def semantic_search_with_expansion(self, query_embedding: np.ndarray, 
                                     query_expansion: QueryExpansion, 
                                     k: int = 50) -> List[Tuple[str, float]]:
        """
        Fejlett szemantikus keresés query expansion-nel.
        """
        # Alapvető FAISS keresés
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, k * 2)
        
        semantic_results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.doc_ids):
                doc_id = self.doc_ids[idx]
                similarity = 1.0 / (1.0 + distance)
                semantic_results.append((doc_id, similarity))
        
        # Query expansion alapú re-ranking
        expanded_scores = {}
        for doc_id, base_score in semantic_results:
            # Jogi területi egyezés boost
            doc_data = self.graph.nodes.get(doc_id, {})
            legal_area = doc_data.get('jogterulet', '').lower()
            
            area_boost = 0.0
            if legal_area:
                for term in query_expansion.expanded_terms:
                    if term in legal_area:
                        area_boost += 0.1
            
            # Kapcsolódó jogszabályok boost
            statute_boost = 0.0
            doc_connections = self._get_document_statutes(doc_id)
            for statute in query_expansion.related_statutes:
                if any(statute in conn.lower() for conn in doc_connections):
                    statute_boost += 0.15
            
            # Kombinált pontszám
            enhanced_score = base_score + area_boost + statute_boost
            expanded_scores[doc_id] = enhanced_score
        
        # Re-ranking és top-k kiválasztás
        sorted_results = sorted(expanded_scores.items(), 
                              key=lambda x: x[1], reverse=True)
        
        return sorted_results[:k]
    
    def _get_document_statutes(self, doc_id: str) -> List[str]:
        """Dokumentumhoz kapcsolódó jogszabályok lekérése."""
        statutes = []
        if doc_id in self.graph:
            for neighbor in self.graph.successors(doc_id):
                neighbor_data = self.graph.nodes.get(neighbor, {})
                if neighbor_data.get('type') == 'jogszabaly':
                    reference = neighbor_data.get('reference', '')
                    if reference:
                        statutes.append(reference)
        return statutes
    
    def advanced_graph_expansion(self, initial_docs: List[str], 
                               expansion_depth: int = 2,
                               max_neighbors: int = 15) -> Dict[str, Dict[str, float]]:
        """
        Fejlett gráf alapú bővítés többféle pontszámítással.
        """
        expanded_docs = set(initial_docs)
        
        # Többrétegű pontszámítás
        scores = {
            'graph_relevance': {},
            'authority': {},
            'temporal': {},
            'diversity': {}
        }
        
        # Kezdeti dokumentumok pontszámai
        for doc_id in initial_docs:
            if doc_id in self.graph:
                scores['graph_relevance'][doc_id] = 1.0
                scores['authority'][doc_id] = self.authority_scores.get(doc_id, 0.0)
                scores['temporal'][doc_id] = self.temporal_weights.get(doc_id, 0.5)
        
        current_layer = set(initial_docs)
        
        for depth in range(expansion_depth):
            next_layer = set()
            
            for doc_id in current_layer:
                if doc_id not in self.graph:
                    continue
                
                # Kapcsolatok súlyok szerint rendezve
                weighted_neighbors = self._get_weighted_neighbors(doc_id, max_neighbors)
                
                for neighbor, edge_weight, relation_type in weighted_neighbors:
                    if neighbor not in expanded_docs:
                        # Gráf releváncia pontszám
                        base_score = scores['graph_relevance'].get(doc_id, 0.0)
                        decay_factor = 0.8 ** (depth + 1)
                        
                        # Kapcsolat típus szerinti súlyozás
                        type_weights = {
                            'hivatkozik': 1.0,
                            'hivatkozik_jogszabalyra': 0.8,
                            'targyalta': 0.6
                        }
                        type_weight = type_weights.get(relation_type, 0.5)
                        
                        graph_score = (base_score * decay_factor * edge_weight * type_weight)
                        scores['graph_relevance'][neighbor] = max(
                            scores['graph_relevance'].get(neighbor, 0.0), 
                            graph_score
                        )
                        
                        # Authority és temporal pontszámok
                        scores['authority'][neighbor] = self.authority_scores.get(neighbor, 0.0)
                        scores['temporal'][neighbor] = self.temporal_weights.get(neighbor, 0.5)
                        
                        next_layer.add(neighbor)
                        expanded_docs.add(neighbor)
            
            current_layer = next_layer
            if not current_layer:
                break
        
        # Diverzitási pontszám (különböző jogi területek preferálása)
        self._compute_diversity_scores(scores, expanded_docs)
        
        return scores
    
    def _get_weighted_neighbors(self, doc_id: str, max_neighbors: int) -> List[Tuple[str, float, str]]:
        """Súlyozott szomszédok lekérése."""
        neighbors = []
        
        # Kimenő kapcsolatok
        for neighbor in self.graph.successors(doc_id):
            neighbor_data = self.graph.nodes.get(neighbor, {})
            if neighbor_data.get('type') == 'dokumentum':
                edge_data = self.graph[doc_id][neighbor]
                weight = edge_data.get('weight', 1)
                relation_type = edge_data.get('relation_type', '')
                neighbors.append((neighbor, weight, relation_type))
        
        # Bejövő kapcsolatok
        for neighbor in self.graph.predecessors(doc_id):
            neighbor_data = self.graph.nodes.get(neighbor, {})
            if neighbor_data.get('type') == 'dokumentum':
                edge_data = self.graph[neighbor][doc_id]
                weight = edge_data.get('weight', 1)
                relation_type = edge_data.get('relation_type', '')
                neighbors.append((neighbor, weight * 0.8, relation_type))  # Kisebb súly bejövő kapcsolatokhoz
        
        # Rendezés súly szerint és limitálás
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:max_neighbors]
    
    def _compute_diversity_scores(self, scores: Dict[str, Dict[str, float]], 
                                expanded_docs: Set[str]):
        """Diverzitási pontszámok számítása."""
        # Jogi területek eloszlása
        legal_areas = defaultdict(int)
        for doc_id in expanded_docs:
            doc_data = self.graph.nodes.get(doc_id, {})
            legal_area = doc_data.get('jogterulet')
            if legal_area:
                legal_areas[legal_area] += 1
        
        # Diverzitási bonus ritkább területekhez
        total_docs = len(expanded_docs)
        for doc_id in expanded_docs:
            doc_data = self.graph.nodes.get(doc_id, {})
            legal_area = doc_data.get('jogterulet')
            if legal_area and total_docs > 0:
                area_frequency = legal_areas[legal_area] / total_docs
                diversity_score = 1.0 - area_frequency  # Ritkább területek magasabb pontszám
                scores['diversity'][doc_id] = diversity_score
            else:
                scores['diversity'][doc_id] = 0.5
    
    def advanced_hybrid_search(self, query_embedding: np.ndarray, query: str,
                             k: int = 20, weights: Dict[str, float] = None) -> List[SearchResult]:
        """
        Fejlett hibrid keresés többrétegű pontszámítással.
        """
        # Alapértelmezett súlyok
        if weights is None:
            weights = {
                'semantic': 0.40,
                'graph_relevance': 0.25,
                'authority': 0.15,
                'temporal': 0.10,
                'diversity': 0.10
            }
        
        # 1. Query expansion
        query_expansion = self.expand_query(query)
        self.logger.info(f"Query expansion: {len(query_expansion.expanded_terms)} kifejezés")
        
        # 2. Fejlett szemantikus keresés
        semantic_results = self.semantic_search_with_expansion(
            query_embedding, query_expansion, k * 3
        )
        semantic_scores = {doc_id: score for doc_id, score in semantic_results}
        
        # 3. Fejlett gráf bővítés
        initial_docs = [doc_id for doc_id, _ in semantic_results[:k]]
        graph_scores = self.advanced_graph_expansion(initial_docs, expansion_depth=2)
        
        # 4. Összesített pontszámítás
        all_docs = set(semantic_scores.keys()) | set(graph_scores['graph_relevance'].keys())
        
        # Normalizálás
        scaler = MinMaxScaler()
        
        score_arrays = {}
        for score_type in ['semantic'] + list(graph_scores.keys()):
            if score_type == 'semantic':
                values = [semantic_scores.get(doc, 0.0) for doc in all_docs]
            else:
                values = [graph_scores[score_type].get(doc, 0.0) for doc in all_docs]
            
            if values and max(values) > 0:
                normalized = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
                score_arrays[score_type] = {doc: norm_val for doc, norm_val in zip(all_docs, normalized)}
            else:
                score_arrays[score_type] = {doc: 0.0 for doc in all_docs}
        
        # 5. Hibrid pontszám kombinálás
        hybrid_results = []
        
        for doc_id in all_docs:
            semantic_norm = score_arrays['semantic'].get(doc_id, 0.0)
            graph_norm = score_arrays['graph_relevance'].get(doc_id, 0.0)
            authority_norm = score_arrays['authority'].get(doc_id, 0.0)
            temporal_norm = score_arrays['temporal'].get(doc_id, 0.0)
            diversity_norm = score_arrays['diversity'].get(doc_id, 0.0)
            
            # Súlyozott kombinálás
            hybrid_score = (
                weights['semantic'] * semantic_norm +
                weights['graph_relevance'] * graph_norm +
                weights['authority'] * authority_norm +
                weights['temporal'] * temporal_norm +
                weights['diversity'] * diversity_norm
            )
            
            # Kapcsolatok lekérése
            connections = self.get_document_connections(doc_id, max_connections=3)
            
            hybrid_results.append(SearchResult(
                doc_id=doc_id,
                semantic_score=semantic_scores.get(doc_id, 0.0),
                graph_score=graph_scores['graph_relevance'].get(doc_id, 0.0),
                temporal_score=temporal_norm,
                authority_score=authority_norm,
                diversity_score=diversity_norm,
                hybrid_score=hybrid_score,
                rank=0,
                connections=connections
            ))
        
        # 6. Végső rendezés és ranking
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        for i, result in enumerate(hybrid_results[:k]):
            result.rank = i + 1
        
            return hybrid_results[:k]
    
    def get_document_connections(self, doc_id: str, max_connections: int = 5) -> Dict[str, List[str]]:
        """
        Egy dokumentum közvetlen kapcsolatainak lekérése.
        
        Args:
            doc_id: Dokumentum azonosító
            max_connections: Maximum kapcsolatok száma típusonként
            
        Returns:
            Dictionary kapcsolat típusok szerint
        """
        if doc_id not in self.graph:
            return {}
        
        connections = {
            'hivatkozott_dokumentumok': [],
            'hivatkozo_dokumentumok': [],
            'kapcsolodo_jogszabalyok': [],
            'targyalo_birosagok': []
        }
        
        # Kimenő kapcsolatok
        for neighbor in self.graph.successors(doc_id):
            edge_data = self.graph[doc_id][neighbor]
            relation_type = edge_data.get('relation_type', '')
            neighbor_data = self.graph.nodes.get(neighbor, {})
            
            if relation_type == 'hivatkozik' and neighbor_data.get('type') == 'dokumentum':
                connections['hivatkozott_dokumentumok'].append(neighbor)
            elif relation_type == 'hivatkozik_jogszabalyra':
                connections['kapcsolodo_jogszabalyok'].append(neighbor)
            elif relation_type == 'targyalta':
                connections['targyalo_birosagok'].append(neighbor)
        
        # Bejövő kapcsolatok
        for neighbor in self.graph.predecessors(doc_id):
            edge_data = self.graph[neighbor][doc_id]
            relation_type = edge_data.get('relation_type', '')
            neighbor_data = self.graph.nodes.get(neighbor, {})
            
            if relation_type == 'hivatkozik' and neighbor_data.get('type') == 'dokumentum':
                connections['hivatkozo_dokumentumok'].append(neighbor)
        
        # Kapcsolatok limitálása
        for key in connections:
            connections[key] = connections[key][:max_connections]
        
        return connections

class SemanticSearch:
    def __init__(self, embedding_model: OpenAIEmbeddingModel, documents: List[str]):
        self.embedding_model = embedding_model # Most már konkrét típust várunk
        self.documents = documents
        # Itt lehetne inicializálni a dokumentumok indexelését a model segítségével, ha szükséges
        # pl. self.document_embeddings = self.embedding_model.encode(self.documents)
        # és egy Faiss indexet építeni.
        print(f"SemanticSearch initialized with {len(documents)} documents.")
        if not documents:
            print("Warning: SemanticSearch initialized with an empty list of documents.")

    def search_candidates(self, query: str, top_k: int) -> List[Tuple[int, str, float]]:
        """
        Placeholder for semantic search.
        Returns a list of dummy candidates.
        Each candidate is (original_document_index, document_text, initial_score).
        """
        print(f"Warning: SemanticSearch.search_candidates is a placeholder and returns dummy data for query: '{query}'")
        
        if not self.documents:
            print("Warning: No documents available for search. Returning empty list or padding.")
            # Kevesebb mint top_k dummy eredményt adunk vissza, ha nincsenek dokumentumok
            return [(-1, "Padding dummy document text", 0.0)] * top_k 

        # Visszaadunk top_k darab dummy eredményt az elérhető dokumentumokból
        dummy_candidates = []
        num_available_docs = len(self.documents)
        
        for i in range(min(top_k, num_available_docs)):
            dummy_candidates.append(
                (i, self.documents[i][:150] + "...", 0.5) # doc_id, text_snippet, score
            )
        
        # Ha kevesebb releváns dokumentumot találtunk (vagy kevesebb van), mint top_k,
        # töltsük fel üres dummy-kkal, hogy a kimenet mérete mindig top_k legyen.
        while len(dummy_candidates) < top_k:
            dummy_candidates.append(
                (-1, "Padding dummy document text", 0.0) # doc_id = -1 jelzi, hogy ez egy padding elem
            )
            
        return dummy_candidates # Biztosítjuk, hogy pontosan top_k elemet adjunk vissza

if __name__ == '__main__':
    # Példa használat (teszteléshez)
    # Győződj meg róla, hogy az OpenAIEmbeddingModel és a configs.config működik
    try:
        # Szükséges az OpenAI API kulcs a modell inicializálásához
        # Ellenőrizd, hogy a kulcs be van-e állítva a .env fájlban
        from configs import config # Újraimportáljuk a példa kedvéért
        if config.OPENAI_API_KEY:
            print("Initializing OpenAIEmbeddingModel for SemanticSearch example...")
            model = OpenAIEmbeddingModel()
            print("OpenAIEmbeddingModel initialized.")
            
            docs = [
                "Az almafa virágzik a kertben.", 
                "A körte a legfinomabb gyümölcs.", 
                "Budapest Magyarország fővárosa.",
                "Az AI forradalmasítja a technológiát.",
                "A nap süt, az ég kék."
            ]
            search_engine = SemanticSearch(embedding_model=model, documents=docs)
            print("SemanticSearch initialized.")
            
            test_query = "Milyen az időjárás?"
            top_results = 3
            candidates = search_engine.search_candidates(test_query, top_results)
            
            print(f"\nSearch results for query: '{test_query}' (top {top_results}):")
            for idx, (doc_id, text, score) in enumerate(candidates):
                print(f"  {idx+1}. ID: {doc_id}, Score: {score:.2f}, Text: '{text}'")
            
            assert len(candidates) == top_results
            print("\nSemanticSearch placeholder example executed successfully.")
        else:
            print("Skipping SemanticSearch example as OpenAI API key is not configured.")
            
    except ImportError as e:
        print(f"ImportError during SemanticSearch example: {e}")
        print("Ensure `models.embedding.OpenAIEmbeddingModel` and `configs.config` are accessible.")
        print("You might need to run this from the project root or ensure PYTHONPATH is set.")
    except Exception as e:
        print(f"Error during SemanticSearch example: {e}") 