"""
Module for semantic search functionality using vector embeddings.
"""

import numpy as np
import faiss
from typing import List, Tuple
from ..models.embedding import EmbeddingModel # Assuming EmbeddingModel is defined correctly

# Keep compute_similarity if needed elsewhere, or remove if only used internally
def compute_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

class SemanticSearch:
    """Class for performing semantic search using a vector index."""

    def __init__(self, model: EmbeddingModel, documents: List[str]):
        """
        Initialize semantic search with documents and an embedding model.

        Args:
            model: Embedding model instance.
            documents: List of document strings.
        """
        self.model = model
        self.documents = documents
        if not documents:
            raise ValueError("Document list cannot be empty.")
        self.doc_embeddings = self._encode_documents(documents)
        self._create_index()

    def _encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode documents into embedding vectors."""
        # Consider batching for large document sets
        embeddings = self.model.encode(documents)
        return np.array(embeddings).astype('float32')

    def _create_index(self):
        """Create FAISS index from document embeddings."""
        if self.doc_embeddings.shape[0] == 0:
            raise ValueError("Cannot create index with zero document embeddings.")
        dimension = self.doc_embeddings.shape[1]
        # Using IndexFlatL2, consider IndexFlatIP for cosine similarity if embeddings are normalized
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.doc_embeddings)

    def search_candidates(self, query: str, top_k: int) -> List[Tuple[int, str, float]]:
        """
        Retrieve the top_k most similar documents to the query based on vector similarity.
        This serves as the candidate generation step for RL re-ranking.

        Args:
            query: Query string.
            top_k: Number of candidate documents to return.

        Returns:
            List of (index, document, similarity_score) tuples.
            Similarity score here is L2 distance (lower is better) from FAISS.
            We might convert it to cosine similarity if needed downstream.
        """
        if not hasattr(self, 'index') or self.index.ntotal == 0:
             return []
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        # Ensure top_k is not greater than the number of documents in the index
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # FAISS returns -1 for invalid indices
                # Convert L2 distance to a similarity score (e.g., 1 / (1 + distance))
                # Or calculate cosine similarity explicitly if needed
                l2_distance = distances[0][i]
                # Example conversion: higher score = more similar
                similarity_score = 1.0 / (1.0 + l2_distance)
                # Alternatively, calculate cosine similarity:
                # cosine_sim = compute_similarity(query_embedding[0], self.doc_embeddings[idx])
                results.append((int(idx), self.documents[int(idx)], float(similarity_score)))

        # Sort by similarity score descending if needed (depends on score calculation)
        results.sort(key=lambda x: x[2], reverse=True)
        return results