"""Vector store using NumPy for storage and retrieval."""

import numpy as np

from src.utils import similarity


class VectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self) -> None:
        """Initialize empty vector store."""
        self.embeddings: np.ndarray | None = None
        self.chunks: list[str] = []

    def add(self, chunks: list[str], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the store.

        Args:
            chunks: List of text chunks.
            embeddings: Corresponding embeddings array of shape (n_chunks, embedding_dim).
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if self.embeddings is None:
            self.embeddings = embeddings
            self.chunks = chunks
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Find k most similar chunks to query.

        Args:
            query_embedding: Query vector of shape (embedding_dim,).
            k: Number of results to return.

        Returns:
            List of (chunk, similarity_score) tuples, sorted by similarity descending.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Calculate cosine similarities between query and all embeddings
        similarities = np.array([similarity(query_embedding, emb) for emb in self.embeddings])

        # Get top k indices
        k = min(k, len(self.chunks))
        top_indices = np.argsort(similarities)[::-1][:k]

        return [(self.chunks[i], similarities[i]) for i in top_indices]

    def __len__(self) -> int:
        """Return number of chunks in the store."""
        return len(self.chunks)


