"""Embed text using Sentence Transformers.

Created by @pytholic on 2026.01.21
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Wrapper for sentence-transformers embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: str | list[str]) -> np.ndarray:
        """Embed single text or batch of texts."""
        return self.model.encode(texts, show_progress_bar=False)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
