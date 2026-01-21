"""Utility functions for RAG pipeline.

Created by @pytholic on 2026.01.21
"""

import numpy as np


def similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
