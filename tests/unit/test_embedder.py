"""Tests for embedder module."""

import numpy as np
import pytest

from src.embedder import Embedder
from src.utils import similarity


class TestEmbedder:
    """Tests for Embedder class."""

    @pytest.fixture
    def embedder(self):
        """Create embedder instance for testing."""
        return Embedder()

    def test_embed_single_text(self, embedder):
        """Test embedding a single text string."""
        text = "This is a test sentence."
        embedding = embedder.embed(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert embedding.shape[0] == 384  # all-MiniLM-L6-v2 dimension

    @pytest.mark.slow
    def test_embed_batch(self, embedder):
        """Test embedding multiple texts at once."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedder.embed(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert embeddings.shape == (3, 384)

    def test_embed_empty_string(self, embedder):
        """Test embedding empty string returns valid embedding."""
        embedding = embedder.embed("")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 384

    @pytest.mark.slow
    def test_similarity_identical_texts(self, embedder):
        """Test similarity of identical texts is close to 1.0."""
        text = "This is a test."
        emb1 = embedder.embed(text)
        emb2 = embedder.embed(text)

        sim = similarity(emb1, emb2)
        assert 0.99 <= sim <= 1.0


    @pytest.mark.slow
    def test_similarity_different_texts(self, embedder):
        """Test similarity of different texts is less than identical."""
        emb1 = embedder.embed("Machine learning is fascinating.")
        emb2 = embedder.embed("The weather is nice today.")

        sim = similarity(emb1, emb2)
        assert 0.0 <= sim < 0.5  # Should be relatively low


    @pytest.mark.slow
    def test_similarity_related_texts(self, embedder):
        """Test similarity of semantically related texts is higher."""
        emb1 = embedder.embed("Machine learning and AI are related.")
        emb2 = embedder.embed("Artificial intelligence and ML are connected.")

        sim = similarity(emb1, emb2)
        assert sim > 0.5  # Should be relatively high

    def test_custom_model(self):
        """Test initialization with custom model name."""
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        embedding = embedder.embed("test")
        assert isinstance(embedding, np.ndarray)
