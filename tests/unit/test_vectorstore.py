"""Tests for vectorstore module."""

import numpy as np
import pytest

from src.vectorstore import VectorStore


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.fixture
    def store(self):
        """Create empty vector store for testing."""
        return VectorStore()

    @pytest.fixture
    def sample_data(self):
        """Sample chunks and embeddings for testing."""
        chunks = ["first chunk", "second chunk", "third chunk"]
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        return chunks, embeddings

    def test_empty_store(self, store):
        """Test empty store initialization."""
        assert len(store) == 0
        assert store.embeddings is None
        assert store.chunks == []

    def test_add_chunks(self, store, sample_data):
        """Test adding chunks to store."""
        chunks, embeddings = sample_data
        store.add(chunks, embeddings)

        assert len(store) == 3
        assert store.chunks == chunks
        assert np.array_equal(store.embeddings, embeddings)

    def test_add_multiple_batches(self, store, sample_data):
        """Test adding chunks in multiple batches."""
        chunks, embeddings = sample_data
        store.add(chunks[:2], embeddings[:2])
        store.add(chunks[2:], embeddings[2:])

        assert len(store) == 3
        assert store.chunks == chunks

    def test_add_mismatched_lengths(self, store):
        """Test error when chunks and embeddings lengths don't match."""
        chunks = ["chunk1", "chunk2"]
        embeddings = np.array([[1.0, 0.0]])

        with pytest.raises(ValueError, match="Number of chunks must match"):
            store.add(chunks, embeddings)

    def test_search_empty_store(self, store):
        """Test search on empty store returns empty list."""
        query = np.array([1.0, 0.0, 0.0])
        results = store.search(query)
        assert results == []

    def test_search_returns_most_similar(self, store, sample_data):
        """Test search returns chunks sorted by similarity."""
        chunks, embeddings = sample_data
        store.add(chunks, embeddings)

        # Query similar to first embedding
        query = np.array([0.9, 0.1, 0.0])
        results = store.search(query, k=2)

        assert len(results) == 2
        assert results[0][0] == "first chunk"  # Most similar
        assert results[0][1] > results[1][1]  # Descending order

    def test_search_respects_k(self, store, sample_data):
        """Test search returns at most k results."""
        chunks, embeddings = sample_data
        store.add(chunks, embeddings)

        query = np.array([1.0, 0.0, 0.0])
        results = store.search(query, k=2)

        assert len(results) == 2

    def test_search_k_larger_than_store(self, store, sample_data):
        """Test search when k exceeds number of chunks."""
        chunks, embeddings = sample_data
        store.add(chunks, embeddings)

        query = np.array([1.0, 0.0, 0.0])
        results = store.search(query, k=10)

        assert len(results) == 3  # Returns all available

    def test_len_after_operations(self, store, sample_data):
        """Test __len__ tracks store size correctly."""
        chunks, embeddings = sample_data

        assert len(store) == 0
        store.add(chunks[:1], embeddings[:1])
        assert len(store) == 1
        store.add(chunks[1:], embeddings[1:])
        assert len(store) == 3
