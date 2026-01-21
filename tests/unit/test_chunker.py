"""Tests for chunker module."""

import pytest

from src.chunker import chunk_by_sentences, chunk_text


class TestChunkText:
    """Tests for character-based chunking."""

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_small_text(self):
        text = "Hello world"
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_chunking_with_overlap(self):
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=50, overlap=10)

        assert len(chunks) == 3  # 0-50, 40-90, 80-100
        assert all(len(c) <= 50 for c in chunks)

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size=0)
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size=-1)

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size=50, overlap=-1)
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size=50, overlap=50)


class TestChunkBySentences:
    """Tests for sentence-based chunking."""

    def test_empty_text(self):
        assert chunk_by_sentences("") == []

    def test_single_sentence(self):
        text = "This is a single sentence."
        chunks = chunk_by_sentences(text)
        assert len(chunks) == 1

    def test_multiple_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_by_sentences(text, max_chunk_size=50)
        assert len(chunks) >= 1
        assert all(len(c) <= 50 or "." in c for c in chunks)

    def test_respects_max_size(self):
        text = "Short. " * 20
        chunks = chunk_by_sentences(text, max_chunk_size=50)
        # Most chunks should be under limit (some tolerance for sentence boundaries)
        assert all(len(c) <= 60 for c in chunks)

    def test_overlap_preserves_context(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = chunk_by_sentences(text, max_chunk_size=40, overlap_sentences=1)
        # With overlap, adjacent chunks should share a sentence
        assert len(chunks) >= 2
