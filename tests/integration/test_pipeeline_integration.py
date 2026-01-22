"""Integration tests for the RAG pipeline.

Created by @pytholic on 2026.01.22
"""

import pytest

from src.pipeline import RAGPipeline


@pytest.mark.integration
class TestPipelineIntegration:
    def test_end_to_end(self):
        """Test the end-to-end pipeline."""
        pipeline = RAGPipeline(provider="ollama")
        pipeline.ingest(["Python is a programming language. It was created by Guido van Rossum."])

        result = pipeline.query("Who created Python?")

        assert "Guido" in result
