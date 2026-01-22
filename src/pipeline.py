"""RAG pipeline orchestrating chunking, embedding, storage, and generation.

Created by @pytholic on 2026.01.22
"""

from typing import Any

from src.chunker import chunk_by_sentences
from src.embedder import Embedder
from src.generator import BaseGenerator, get_generator
from src.vectorstore import VectorStore


class RAGPipeline:
    """RAG pipeline orchestrating chunking, embedding, storage, and generation."""

    def __init__(
        self,
        generator: BaseGenerator | None = None,
        provider: str = "ollama",
        chunk_size: int = 500,
        overlap_sentences: int = 1,
        top_k: int = 3,
    ):
        self.embedder = Embedder()
        self.vectorstore = VectorStore()
        self.generator = generator or get_generator(provider)
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences
        self.top_k = top_k

    def ingest(self, documents: list[str]) -> int:
        """Ingest documents into the pipeline.

        Args:
            documents: List of document texts.

        Returns:
            Number of chunks created.
        """
        all_chunks: list[str] = []
        for doc in documents:
            chunks = chunk_by_sentences(doc, self.chunk_size, self.overlap_sentences)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        embeddings = self.embedder.embed(all_chunks)
        self.vectorstore.add(all_chunks, embeddings)

        return len(all_chunks)

    def query(self, question: str) -> str:
        """Query the pipeline and generate a response.

        Args:
            question: User question.

        Returns:
            Generated response based on retrieved context.
        """
        query_embedding = self.embedder.embed(question)
        results = self.vectorstore.search(query_embedding, k=self.top_k)

        if not results:
            return "No relevant context found."

        context = [chunk for chunk, _ in results]
        response = self.generator.generate(question, context)
        return response

    def query_with_sources(self, question: str) -> dict[str, Any]:
        """Query and return response with source chunks and scores.

        Args:
            question: User question.

        Returns:
            Dict with 'answer', 'sources' (chunks), and 'scores'.
        """
        query_embedding = self.embedder.embed(question)
        results = self.vectorstore.search(query_embedding, k=self.top_k)

        if not results:
            return {"answer": "No relevant context found.", "sources": [], "scores": []}
        
        # I also want to ouput exact sentences that contained the answer
        # not just the source number


        contexts, scores = zip(*results)

        return {
            "answer": self.generator.generate(question, list(contexts)),
            "sources": contexts,
            "scores": scores,
        }
