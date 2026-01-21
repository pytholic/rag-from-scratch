"""Text chunking utilities for RAG pipeline.
Splits documents into overlapping chunks for embedding and retrieval.

Created by @pytholic on 2026.01.20
"""

import re


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Input text to chunk.
        chunk_size: Target size of each chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def chunk_by_sentences(
    text: str,
    max_chunk_size: int = 500,
    overlap_sentences: int = 2,
) -> list[str]:
    """Split text into chunks at sentence boundaries.

    More natural splits that preserve semantic units.

    Args:
        text: Input text to chunk.
        max_chunk_size: Maximum chunk size in characters.
        overlap_sentences: Number of sentences to overlap.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    # Simple sentence splitting (handles ., !, ?)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_size = 0

    # If single sentence exceeds max, fall back to character chunking
    for sentence in sentences:
        sentence_size = len(sentence)
        if sentence_size > max_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            chunks.extend(chunk_text(sentence, max_chunk_size, overlap=50))
            continue

        # Check if adding this sentence exceeds limit
        if current_size + sentence_size + 1 > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))

            # Keep overlap sentences for context
            if overlap_sentences > 0 and len(current_chunk) >= overlap_sentences:
                current_chunk = current_chunk[-overlap_sentences:]
                # character-count + spaces between sentences, need for max_chunk_size check
                current_size = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
            else:
                current_chunk = []
                current_size = 0

        current_chunk.append(sentence)
        current_size += sentence_size + (1 if len(current_chunk) > 0 else 0)

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
