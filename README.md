# RAG From Scratch

Minimal RAG implementation — NumPy + LLM API only.

## Purpose

Understand the core RAG loop without framework abstractions:

```
documents → chunk → embed → store → query → retrieve → generate
```

## Setup

```bash
uv sync
cp .env.example .env  # Add GEMINI_API_KEY if using Gemini
```

## Usage

```bash
# Run with Ollama (default)
uv run python main.py

# Run with Gemini
uv run python main.py --provider gemini

# Run tests
uv run pytest tests/
```

## Architecture

| Module | Responsibility |
|--------|----------------|
| `chunker.py` | Split documents into overlapping chunks |
| `embedder.py` | Get embeddings via sentence-transformers |
| `vectorstore.py` | Store/search vectors with NumPy |
| `generator.py` | Generate responses (Ollama/Gemini) |
| `pipeline.py` | Orchestrate the RAG flow |

## Goals

- Understand chunking strategies and overlap
- See how embeddings enable semantic search
- Implement cosine similarity from scratch
- Build context-augmented prompts