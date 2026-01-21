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
```

## Usage

```bash
uv run python main.py
```

## Architecture

| Module | Responsibility |
|--------|----------------|
| `chunker.py` | Split documents into overlapping chunks |
| `embedder.py` | Get embeddings from OpenAI API |
| `vectorstore.py` | Store/search vectors with NumPy |
| `generator.py` | Generate responses with context |
| `pipeline.py` | Orchestrate the RAG flow |

## Goals

- Understand chunking strategies and overlap
- See how embeddings enable semantic search
- Implement cosine similarity from scratch
- Build context-augmented prompts