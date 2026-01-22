"""CLI entrypoint for RAG pipeline.

Created by @pytholic on 2026.01.22
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-from-scratch",
        description="Minimal RAG (Retrieval-Augmented Generation) pipeline from scratch",
        epilog="Example: python main.py --provider ollama --data-dir ./data --top-k 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--provider",
        choices=["ollama", "gemini"],
        default="ollama",
        metavar="PROVIDER",
        help="LLM provider to use for answer generation. Choices: ollama, gemini (default: ollama)",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=Path("data"),
        metavar="DIR",
        help="Directory containing documents to ingest. Supports .md and .txt files (default: ./data)",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=3,
        metavar="TOP_K",
        help="Number of most relevant chunks to retrieve for context (default: 3)",
    )
    args = parser.parse_args()

    # Load documents
    # allowed types include markdown, text, and pdf
    ALLOWED_EXTENSIONS = [".md", ".txt"]
    documents: list[Path] = []
    for file in args.data_dir.rglob("*"):
        if file.suffix in ALLOWED_EXTENSIONS:
            documents.append(file)

    # Lazy import to avoid loading models when just showing help
    from src.pipeline import RAGPipeline

    # Initialize pipeline
    print(f"Initializing pipeline with {args.provider}...")
    pipeline = RAGPipeline(provider=args.provider, top_k=args.top_k)

    # Ingest
    num_chunks = pipeline.ingest([doc.read_text() for doc in documents])
    print(f"Ingested {len(documents)} documents into {num_chunks} chunks\n")

    # Interactive query loop
    print("Ready! Type your questions (Ctrl+C to exit)\n")
    try:
        while True:
            question = input("Question: ").strip()
            if not question:
                continue

            result = pipeline.query_with_sources(question)
            print(f"\nAnswer: {result['answer']}\n")
            print(
                f"Sources ({len(result['sources'])} chunks, scores: {[f'{s:.2f}' for s in result['scores']]})\n"
            )
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    main()
