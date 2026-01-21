"""LLM response generation with context for RAG pipeline.

Created by @pytholic on 2026.01.21
"""

import os
from abc import ABC, abstractmethod

import ollama
from dotenv import find_dotenv, load_dotenv
from google import genai

load_dotenv(find_dotenv(), override=True)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use only the context to answer. If the context doesn't contain the answer, say so.
Be concise and direct."""


class BaseGenerator(ABC):
    """Abstract base class for LLM generators."""

    @abstractmethod
    def generate(self, query: str, context: list[str]) -> str:
        """Generate response given query and context chunks."""
        pass

    def _build_prompt(self, query: str, context: list[str]) -> str:
        """Build prompt with context."""
        context_str = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(context))
        return f"""Context:
{context_str}

Question: {query}

Answer based on the context above:"""


class OllamaGenerator(BaseGenerator):
    """Generator using local Ollama."""

    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def generate(self, query: str, context: list[str]) -> str:
        prompt = self._build_prompt(query, context)
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"] or ""


class GeminiGenerator(BaseGenerator):
    """Generator usign Google Gemini API."""

    def __init__(self, model: str = "gemini-3-flash-preview"):
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)

    def generate(self, query: str, context: list[str]) -> str:
        prompt = self._build_prompt(query, context)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
        )
        return response.text or ""


def get_generator(provider: str = "ollama", **kwargs) -> BaseGenerator:
    """Factory function to get generator by provider name."""
    providers = {
        "ollama": OllamaGenerator,
        "gemini": GeminiGenerator,
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")

    return providers[provider](**kwargs)


if __name__ == "__main__":
    generator = get_generator(provider="gemini", model="gemini-3-flash-preview")
    response = generator.generate(
        query="What is the weather like today?", context=["The weather is sunny and warm today."]
    )
    print(response)
