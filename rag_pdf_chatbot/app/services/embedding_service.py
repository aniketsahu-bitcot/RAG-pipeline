"""Embedding generation abstraction and Ollama-backed implementation.

Defining an abstract EmbeddingService (Interface Segregation + Dependency
Inversion) lets the rest of the app depend on a stable contract instead of
a concrete provider, so swapping Ollama for another embedding backend later
requires no changes outside this module.
"""

from abc import ABC, abstractmethod

import ollama

from app.core.exceptions import EmbeddingServiceError


class EmbeddingService(ABC):
    """Abstract contract for turning text into vector embeddings."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single piece of text."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""


class OllamaEmbeddingService(EmbeddingService):
    """Embedding service backed by a local Ollama server (nomic-embed-text)."""

    def __init__(self, base_url: str, model_name: str) -> None:
        """Initialize with the Ollama host and embedding model name."""
        self._client = ollama.Client(host=base_url)
        self._model_name = model_name

    def embed_text(self, text: str) -> list[float]:
        """Embed a single piece of text via the Ollama embeddings API."""
        try:
            response = self._client.embeddings(model=self._model_name, prompt=text)
            return response["embedding"]
        except Exception as exc:
            raise EmbeddingServiceError(f"Failed to embed text: {exc}") from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts sequentially (Ollama has no native batch API)."""
        return [self.embed_text(text) for text in texts]
