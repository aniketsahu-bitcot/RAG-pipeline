"""Chat/completion abstraction and Ollama-backed implementation."""

from abc import ABC, abstractmethod

import ollama

from app.core.exceptions import LLMServiceError
from app.models.schemas import ChatTurn


class LLMService(ABC):
    """Abstract contract for generating a chat completion."""

    @abstractmethod
    def generate_answer(self, system_prompt: str, history: list[ChatTurn], user_query: str) -> str:
        """Generate an answer given a system prompt, prior turns, and the new query."""


class OllamaLLMService(LLMService):
    """LLM service backed by a local Ollama server (llama3.2)."""

    def __init__(self, base_url: str, model_name: str) -> None:
        """Initialize with the Ollama host and chat model name."""
        self._client = ollama.Client(host=base_url)
        self._model_name = model_name

    def generate_answer(self, system_prompt: str, history: list[ChatTurn], user_query: str) -> str:
        """Call the Ollama chat API with system prompt + history + new query."""
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_query})

        try:
            response = self._client.chat(model=self._model_name, messages=messages)
            return response["message"]["content"]
        except Exception as exc:
            raise LLMServiceError(f"Failed to generate answer: {exc}") from exc
