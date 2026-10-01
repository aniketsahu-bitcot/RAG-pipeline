"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings.

    Reads configuration from environment variables / a .env file so that
    infrastructure details (model names, hosts, ports) stay decoupled from
    application code, in line with the Dependency Inversion Principle.
    """

    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "pdf_documents"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 4

    max_history_turns: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_")


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
