"""Dependency-injection wiring: composes concrete implementations behind abstractions.

Kept as a single composition root so every other module depends only on
abstract service types, satisfying the Dependency Inversion Principle.
"""

from functools import lru_cache

from app.config import get_settings
from app.services.embedding_service import EmbeddingService, OllamaEmbeddingService
from app.services.llm_service import LLMService, OllamaLLMService
from app.services.pdf_service import PDFService
from app.services.rag_service import RAGService
from app.services.session_service import SessionService
from app.services.vector_store_service import ChromaVectorStoreService, VectorStoreService


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Provide the singleton embedding service (Ollama-backed)."""
    settings = get_settings()
    return OllamaEmbeddingService(settings.ollama_base_url, settings.ollama_embedding_model)


@lru_cache
def get_llm_service() -> LLMService:
    """Provide the singleton LLM service (Ollama-backed)."""
    settings = get_settings()
    return OllamaLLMService(settings.ollama_base_url, settings.ollama_llm_model)


@lru_cache
def get_vector_store_service() -> VectorStoreService:
    """Provide the singleton vector store service (Chroma-backed)."""
    settings = get_settings()
    return ChromaVectorStoreService(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
        embedding_service=get_embedding_service(),
    )


@lru_cache
def get_pdf_service() -> PDFService:
    """Provide the singleton PDF service."""
    settings = get_settings()
    return PDFService(settings.chunk_size, settings.chunk_overlap)


@lru_cache
def get_session_service() -> SessionService:
    """Provide the singleton session service."""
    settings = get_settings()
    return SessionService(settings.max_history_turns)


@lru_cache
def get_rag_service() -> RAGService:
    """Provide the singleton RAG orchestrator, wired with its dependencies."""
    settings = get_settings()
    return RAGService(
        pdf_service=get_pdf_service(),
        vector_store=get_vector_store_service(),
        llm_service=get_llm_service(),
        session_service=get_session_service(),
        retrieval_top_k=settings.retrieval_top_k,
    )
