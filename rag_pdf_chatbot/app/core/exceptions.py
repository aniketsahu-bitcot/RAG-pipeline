"""Custom application exceptions used across services and routers."""


class RAGApplicationError(Exception):
    """Base class for all application-specific errors."""


class DocumentNotFoundError(RAGApplicationError):
    """Raised when a referenced document_id does not exist."""


class SessionNotFoundError(RAGApplicationError):
    """Raised when a referenced session_id does not exist."""


class UnsupportedFileTypeError(RAGApplicationError):
    """Raised when an uploaded file is not a supported type (PDF only)."""


class EmbeddingServiceError(RAGApplicationError):
    """Raised when the embedding backend fails."""


class LLMServiceError(RAGApplicationError):
    """Raised when the language model backend fails."""
