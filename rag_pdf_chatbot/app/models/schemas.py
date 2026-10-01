"""Pydantic request/response schemas for the API."""

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Returned after a PDF has been successfully ingested."""

    document_id: str
    filename: str
    num_chunks: int
    uploaded_at: datetime


class DocumentInfo(BaseModel):
    """Metadata describing an indexed document."""

    document_id: str
    filename: str
    num_chunks: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    """Wraps a list of indexed documents."""

    documents: list[DocumentInfo]


class ChatRequest(BaseModel):
    """Incoming chat request body."""

    query: str = Field(..., min_length=1, description="User's question")
    session_id: str | None = Field(
        None, description="Existing session id; a new one is created if omitted"
    )
    document_id: str | None = Field(None, description="Restrict retrieval to a single document")


class ChatResponse(BaseModel):
    """Response returned from the /chat endpoint."""

    session_id: str
    answer: str


class ChatTurn(BaseModel):
    """A single message in a session's conversation history."""

    role: str
    content: str
    timestamp: datetime


class SessionHistoryResponse(BaseModel):
    """Full turn history for a session."""

    session_id: str
    turns: list[ChatTurn]
