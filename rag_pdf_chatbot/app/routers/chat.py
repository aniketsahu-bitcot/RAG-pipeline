"""Chat and session-history endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.exceptions import DocumentNotFoundError, SessionNotFoundError
from app.dependencies import get_rag_service, get_session_service
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    SessionHistoryResponse,
)
from app.services.rag_service import RAGService
from app.services.session_service import SessionService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse, summary="Ask a question (multi-turn)")
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    session_service: SessionService = Depends(get_session_service),
) -> ChatResponse:
    """Answer a question using RAG, maintaining multi-turn session history.

    If session_id is omitted or unknown, a new session is created and
    returned in the response so the client can reuse it on the next call.
    """
    if request.session_id and session_service.session_exists(request.session_id):
        session_id = request.session_id
    else:
        session_id = session_service.create_session()

    try:
        return rag_service.answer_query(request.query, session_id, request.document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get a session's conversation history",
)
async def get_session_history(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
) -> SessionHistoryResponse:
    """Return the full turn history for a session."""
    try:
        turns = session_service.get_history(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return SessionHistoryResponse(session_id=session_id, turns=list(turns))


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear a session's history",
)
async def delete_session(
    session_id: str,
    session_service: SessionService = Depends(get_session_service),
) -> None:
    """Clear a session's history."""
    try:
        session_service.clear_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
