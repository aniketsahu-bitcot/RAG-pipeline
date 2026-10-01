"""Document upload/listing/deletion endpoints."""

import os
import shutil
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.exceptions import DocumentNotFoundError, UnsupportedFileTypeError
from app.dependencies import get_rag_service
from app.models.schemas import DocumentListResponse, DocumentUploadResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF and index it for retrieval",
)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentUploadResponse:
    """Upload a PDF, extract its text, chunk it, and index it for retrieval."""
    if not file.filename.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError("Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        info = rag_service.ingest_pdf(tmp_path, file.filename)
    finally:
        os.remove(tmp_path)

    return DocumentUploadResponse(**info.model_dump())


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
async def list_documents(
    rag_service: RAGService = Depends(get_rag_service),
) -> DocumentListResponse:
    """List all currently indexed documents."""
    return DocumentListResponse(documents=rag_service.list_documents())


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete_document(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service),
) -> None:
    """Delete a document and its chunks from the index."""
    try:
        rag_service.delete_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
