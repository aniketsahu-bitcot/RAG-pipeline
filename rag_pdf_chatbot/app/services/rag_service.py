"""RAG orchestration: ties PDF ingestion, retrieval, and generation together."""

import uuid
from datetime import datetime, timezone

from app.core.exceptions import DocumentNotFoundError
from app.models.schemas import ChatResponse, DocumentInfo
from app.services.llm_service import LLMService
from app.services.pdf_service import PDFService
from app.services.session_service import SessionService
from app.services.vector_store_service import VectorStoreService

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about uploaded PDF "
    "documents. Use only the provided context to answer. If the answer is "
    "not contained in the context, say you don't know. Be concise."
)


class RAGService:
    """Coordinates PDFService, VectorStoreService, LLMService, and SessionService.

    This class depends only on abstractions for the vector store and LLM
    (Dependency Inversion), and holds no embedding/storage/model-specific
    logic itself (Single Responsibility) -- it purely orchestrates.
    """

    def __init__(
        self,
        pdf_service: PDFService,
        vector_store: VectorStoreService,
        llm_service: LLMService,
        session_service: SessionService,
        retrieval_top_k: int,
    ) -> None:
        """Initialize with all collaborating services and retrieval settings."""
        self._pdf_service = pdf_service
        self._vector_store = vector_store
        self._llm_service = llm_service
        self._session_service = session_service
        self._retrieval_top_k = retrieval_top_k
        self._documents: dict[str, DocumentInfo] = {}

    def ingest_pdf(self, file_path: str, filename: str) -> DocumentInfo:
        """Extract, chunk, embed, and store a PDF; register it in the catalog."""
        text = self._pdf_service.extract_text(file_path)
        chunks = self._pdf_service.chunk_text(text)
        document_id = str(uuid.uuid4())

        self._vector_store.add_chunks(document_id, filename, chunks)

        info = DocumentInfo(
            document_id=document_id,
            filename=filename,
            num_chunks=len(chunks),
            uploaded_at=datetime.now(timezone.utc),
        )
        self._documents[document_id] = info
        return info

    def list_documents(self) -> list[DocumentInfo]:
        """Return metadata for all ingested documents."""
        return list(self._documents.values())

    def delete_document(self, document_id: str) -> None:
        """Remove a document's chunks from the vector store and catalog."""
        if document_id not in self._documents:
            raise DocumentNotFoundError(f"Document '{document_id}' not found.")
        self._vector_store.delete_document(document_id)
        del self._documents[document_id]

    def answer_query(self, query: str, session_id: str, document_id: str | None) -> ChatResponse:
        """Retrieve context, generate an answer, and update session history."""
        if document_id is not None and document_id not in self._documents:
            raise DocumentNotFoundError(f"Document '{document_id}' not found.")

        hits = self._vector_store.query(query, self._retrieval_top_k, document_id)

        context_block = "\n\n".join(
            f"[Source: {h['filename']} chunk {h['chunk_index']}]\n{h['text']}" for h in hits
        )
        augmented_query = (
            f"Context:\n{context_block}\n\nQuestion: {query}" if context_block else query
        )

        history = self._session_service.get_history(session_id)
        answer = self._llm_service.generate_answer(SYSTEM_PROMPT, history, augmented_query)

        self._session_service.add_turn(session_id, "user", query)
        self._session_service.add_turn(session_id, "assistant", answer)

        return ChatResponse(session_id=session_id, answer=answer)
