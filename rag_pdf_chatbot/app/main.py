"""FastAPI application entrypoint."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import RAGApplicationError, UnsupportedFileTypeError
from app.routers import chat, documents

app = FastAPI(
    title="RAG PDF Q&A Chatbot",
    description=(
        "Upload PDFs and ask multi-turn questions over them using a local "
        "Ollama LLM (llama3.2), Ollama embeddings (nomic-embed-text), and ChromaDB."
    ),
    version="1.0.0",
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.exception_handler(UnsupportedFileTypeError)
async def unsupported_file_handler(request: Request, exc: UnsupportedFileTypeError) -> JSONResponse:
    """Return 400 for unsupported file uploads."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RAGApplicationError)
async def rag_error_handler(request: Request, exc: RAGApplicationError) -> JSONResponse:
    """Fallback handler for any unhandled domain error."""
    return JSONResponse(status_code=500, content={"detail": str(exc)})
