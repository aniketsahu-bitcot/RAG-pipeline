# RAG PDF Q&A Chatbot

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black.svg)](https://ollama.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A FastAPI backend that lets you upload PDF documents and ask questions about them in a
multi-turn conversation. It uses Retrieval-Augmented Generation (RAG): relevant passages
are retrieved from the PDFs and passed to a local Ollama LLM, which answers using only
that context. Everything runs locally — no external API keys required.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Development](#development)

## Features

- **PDF ingestion** — upload a PDF; its text is extracted, chunked, embedded and indexed.
- **Grounded answers** — the LLM answers only from retrieved context and says when it
  doesn't know.
- **Multi-turn chat** — follow-up questions keep the context of earlier turns via a
  `session_id`.
- **Per-document filtering** — optionally restrict retrieval to a single document.
- **Fully local** — LLM and embeddings run on your machine through Ollama.
- **Interactive docs** — Swagger UI at `/docs` for trying every endpoint.
- **Pluggable services** — LLM, embedding and vector store sit behind abstract
  interfaces, so backends can be swapped without touching the RAG logic.

## How It Works

```
Upload:  PDF ──► extract text ──► split into chunks ──► embed (nomic-embed-text) ──► ChromaDB

Chat:    question ──► embed ──► top-k similar chunks from ChromaDB
                                        │
                                        ▼
         answer ◄── llama3.2 ◄── system prompt + session history + context + question
```

## Tech Stack

| Component    | Technology                                   |
| ------------ | -------------------------------------------- |
| API          | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn |
| LLM          | [Ollama](https://ollama.com) — `llama3.2`    |
| Embeddings   | Ollama — `nomic-embed-text`                  |
| Vector store | [ChromaDB](https://www.trychroma.com) (persistent, on disk) |
| PDF parsing  | [pypdf](https://pypi.org/project/pypdf/)     |
| Config       | pydantic-settings (environment variables / `.env`) |
| Lint/format  | [Ruff](https://docs.astral.sh/ruff/)         |

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** installed and running

## Quick Start

1. **Pull the Ollama models** (with Ollama running — start it with `ollama serve` if needed):

   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **(Optional) Configure** — copy the example env file and adjust as needed:

   ```bash
   cp .env.example .env
   ```

4. **Run the API:**

   ```bash
   uvicorn app.main:app --reload
   ```

5. **Open the interactive docs** at <http://localhost:8000/docs>.

## Configuration

All settings are read from environment variables (or a `.env` file) with the `RAG_` prefix.
Every setting has a default, so a `.env` file is optional.

| Variable                     | Default                  | Description                                  |
| ---------------------------- | ------------------------ | -------------------------------------------- |
| `RAG_OLLAMA_BASE_URL`        | `http://localhost:11434` | Ollama server URL                            |
| `RAG_OLLAMA_LLM_MODEL`       | `llama3.2`               | Model used to generate answers               |
| `RAG_OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text`       | Model used to embed chunks and queries       |
| `RAG_CHROMA_PERSIST_DIR`     | `./data/chroma`          | Directory where ChromaDB stores its data     |
| `RAG_CHROMA_COLLECTION_NAME` | `pdf_documents`          | ChromaDB collection name                     |
| `RAG_CHUNK_SIZE`             | `1000`                   | Characters per chunk                         |
| `RAG_CHUNK_OVERLAP`          | `200`                    | Characters shared between adjacent chunks    |
| `RAG_RETRIEVAL_TOP_K`        | `4`                      | Number of chunks retrieved per question      |
| `RAG_MAX_HISTORY_TURNS`      | `10`                     | Max user/assistant turn pairs kept per session |

## API Reference

| Method   | Endpoint                         | Description                                         | Success |
| -------- | -------------------------------- | --------------------------------------------------- | ------- |
| `POST`   | `/documents/upload`              | Upload a PDF (multipart field `file`) and index it  | `201`   |
| `GET`    | `/documents`                     | List indexed documents                              | `200`   |
| `DELETE` | `/documents/{document_id}`       | Delete a document and its chunks                    | `204`   |
| `POST`   | `/chat`                          | Ask a question (multi-turn via `session_id`)        | `200`   |
| `GET`    | `/sessions/{session_id}/history` | Get a session's conversation history                | `200`   |
| `DELETE` | `/sessions/{session_id}`         | Clear a session's history                           | `204`   |

## Usage Examples

**1. Upload a PDF:**

```bash
curl -X POST http://localhost:8000/documents/upload -F "file=@report.pdf"
```

```json
{
  "document_id": "3f2c9a1e-...",
  "filename": "report.pdf",
  "num_chunks": 42,
  "uploaded_at": "2026-10-01T10:15:00Z"
}
```

**2. Ask a question** (omit `session_id` to start a new conversation):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'
```

```json
{
  "session_id": "b7d1e4f0-...",
  "answer": "The document is a quarterly report covering ..."
}
```

**3. Ask a follow-up** — reuse the `session_id`, and optionally restrict retrieval to one
document with `document_id`:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize its main conclusions.", "session_id": "b7d1e4f0-...", "document_id": "3f2c9a1e-..."}'
```

**4. View the conversation history:**

```bash
curl http://localhost:8000/sessions/b7d1e4f0-.../history
```

## Project Structure

```
rag_pdf_chatbot/
├── app/
│   ├── main.py                      # FastAPI app + exception handlers
│   ├── config.py                    # Settings (env-driven)
│   ├── dependencies.py              # Dependency injection wiring
│   ├── core/
│   │   └── exceptions.py            # Domain exceptions
│   ├── models/
│   │   └── schemas.py               # Pydantic request/response models
│   ├── routers/
│   │   ├── documents.py             # Upload / list / delete documents
│   │   └── chat.py                  # Chat + session history endpoints
│   └── services/
│       ├── embedding_service.py     # EmbeddingService (abstract) + Ollama impl
│       ├── llm_service.py           # LLMService (abstract) + Ollama impl
│       ├── vector_store_service.py  # VectorStoreService (abstract) + Chroma impl
│       ├── pdf_service.py           # PDF text extraction + chunking
│       ├── session_service.py       # In-memory multi-turn session history
│       └── rag_service.py           # Orchestrates the services above
├── data/chroma/                     # ChromaDB storage (created at runtime)
├── .env.example                     # Example configuration
├── pyproject.toml                   # Ruff lint/format config
├── requirements.txt                 # Python dependencies
└── README.md
```

## Development

### Linting and formatting

The project uses [Ruff](https://docs.astral.sh/ruff/) (installed via `requirements.txt`,
configured in `pyproject.toml`).

```bash
ruff check .           # lint
ruff check . --fix     # lint and auto-fix
ruff format .          # format
ruff format --check .  # verify formatting without changing files (e.g. in CI)
```

**VS Code:** install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
(`charliermarsh.ruff`). The included `.vscode/settings.json` formats code and organizes
imports on save.

Run `ruff check . --fix` and `ruff format .` before committing.