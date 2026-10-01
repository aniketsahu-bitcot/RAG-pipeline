"""Vector storage abstraction and ChromaDB-backed implementation."""

from abc import ABC, abstractmethod
from typing import Any

import chromadb

from app.services.embedding_service import EmbeddingService


class VectorStoreService(ABC):
    """Abstract contract for storing and querying embedded text chunks."""

    @abstractmethod
    def add_chunks(self, document_id: str, filename: str, chunks: list[str]) -> None:
        """Embed and persist a document's text chunks."""

    @abstractmethod
    def query(
        self, query_text: str, top_k: int, document_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the top_k most relevant chunks for a query."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a document."""

    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        """Check whether a document has any stored chunks."""


class ChromaVectorStoreService(VectorStoreService):
    """ChromaDB implementation of VectorStoreService.

    Depends on an injected EmbeddingService (Dependency Inversion) rather
    than embedding text itself, keeping embedding and storage concerns
    separate (Single Responsibility).
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str,
        embedding_service: EmbeddingService,
    ) -> None:
        """Initialize the persistent Chroma client/collection and embedder."""
        self._embedding_service = embedding_service
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_chunks(self, document_id: str, filename: str, chunks: list[str]) -> None:
        """Embed each chunk and add it to the Chroma collection."""
        embeddings = self._embedding_service.embed_batch(chunks)
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"document_id": document_id, "filename": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

    def query(
        self, query_text: str, top_k: int, document_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Embed the query and fetch the nearest stored chunks."""
        query_embedding = self._embedding_service.embed_text(query_text)
        where_filter = {"document_id": document_id} if document_id else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )

        hits: list[dict[str, Any]] = []
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        for text, metadata, distance in zip(documents[0], metadatas[0], distances[0], strict=False):
            hits.append(
                {
                    "text": text,
                    "document_id": metadata["document_id"],
                    "filename": metadata["filename"],
                    "chunk_index": metadata["chunk_index"],
                    "score": 1.0 - distance,
                }
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        """Delete every chunk whose metadata matches the given document_id."""
        self._collection.delete(where={"document_id": document_id})

    def document_exists(self, document_id: str) -> bool:
        """Return True if at least one chunk exists for the document_id."""
        result = self._collection.get(where={"document_id": document_id}, limit=1)
        return len(result.get("ids", [])) > 0
