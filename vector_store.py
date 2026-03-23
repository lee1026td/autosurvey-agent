"""ChromaDB-based vector store for document embeddings."""

from pathlib import Path
from typing import Any, Callable

import chromadb
from chromadb.config import Settings


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "research_docs",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            embedding_fn: Function to generate embeddings (e.g., LLMClient.embed)
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection_name = collection_name
        self.embedding_fn = embedding_fn
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(
        self,
        doc_id: str,
        content: str,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a single document to the vector store.

        Args:
            doc_id: Unique document identifier
            content: Document text content
            embedding: Pre-computed embedding (or computed via embedding_fn)
            metadata: Additional document metadata
        """
        if embedding is None and self.embedding_fn:
            embedding = self.embedding_fn(content)

        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding] if embedding else None,
            documents=[content],
            metadatas=[metadata or {}],
        )

    def add_documents(
        self,
        doc_ids: list[str],
        contents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add multiple documents in batch."""
        if not doc_ids:
            return

        self.collection.upsert(
            ids=doc_ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas or [{} for _ in doc_ids],
        )

    def query(
        self,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query the vector store for similar documents.

        Args:
            query_text: Query text (used if query_embedding not provided)
            query_embedding: Pre-computed query embedding
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            List of matching documents with scores and metadata
        """
        if query_embedding is None and self.embedding_fn and query_text:
            query_embedding = self.embedding_fn(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding] if query_embedding else None,
            query_texts=[query_text] if query_text and not query_embedding else None,
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted.append({
                    "doc_id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })

        return formatted

    def get_document_count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()

    def delete_documents(self, doc_ids: list[str]) -> None:
        """Delete specific documents by ID."""
        if doc_ids:
            self.collection.delete(ids=doc_ids)
