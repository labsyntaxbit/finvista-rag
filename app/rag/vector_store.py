"""ChromaDB vector store for document embeddings."""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.rag.embeddings import EmbeddingService
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Manages document chunk storage and semantic search in ChromaDB."""

    COLLECTION_NAME = "finvista_documents"

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.embedding_service = EmbeddingService()
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store initialized at %s", self.persist_dir)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_service.embed_texts(texts)
        ids = [c.get("chunk_id", str(uuid.uuid4())) for c in chunks]
        metadatas = [
            {
                "filename": c["filename"],
                "chunk_index": c["chunk_index"],
                "total_chunks": c["total_chunks"],
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Indexed %d chunks into vector store", len(chunks))
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filename_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        top_k = top_k or settings.top_k_results
        query_embedding = self.embedding_service.embed_query(query)

        where_filter = {"filename": filename_filter} if filename_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                retrieved.append(
                    {
                        "text": doc,
                        "filename": metadata.get("filename", "unknown"),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "total_chunks": metadata.get("total_chunks", 1),
                        "score": round(1 - distance, 4),
                    }
                )

        logger.info("Retrieved %d chunks for query", len(retrieved))
        return retrieved

    def get_document_count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> List[str]:
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["metadatas"])
        filenames = {m["filename"] for m in results["metadatas"]}
        return sorted(filenames)

    def delete_document(self, filename: str) -> None:
        results = self.collection.get(where={"filename": filename}, include=[])
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info("Deleted document: %s (%d chunks)", filename, len(results["ids"]))

    def clear_all(self) -> None:
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared all documents from vector store")
