"""Semantic retrieval of relevant document chunks."""

from typing import List, Dict, Any, Optional

from app.rag.vector_store import VectorStore
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """Retrieves contextually relevant document chunks for a user query."""

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filename_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        top_k = top_k or settings.top_k_results
        logger.info("Retrieving context for query: %s", query[:80])
        return self.vector_store.search(query, top_k=top_k, filename_filter=filename_filter)

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant documents found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}: {chunk['filename']}, Chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}, "
                f"Relevance: {chunk['score']}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def get_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations = []
        seen = set()
        for chunk in chunks:
            key = (chunk["filename"], chunk["chunk_index"])
            if key not in seen:
                seen.add(key)
                citations.append(
                    {
                        "filename": chunk["filename"],
                        "chunk_index": chunk["chunk_index"],
                        "score": chunk["score"],
                        "excerpt": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    }
                )
        return citations
