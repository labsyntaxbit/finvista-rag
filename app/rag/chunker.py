"""Text preprocessing and document chunking."""

import re
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextChunker:
    """Preprocesses text and splits documents into retrieval-friendly chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def preprocess(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        cleaned = self.preprocess(document["text"])
        chunks = self.splitter.split_text(cleaned)

        result = []
        for i, chunk in enumerate(chunks):
            result.append(
                {
                    "chunk_id": f"{document['filename']}_chunk_{i}",
                    "text": chunk,
                    "filename": document["filename"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )

        logger.info(
            "Created %d chunks from %s (chunk_size=%d, overlap=%d)",
            len(result),
            document["filename"],
            self.chunk_size,
            self.chunk_overlap,
        )
        return result
