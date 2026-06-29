"""Embedding generation using sentence-transformers."""

from typing import List

from sentence_transformers import SentenceTransformer

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Generates vector embeddings for text chunks."""

    _model: SentenceTransformer | None = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model

    @property
    def model(self) -> SentenceTransformer:
        if EmbeddingService._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            EmbeddingService._model = SentenceTransformer(self.model_name)
        return EmbeddingService._model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        logger.info("Generating embeddings for %d texts", len(texts))
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
