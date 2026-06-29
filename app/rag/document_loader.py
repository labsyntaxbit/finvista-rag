"""PDF document loading and text extraction."""

from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader

from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """Loads and extracts text from PDF documents."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    def load_pdf(self, file_path: str | Path) -> Dict[str, Any]:
        path = Path(file_path)
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}. Only PDF is supported.")

        logger.info("Loading PDF document: %s", path.name)
        reader = PdfReader(str(path))
        pages: List[str] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())

        full_text = "\n\n".join(pages)
        if not full_text.strip():
            raise ValueError(f"No extractable text found in {path.name}")

        logger.info("Extracted %d pages from %s", len(pages), path.name)
        return {
            "filename": path.name,
            "filepath": str(path),
            "text": full_text,
            "page_count": len(reader.pages),
            "char_count": len(full_text),
        }

    def load_from_bytes(self, filename: str, content: bytes) -> Dict[str, Any]:
        import io

        logger.info("Loading PDF from bytes: %s", filename)
        reader = PdfReader(io.BytesIO(content))
        pages: List[str] = []

        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())

        full_text = "\n\n".join(pages)
        if not full_text.strip():
            raise ValueError(f"No extractable text found in {filename}")

        return {
            "filename": filename,
            "filepath": filename,
            "text": full_text,
            "page_count": len(reader.pages),
            "char_count": len(full_text),
        }
