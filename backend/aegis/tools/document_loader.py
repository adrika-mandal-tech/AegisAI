# backend/aegis/tools/document_loader.py
"""Unified Document Loader backed by the Intelligent DocumentParserRouter.

Supports PDF (text & scanned/hybrid via PyMuPDF + selective OCR, Docling, pypdf),
DOCX, PPTX, XLSX, XLS, CSV, JSON, HTML, Markdown, TXT, ZIP, and images
(PNG, JPG, JPEG, WEBP, TIFF, BMP).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from aegis.core.config import settings
from aegis.core.logging import get_logger
from aegis.ingestion.parser_router import NormalizedDocument, get_parser_router

log = get_logger(__name__)


class DocumentLoader:
    """Load documents and return plain text or a :class:`NormalizedDocument`."""

    def __init__(self) -> None:
        self._router = get_parser_router()
        self._prefer_fast_pdf = bool(getattr(settings, "AEGIS_PDF_FAST_TEXT_PATH", True))
        self._pypdf_available = self._check_pypdf()
        self._ocr_available = bool(getattr(settings, "AEGIS_OCR_ENABLED", True))

    @staticmethod
    def _check_pypdf() -> bool:
        try:
            import pypdf  # noqa: F401, PLC0415

            return True
        except Exception:
            return False

    @property
    def _converter(self) -> Any | None:
        return self._router._get_docling()

    def load_normalized(
        self,
        source: str | Path,
        *,
        document_id: str | None = None,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> NormalizedDocument:
        """Parse *source* via :class:`DocumentParserRouter` and return NormalizedDocument."""
        return self._router.parse(
            source,
            document_id=document_id,
            filename=filename,
            mime_type=mime_type,
            prefer_docling=not self._prefer_fast_pdf,
        )

    def load(self, source: str | Path, *, timeout: float = 120.0) -> str:
        """Parse a local file or HTTP(S) URL and return extracted plain text."""
        src_str = str(source)
        if src_str.startswith(("http://", "https://")):
            from aegis.tools.multi_loader import load_url  # noqa: PLC0415

            chunks = load_url(src_str)
            return "\n\n".join(chunks)

        start = time.time()
        norm = self.load_normalized(source)
        if norm.extracted_text and norm.extracted_text.strip():
            log.info(
                "DocumentLoader.load complete",
                source=src_str,
                parser=norm.parser_used,
                chars=len(norm.extracted_text),
                pages=norm.page_count,
                elapsed=round(time.time() - start, 3),
            )
            return norm.extracted_text

        log.warning(
            "DocumentLoader.load produced empty text",
            source=src_str,
            status=norm.extraction_status,
            errors=norm.errors,
        )
        return ""

    def _load_with_pypdf(self, source: str | Path) -> str:
        """Direct PyPDF helper retained for backward compatibility."""
        doc = NormalizedDocument(
            document_id="probe",
            filename=Path(str(source)).name,
            file_type="pdf",
        )
        if self._router._try_pypdf(Path(str(source)), doc):
            return doc.extracted_text
        return ""

    def _load_with_ocr(self, source: str | Path) -> str:
        """Direct OCR helper retained for backward compatibility."""
        text, _ = self._router._run_ocr_on_image(source)
        return text

    def load_and_chunk(
        self,
        source: str | Path,
        *,
        chunk_size: int | None = None,
        overlap: int | None = None,
        strategy: str | None = None,
        timeout: float = 120.0,
    ) -> list[str]:
        """Load a document and split into overlapping word-based chunks (900w / 10%)."""
        from aegis.ingestion.chunker import chunk_normalized_document, chunk_text  # noqa: PLC0415

        src_str = str(source)
        if src_str.startswith(("http://", "https://")):
            text = self.load(source, timeout=timeout)
            return chunk_text(text, chunk_size=chunk_size, overlap=overlap) if text else []

        norm = self.load_normalized(source)
        if not norm.extracted_text or not norm.extracted_text.strip():
            return []

        doc_chunks = chunk_normalized_document(
            norm,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return [c.text for c in doc_chunks]


_LOADER: DocumentLoader | None = None
_LOADER_LOCK = threading.Lock()


def get_document_loader() -> DocumentLoader:
    """Return the shared process-wide :class:`DocumentLoader`."""
    global _LOADER
    if _LOADER is not None:
        return _LOADER
    with _LOADER_LOCK:
        if _LOADER is None:
            _LOADER = DocumentLoader()
        return _LOADER


def reset_document_loader() -> None:
    """Drop the shared loader (used by tests)."""
    global _LOADER
    with _LOADER_LOCK:
        _LOADER = None
