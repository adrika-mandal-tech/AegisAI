# backend/aegis/ingestion/parser_router.py
"""Intelligent Document Parser Router & Normalized Document Model.

Implements Section 3 (Document Intelligence Engine):
- Inspects file extension, MIME type, size, and per-page text density.
- Intelligently routes across:
  1. PyMuPDF (``fitz``) — fast PDF page inspection, text, headings, tables, and
     selective page-pixmap rendering for scanned/hybrid PDFs.
  2. Docling — rich layout-aware document conversion & structured export.
  3. pypdf — lightweight secondary PDF text-layer fallback.
  4. Selective OCR (PaddleOCR / RapidOCR ONNX) — invoked ONLY on image files or
     specific PDF pages whose text layer is missing/sparse (< 40 chars).
  5. Native structured parsers for DOCX (python-docx), PPTX (python-pptx),
     XLSX (openpyxl), XLS (xlrd), CSV, JSON, HTML, Markdown, TXT, and ZIP.
- Produces a unified :class:`NormalizedDocument` schema for downstream
  document-aware chunking, embedding, Qdrant indexing, and dataset analytics.
"""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.core.config import settings
from aegis.core.logging import get_logger

log = get_logger(__name__)

# Minimum extractable non-whitespace characters on a PDF page before we treat
# that specific page as scanned/image-based and trigger selective page OCR.
SPARSE_PAGE_CHAR_THRESHOLD = 40

IMAGE_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp", ".gif",
}
TEXT_EXTENSIONS: set[str] = {
    ".txt", ".md", ".rst", ".log", ".py", ".js", ".ts", ".c", ".cpp", ".h",
    ".yaml", ".yml", ".xml", ".toml", ".ini", ".cfg",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_to_markdown(headers: list[str], rows: list[list[str]], max_rows: int = 200) -> str:
    """Render a structured table into clean GitHub-flavored Markdown."""
    clean_headers = [str(h or "").replace("\n", " ").strip() for h in headers]
    if not any(clean_headers) and rows:
        clean_headers = [f"Col_{i + 1}" for i in range(len(rows[0]))]
    if not clean_headers:
        return ""
    sep = ["---" for _ in clean_headers]
    lines = [
        "| " + " | ".join(clean_headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows[:max_rows]:
        padded = [
            str(row[i] if i < len(row) and row[i] is not None else "").replace("\n", " ").strip()
            for i in range(len(clean_headers))
        ]
        lines.append("| " + " | ".join(padded) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n*(Showing {max_rows} of {len(rows)} rows)*")
    return "\n".join(lines)


@dataclass
class NormalizedDocument:
    """Canonical output of the Document Intelligence Engine (Section 3)."""

    document_id: str
    filename: str
    file_type: str
    page_count: int = 1
    extracted_text: str = ""
    pages: list[dict[str, Any]] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parser_used: str = "unknown"
    ocr_confidence: float | None = None
    extraction_status: str = "empty"  # "success" | "partial" | "empty" | "error"
    processing_timestamp: str = field(default_factory=_utc_now_iso)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dictionary."""
        return asdict(self)


class DocumentParserRouter:
    """Intelligent multi-engine parser router with lazy singleton backends."""

    def __init__(self) -> None:
        self._docling_converter: Any | None = None
        self._docling_checked: bool = False
        self._ocr_engine_obj: Any | None = None
        self._ocr_engine_name: str = ""
        self._ocr_checked: bool = False
        self._lock = threading.Lock()

    # ── Lazy engine initializers ──────────────────────────────────────────

    def _get_docling(self) -> Any | None:
        if self._docling_checked:
            return self._docling_converter
        with self._lock:
            if self._docling_checked:
                return self._docling_converter
            self._docling_checked = True
            try:
                from docling.document_converter import DocumentConverter  # noqa: PLC0415

                self._docling_converter = DocumentConverter()
                log.info("DocumentParserRouter: Docling initialized")
            except Exception as exc:
                log.warning("DocumentParserRouter: Docling unavailable", error=str(exc)[:200])
                self._docling_converter = None
        return self._docling_converter

    def _get_ocr(self) -> tuple[Any | None, str]:
        if not getattr(settings, "AEGIS_OCR_ENABLED", True):
            return None, ""
        if self._ocr_checked:
            return self._ocr_engine_obj, self._ocr_engine_name
        with self._lock:
            if self._ocr_checked:
                return self._ocr_engine_obj, self._ocr_engine_name
            self._ocr_checked = True
            preferred = getattr(settings, "AEGIS_OCR_ENGINE", "auto").lower()

            if preferred == "paddle":
                if self._try_init_paddle():
                    return self._ocr_engine_obj, self._ocr_engine_name

            # RapidOCR (ONNX PP-OCRv4/v5/v6) is fast and works cleanly on macOS arm64 / Python 3.14
            if self._try_init_rapidocr():
                return self._ocr_engine_obj, self._ocr_engine_name

            self._try_init_paddle()
        return self._ocr_engine_obj, self._ocr_engine_name

    def _try_init_paddle(self) -> bool:
        try:
            from paddleocr import PaddleOCR  # noqa: PLC0415

            langs = getattr(settings, "AEGIS_OCR_LANGUAGES", "en").split(",")
            lang = langs[0].strip() if langs else "en"
            try:
                self._ocr_engine_obj = PaddleOCR(use_angle_cls=True, lang=lang)
            except Exception:
                self._ocr_engine_obj = PaddleOCR(lang=lang)
            self._ocr_engine_name = "paddleocr"
            log.info("DocumentParserRouter: PaddleOCR initialized", lang=lang)
            return True
        except Exception as exc:
            log.debug("PaddleOCR init skipped", error=str(exc)[:160])
            return False

    def _try_init_rapidocr(self) -> bool:
        try:
            from rapidocr import RapidOCR  # noqa: PLC0415

            self._ocr_engine_obj = RapidOCR()
            self._ocr_engine_name = "rapidocr"
            log.info("DocumentParserRouter: RapidOCR initialized")
            return True
        except Exception as exc:
            log.debug("RapidOCR init skipped", error=str(exc)[:160])
            return False

    # ── OCR execution helpers ─────────────────────────────────────────────

    def _run_ocr_on_image(self, image_source: str | bytes | Path) -> tuple[str, float | None]:
        """Run OCR on an image path or PNG bytes and return (text, avg_confidence)."""
        ocr_obj, engine_name = self._get_ocr()
        if ocr_obj is None:
            return "", None

        tmp_path: Path | None = None
        try:
            target: str
            if isinstance(image_source, bytes):
                fd, name = tempfile.mkstemp(suffix=".png", prefix="aegis_ocr_")
                os.close(fd)
                tmp_path = Path(name)
                tmp_path.write_bytes(image_source)
                target = str(tmp_path)
            else:
                target = str(image_source)

            if engine_name == "rapidocr":
                raw = ocr_obj(target)
                lines = raw[0] if isinstance(raw, tuple) else getattr(raw, "txts", None) or raw
                # Handle RapidOCR v3 RapidOCROutput object as well as v1/v2 tuple output
                if hasattr(raw, "txts") and raw.txts:
                    txts = [str(t).strip() for t in raw.txts if str(t).strip()]
                    scores = [float(s) for s in (getattr(raw, "scores", None) or []) if s is not None]
                    avg_conf = round(sum(scores) / len(scores), 4) if scores else None
                    return "\n".join(txts), avg_conf

                if not lines:
                    return "", None
                parts: list[str] = []
                confs: list[float] = []
                for item in lines:
                    if not item:
                        continue
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        if isinstance(item[1], str):
                            parts.append(item[1])
                            if len(item) >= 3 and isinstance(item[2], (int, float)):
                                confs.append(float(item[2]))
                        elif isinstance(item[1], (list, tuple)) and item[1]:
                            parts.append(str(item[1][0]))
                            if len(item[1]) >= 2 and isinstance(item[1][1], (int, float)):
                                confs.append(float(item[1][1]))
                avg = round(sum(confs) / len(confs), 4) if confs else None
                return "\n".join(p for p in parts if p.strip()), avg

            # PaddleOCR path
            res = ocr_obj.ocr(target, cls=True)
            parts = []
            confs = []
            for page_res in res or []:
                for line in page_res or []:
                    if line and len(line) >= 2:
                        val = line[1]
                        if isinstance(val, (tuple, list)) and val:
                            parts.append(str(val[0]))
                            if len(val) >= 2 and isinstance(val[1], (int, float)):
                                confs.append(float(val[1]))
                        else:
                            parts.append(str(val))
            avg = round(sum(confs) / len(confs), 4) if confs else None
            return "\n".join(p for p in parts if p.strip()), avg
        except Exception as exc:
            log.warning("OCR execution failed", engine=engine_name, error=str(exc)[:200])
            return "", None
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    # ── Public entry point ────────────────────────────────────────────────

    def parse(
        self,
        source: str | Path,
        *,
        document_id: str | None = None,
        filename: str | None = None,
        mime_type: str | None = None,
        prefer_docling: bool = False,
    ) -> NormalizedDocument:
        """Inspect *source*, dispatch to the best parser, and return NormalizedDocument."""
        start = time.time()
        path = Path(str(source))
        doc_id = document_id or str(uuid.uuid4())
        fname = filename or path.name
        ext = Path(fname).suffix.lower() or path.suffix.lower()
        file_type = ext.lstrip(".") or (mime_type or "unknown")

        doc = NormalizedDocument(
            document_id=doc_id,
            filename=fname,
            file_type=file_type,
        )

        if not path.exists() or not path.is_file():
            doc.extraction_status = "error"
            doc.errors.append(f"File not found: {source}")
            return doc

        file_size = path.stat().st_size
        max_bytes = int(getattr(settings, "AEGIS_DOCUMENT_MAX_BYTES", 100 * 1024 * 1024))
        doc.metadata["file_size_bytes"] = file_size
        doc.metadata["mime_type"] = mime_type or ""

        if file_size == 0:
            doc.extraction_status = "empty"
            doc.errors.append("Uploaded file is empty (0 bytes)")
            return doc

        if file_size > max_bytes:
            doc.extraction_status = "error"
            doc.errors.append(f"File exceeds maximum allowed size ({file_size} > {max_bytes} bytes)")
            return doc

        try:
            if ext == ".pdf":
                self._parse_pdf(path, doc, prefer_docling=prefer_docling)
            elif ext in IMAGE_EXTENSIONS:
                self._parse_image(path, doc)
            elif ext in {".docx", ".doc"}:
                self._parse_docx(path, doc)
            elif ext == ".pptx":
                self._parse_pptx(path, doc)
            elif ext in {".xlsx", ".xls"}:
                self._parse_excel(path, doc, ext)
            elif ext == ".csv":
                self._parse_csv(path, doc)
            elif ext == ".json":
                self._parse_json(path, doc)
            elif ext in {".html", ".htm"}:
                self._parse_html(path, doc)
            elif ext in {".md", ".markdown"} or ext in TEXT_EXTENSIONS:
                self._parse_text_or_markdown(path, doc)
            elif ext == ".zip":
                self._parse_zip(path, doc)
            else:
                # Unknown extension: try Docling first, then plain UTF-8 text
                if not self._try_docling_generic(path, doc):
                    self._parse_text_or_markdown(path, doc)
        except Exception as exc:
            log.exception("DocumentParserRouter.parse unexpected error", filename=fname)
            doc.errors.append(f"Parser error: {exc}")
            if not doc.extracted_text.strip():
                doc.extraction_status = "error"

        elapsed_ms = int((time.time() - start) * 1000)
        doc.metadata["parse_latency_ms"] = elapsed_ms
        doc.metadata["char_count"] = len(doc.extracted_text)
        doc.metadata["word_count"] = len(doc.extracted_text.split())
        doc.metadata["section_count"] = len(doc.sections)
        doc.metadata["table_count"] = len(doc.tables)

        if doc.extracted_text and doc.extracted_text.strip():
            if doc.extraction_status not in {"partial"}:
                doc.extraction_status = "success"
        elif doc.extraction_status != "error":
            doc.extraction_status = "empty"
            if not doc.errors:
                doc.errors.append("No extractable text found in document")

        log.info(
            "DocumentParserRouter.parse complete",
            filename=fname,
            parser=doc.parser_used,
            status=doc.extraction_status,
            pages=doc.page_count,
            words=doc.metadata["word_count"],
            sections=len(doc.sections),
            tables=len(doc.tables),
            latency_ms=elapsed_ms,
        )
        return doc

    # ── Format-specific parsers ───────────────────────────────────────────

    def _parse_pdf(self, path: Path, doc: NormalizedDocument, *, prefer_docling: bool = False) -> None:
        """Route PDF through PyMuPDF (fitz) + selective per-page OCR -> Docling -> pypdf."""
        if prefer_docling and self._try_docling_generic(path, doc):
            return

        # 1. Primary fast & hybrid-aware PDF inspection via PyMuPDF (fitz)
        if self._try_pymupdf(path, doc):
            return

        # 2. Structured layout fallback via Docling
        if self._try_docling_generic(path, doc):
            return

        # 3. Secondary lightweight fallback via pypdf
        if self._try_pypdf(path, doc):
            return

        # 4. Last-resort whole-file OCR
        text, conf = self._run_ocr_on_image(path)
        if text.strip():
            _, engine_name = self._get_ocr()
            doc.extracted_text = text.strip()
            doc.parser_used = engine_name or "ocr"
            doc.ocr_confidence = conf
            doc.pages = [{
                "page_number": 1,
                "text": text.strip(),
                "ocr_used": True,
                "ocr_confidence": conf,
                "headings": [],
            }]

    def _try_pymupdf(self, path: Path, doc: NormalizedDocument) -> bool:
        """Extract per-page text, headings, tables, and selective page OCR via fitz."""
        try:
            import fitz  # PyMuPDF  # noqa: PLC0415
        except ImportError:
            return False

        try:
            pdf = fitz.open(str(path))
        except Exception as exc:
            log.warning("PyMuPDF failed to open PDF", path=str(path), error=str(exc)[:200])
            return False

        try:
            if pdf.is_encrypted:
                try:
                    pdf.authenticate("")
                except Exception:
                    pass
                if pdf.is_encrypted:
                    doc.errors.append("PDF is password-protected / encrypted")
                    return False

            doc.page_count = len(pdf)
            meta = pdf.metadata or {}
            for key in ("title", "author", "subject", "keywords", "creator", "producer"):
                if meta.get(key):
                    doc.metadata[key] = str(meta[key]).strip()

            page_texts: list[str] = []
            ocr_pages = 0
            ocr_confs: list[float] = []
            table_idx = 0

            for idx in range(len(pdf)):
                page_num = idx + 1
                page = pdf[idx]
                raw_text = (page.get_text("text") or "").strip()
                headings = self._extract_fitz_headings(page)
                ocr_used = False
                page_conf: float | None = None

                # Extract structured tables on this page when PyMuPDF table finder is available
                if hasattr(page, "find_tables"):
                    try:
                        found = page.find_tables()
                        for tbl in getattr(found, "tables", []) or []:
                            extracted_rows = tbl.extract() or []
                            if len(extracted_rows) >= 2:
                                headers = [str(c or "").strip() for c in extracted_rows[0]]
                                rows = [
                                    [str(c or "").strip() for c in r]
                                    for r in extracted_rows[1:]
                                ]
                                md = _table_to_markdown(headers, rows)
                                doc.tables.append({
                                    "table_index": table_idx,
                                    "page_number": page_num,
                                    "headers": headers,
                                    "rows": rows[:500],
                                    "row_count": len(rows),
                                    "markdown": md,
                                })
                                table_idx += 1
                    except Exception:
                        pass

                # Selective OCR ONLY when a page has sparse/missing extractable text
                non_ws_len = len(re.sub(r"\s+", "", raw_text))
                if non_ws_len < SPARSE_PAGE_CHAR_THRESHOLD and getattr(settings, "AEGIS_OCR_ENABLED", True):
                    try:
                        pix = page.get_pixmap(dpi=200, alpha=False)
                        ocr_text, conf = self._run_ocr_on_image(pix.tobytes("png"))
                        if ocr_text and len(ocr_text.strip()) > len(raw_text):
                            raw_text = ocr_text.strip()
                            ocr_used = True
                            page_conf = conf
                            ocr_pages += 1
                            if conf is not None:
                                ocr_confs.append(conf)
                    except Exception as ocr_exc:
                        log.debug("Selective page OCR skipped", page=page_num, error=str(ocr_exc)[:160])

                doc.pages.append({
                    "page_number": page_num,
                    "text": raw_text,
                    "ocr_used": ocr_used,
                    "ocr_confidence": page_conf,
                    "headings": headings,
                })

                if headings:
                    for h in headings:
                        doc.sections.append({
                            "heading": h,
                            "level": 2,
                            "text": raw_text,
                            "page_number": page_num,
                        })
                elif raw_text:
                    doc.sections.append({
                        "heading": f"Page {page_num}",
                        "level": 2,
                        "text": raw_text,
                        "page_number": page_num,
                    })

                if raw_text:
                    page_texts.append(f"[Page {page_num}]\n{raw_text}")

            combined = "\n\n".join(page_texts).strip()
            if not combined:
                return False

            doc.extracted_text = combined
            doc.metadata["ocr_pages_count"] = ocr_pages
            if ocr_pages > 0:
                _, ocr_name = self._get_ocr()
                doc.parser_used = f"pymupdf+{ocr_name or 'ocr'}"
                if ocr_confs:
                    doc.ocr_confidence = round(sum(ocr_confs) / len(ocr_confs), 4)
            else:
                doc.parser_used = "pymupdf"
            return True
        finally:
            pdf.close()

    @staticmethod
    def _extract_fitz_headings(page: Any) -> list[str]:
        """Detect section headings on a PDF page from font size relative to body text."""
        try:
            d = page.get_text("dict")
            spans: list[tuple[float, str]] = []
            for block in d.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_texts: list[str] = []
                    max_size = 0.0
                    for span in line.get("spans", []):
                        txt = (span.get("text") or "").strip()
                        if txt:
                            line_texts.append(txt)
                            max_size = max(max_size, float(span.get("size") or 0.0))
                    joined = " ".join(line_texts).strip()
                    if joined and max_size > 0:
                        spans.append((max_size, joined))
            if not spans:
                return []
            sizes = sorted(s for s, _ in spans)
            median_size = sizes[len(sizes) // 2]
            headings: list[str] = []
            for size, txt in spans:
                if size >= median_size * 1.18 and 3 <= len(txt) <= 140:
                    if txt not in headings:
                        headings.append(txt)
            return headings[:8]
        except Exception:
            return []

    def _try_pypdf(self, path: Path, doc: NormalizedDocument) -> bool:
        """Secondary PDF text extraction using pypdf."""
        try:
            import pypdf  # noqa: PLC0415

            with open(path, "rb") as fh:
                reader = pypdf.PdfReader(fh)
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception:
                        pass
                doc.page_count = len(reader.pages)
                page_texts: list[str] = []
                doc.pages = []
                doc.sections = []
                for idx, page in enumerate(reader.pages):
                    page_num = idx + 1
                    txt = (page.extract_text() or "").strip()
                    doc.pages.append({
                        "page_number": page_num,
                        "text": txt,
                        "ocr_used": False,
                        "ocr_confidence": None,
                        "headings": [],
                    })
                    if txt:
                        page_texts.append(f"[Page {page_num}]\n{txt}")
                        doc.sections.append({
                            "heading": f"Page {page_num}",
                            "level": 2,
                            "text": txt,
                            "page_number": page_num,
                        })
                combined = "\n\n".join(page_texts).strip()
                if combined:
                    doc.extracted_text = combined
                    doc.parser_used = "pypdf"
                    return True
        except Exception as exc:
            log.warning("pypdf fallback failed", path=str(path), error=str(exc)[:200])
        return False

    def _try_docling_generic(self, path: Path, doc: NormalizedDocument) -> bool:
        """Structured layout conversion via Docling."""
        converter = self._get_docling()
        if converter is None:
            return False
        try:
            conv_res = converter.convert(str(path))
            dl_doc = conv_res.document
            md_text = ""
            if hasattr(dl_doc, "export_to_markdown"):
                md_text = (dl_doc.export_to_markdown() or "").strip()
            if not md_text and hasattr(dl_doc, "export_to_text"):
                md_text = (dl_doc.export_to_text() or "").strip()
            if not md_text:
                return False

            doc.extracted_text = md_text
            doc.parser_used = "docling"
            if hasattr(dl_doc, "pages") and dl_doc.pages:
                doc.page_count = max(1, len(dl_doc.pages))
            doc.sections = self._extract_markdown_sections(md_text)
            if not doc.pages:
                doc.pages = [{
                    "page_number": 1,
                    "text": md_text,
                    "ocr_used": False,
                    "ocr_confidence": None,
                    "headings": [s["heading"] for s in doc.sections[:10]],
                }]
            return True
        except Exception as exc:
            log.warning("Docling conversion failed", path=str(path), error=str(exc)[:200])
            return False

    def _parse_image(self, path: Path, doc: NormalizedDocument) -> None:
        """Extract text from scanned image files via OCR."""
        text, conf = self._run_ocr_on_image(path)
        _, ocr_name = self._get_ocr()
        doc.page_count = 1
        doc.parser_used = ocr_name or "ocr"
        doc.ocr_confidence = conf
        doc.extracted_text = text.strip()
        doc.pages = [{
            "page_number": 1,
            "text": doc.extracted_text,
            "ocr_used": True,
            "ocr_confidence": conf,
            "headings": [],
        }]
        if doc.extracted_text:
            doc.sections = [{
                "heading": doc.filename,
                "level": 1,
                "text": doc.extracted_text,
                "page_number": 1,
            }]

    def _parse_docx(self, path: Path, doc: NormalizedDocument) -> None:
        """Extract headings, paragraphs, and tables from Word documents."""
        try:
            import docx  # python-docx  # noqa: PLC0415

            d = docx.Document(str(path))
            blocks: list[str] = []
            current_heading = doc.filename
            current_level = 1
            current_lines: list[str] = []

            for para in d.paragraphs:
                txt = (para.text or "").strip()
                if not txt:
                    continue
                style_name = (para.style.name if para.style else "") or ""
                if style_name.lower().startswith("heading"):
                    if current_lines:
                        doc.sections.append({
                            "heading": current_heading,
                            "level": current_level,
                            "text": "\n".join(current_lines),
                            "page_number": 1,
                        })
                        current_lines = []
                    digits = re.findall(r"\d+", style_name)
                    current_level = int(digits[0]) if digits else 2
                    current_heading = txt
                    blocks.append(f"{'#' * min(current_level, 6)} {txt}")
                else:
                    current_lines.append(txt)
                    blocks.append(txt)

            if current_lines:
                doc.sections.append({
                    "heading": current_heading,
                    "level": current_level,
                    "text": "\n".join(current_lines),
                    "page_number": 1,
                })

            for t_idx, tbl in enumerate(d.tables):
                raw_rows = [
                    [(cell.text or "").strip() for cell in row.cells]
                    for row in tbl.rows
                ]
                if not raw_rows:
                    continue
                headers = raw_rows[0]
                rows = raw_rows[1:] if len(raw_rows) > 1 else []
                md = _table_to_markdown(headers, rows)
                doc.tables.append({
                    "table_index": t_idx,
                    "page_number": 1,
                    "headers": headers,
                    "rows": rows[:500],
                    "row_count": len(rows),
                    "markdown": md,
                })
                if md:
                    blocks.append(md)

            combined = "\n\n".join(blocks).strip()
            if combined:
                doc.extracted_text = combined
                doc.parser_used = "python-docx"
                doc.page_count = max(1, len(doc.sections) or 1)
                doc.pages = [{
                    "page_number": 1,
                    "text": combined,
                    "ocr_used": False,
                    "ocr_confidence": None,
                    "headings": [s["heading"] for s in doc.sections],
                }]
                return
        except Exception as exc:
            log.debug("python-docx failed, falling back to Docling", error=str(exc)[:180])

        self._try_docling_generic(path, doc)

    def _parse_pptx(self, path: Path, doc: NormalizedDocument) -> None:
        """Extract per-slide titles, text, speaker notes, and tables from PPTX."""
        try:
            from pptx import Presentation  # noqa: PLC0415

            prs = Presentation(str(path))
            doc.page_count = max(1, len(prs.slides))
            slide_texts: list[str] = []
            t_idx = 0

            for idx, slide in enumerate(prs.slides):
                slide_num = idx + 1
                title = ""
                if slide.shapes.title and slide.shapes.title.text:
                    title = slide.shapes.title.text.strip()
                heading = title or f"Slide {slide_num}"
                lines: list[str] = []

                for shape in slide.shapes:
                    if getattr(shape, "has_text_frame", False) and shape.text_frame:
                        txt = (shape.text_frame.text or "").strip()
                        if txt and txt != title:
                            lines.append(txt)
                    if getattr(shape, "has_table", False) and shape.table:
                        raw_rows = [
                            [(cell.text or "").strip() for cell in row.cells]
                            for row in shape.table.rows
                        ]
                        if raw_rows:
                            headers = raw_rows[0]
                            rows = raw_rows[1:] if len(raw_rows) > 1 else []
                            md = _table_to_markdown(headers, rows)
                            doc.tables.append({
                                "table_index": t_idx,
                                "page_number": slide_num,
                                "headers": headers,
                                "rows": rows[:500],
                                "row_count": len(rows),
                                "markdown": md,
                            })
                            t_idx += 1
                            if md:
                                lines.append(md)

                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = (slide.notes_slide.notes_text_frame.text or "").strip()
                    if notes:
                        lines.append(f"Speaker Notes: {notes}")

                body = "\n".join(lines).strip()
                full_slide = f"[Slide {slide_num}: {heading}]\n{body}".strip()
                doc.pages.append({
                    "page_number": slide_num,
                    "text": full_slide,
                    "ocr_used": False,
                    "ocr_confidence": None,
                    "headings": [heading],
                })
                doc.sections.append({
                    "heading": f"Slide {slide_num}: {heading}",
                    "level": 2,
                    "text": body or heading,
                    "page_number": slide_num,
                })
                slide_texts.append(full_slide)

            combined = "\n\n".join(slide_texts).strip()
            if combined:
                doc.extracted_text = combined
                doc.parser_used = "python-pptx"
                return
        except Exception as exc:
            log.debug("python-pptx failed, falling back to Docling", error=str(exc)[:180])

        self._try_docling_generic(path, doc)

    def _parse_excel(self, path: Path, doc: NormalizedDocument, ext: str) -> None:
        """Extract sheets, headers, rows, and table summaries from XLSX / XLS."""
        sheets_data: list[tuple[str, list[str], list[list[str]]]] = []

        if ext == ".xls":
            try:
                import xlrd  # noqa: PLC0415

                book = xlrd.open_workbook(str(path))
                for s_idx in range(book.nsheets):
                    sheet = book.sheet_by_index(s_idx)
                    if sheet.nrows == 0:
                        continue
                    headers = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]
                    rows = [
                        [str(sheet.cell_value(r, c)).strip() for c in range(sheet.ncols)]
                        for r in range(1, sheet.nrows)
                    ]
                    sheets_data.append((sheet.name, headers, rows))
                doc.parser_used = "xlrd"
            except Exception as exc:
                doc.errors.append(f"XLS parse error: {exc}")

        if not sheets_data:
            try:
                import openpyxl  # noqa: PLC0415

                wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
                try:
                    for sheet_name in wb.sheetnames:
                        ws = wb[sheet_name]
                        raw_rows = list(ws.iter_rows(values_only=True))
                        if not raw_rows:
                            continue
                        headers = [str(c).strip() if c is not None else "" for c in raw_rows[0]]
                        rows = [
                            [str(c).strip() if c is not None else "" for c in r]
                            for r in raw_rows[1:]
                            if any(c is not None and str(c).strip() for c in r)
                        ]
                        sheets_data.append((sheet_name, headers, rows))
                finally:
                    wb.close()
                doc.parser_used = "openpyxl"
            except Exception as exc:
                doc.errors.append(f"XLSX parse error: {exc}")
                return

        doc.page_count = max(1, len(sheets_data))
        text_blocks: list[str] = []
        for idx, (sheet_name, headers, rows) in enumerate(sheets_data):
            md = _table_to_markdown(headers, rows, max_rows=100)
            doc.tables.append({
                "table_index": idx,
                "page_number": idx + 1,
                "sheet_name": sheet_name,
                "headers": headers,
                "rows": rows[:500],
                "row_count": len(rows),
                "markdown": md,
            })
            row_lines = [f"[Sheet: {sheet_name}] Columns ({len(headers)}): {', '.join(h for h in headers if h)} | Total Rows: {len(rows)}"]
            for r in rows:
                pairs = [
                    f"{headers[i] or f'Col_{i+1}'}: {val}"
                    for i, val in enumerate(r)
                    if val and i < len(headers)
                ]
                if pairs:
                    row_lines.append(f"[Sheet: {sheet_name}] " + " | ".join(pairs))
            section_text = "\n".join(row_lines)
            doc.sections.append({
                "heading": f"Sheet: {sheet_name}",
                "level": 2,
                "text": section_text,
                "page_number": idx + 1,
            })
            doc.pages.append({
                "page_number": idx + 1,
                "text": section_text,
                "ocr_used": False,
                "ocr_confidence": None,
                "headings": [f"Sheet: {sheet_name}"],
            })
            text_blocks.append(section_text)

        doc.extracted_text = "\n\n".join(text_blocks).strip()

    def _parse_csv(self, path: Path, doc: NormalizedDocument) -> None:
        """Extract structured table and row-preserving text from CSV."""
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            raw_rows = [r for r in reader if any(cell.strip() for cell in r)]

        doc.parser_used = "csv"
        doc.page_count = 1
        if not raw_rows:
            return

        headers = [c.strip() for c in raw_rows[0]]
        rows = [[c.strip() for c in r] for r in raw_rows[1:]]
        md = _table_to_markdown(headers, rows, max_rows=100)
        doc.tables.append({
            "table_index": 0,
            "page_number": 1,
            "headers": headers,
            "rows": rows[:500],
            "row_count": len(rows),
            "markdown": md,
        })

        lines = [f"Dataset: {doc.filename} | Columns ({len(headers)}): {', '.join(headers)} | Rows: {len(rows)}"]
        for r in rows:
            pairs = [
                f"{headers[i] if i < len(headers) and headers[i] else f'Col_{i+1}'}: {val}"
                for i, val in enumerate(r)
                if val
            ]
            if pairs:
                lines.append(" | ".join(pairs))
        body = "\n".join(lines)
        doc.extracted_text = body
        doc.sections = [{
            "heading": f"CSV Table: {doc.filename}",
            "level": 1,
            "text": body,
            "page_number": 1,
        }]
        doc.pages = [{
            "page_number": 1,
            "text": body,
            "ocr_used": False,
            "ocr_confidence": None,
            "headings": [f"CSV Table: {doc.filename}"],
        }]

    def _parse_json(self, path: Path, doc: NormalizedDocument) -> None:
        """Parse JSON documents and tabular record arrays."""
        raw = path.read_text(encoding="utf-8", errors="replace")
        doc.parser_used = "json"
        doc.page_count = 1
        try:
            data = json.loads(raw)
        except Exception as exc:
            doc.errors.append(f"Invalid JSON syntax ({exc}); falling back to plain text")
            doc.extracted_text = raw.strip()
            doc.extraction_status = "partial" if doc.extracted_text else "error"
            return

        if isinstance(data, list) and data and all(isinstance(item, dict) for item in data[:50]):
            headers: list[str] = []
            for item in data[:100]:
                for k in item.keys():
                    if str(k) not in headers:
                        headers.append(str(k))
            rows = [[str(item.get(h, "")) for h in headers] for item in data]
            md = _table_to_markdown(headers, rows, max_rows=100)
            doc.tables.append({
                "table_index": 0,
                "page_number": 1,
                "headers": headers,
                "rows": rows[:500],
                "row_count": len(rows),
                "markdown": md,
            })
            lines = [f"JSON Records ({len(data)} items) | Fields: {', '.join(headers)}"]
            for item in data:
                pairs = [f"{k}: {v}" for k, v in item.items() if v is not None and str(v) != ""]
                if pairs:
                    lines.append(" | ".join(pairs))
            doc.extracted_text = "\n".join(lines)
        else:
            pretty = json.dumps(data, indent=2, ensure_ascii=False)
            doc.extracted_text = pretty

        doc.sections = [{
            "heading": f"JSON: {doc.filename}",
            "level": 1,
            "text": doc.extracted_text,
            "page_number": 1,
        }]
        doc.pages = [{
            "page_number": 1,
            "text": doc.extracted_text,
            "ocr_used": False,
            "ocr_confidence": None,
            "headings": [f"JSON: {doc.filename}"],
        }]

    def _parse_html(self, path: Path, doc: NormalizedDocument) -> None:
        """Extract headings, tables, and clean readable text from HTML."""
        raw = path.read_text(encoding="utf-8", errors="replace")
        doc.parser_used = "html"
        doc.page_count = 1

        # Extract HTML headings
        headings = [
            (int(m.group(1)), re.sub(r"<[^>]+>", "", m.group(2)).strip())
            for m in re.finditer(r"<h([1-6])[^>]*>(.*?)</h\1>", raw, flags=re.DOTALL | re.IGNORECASE)
        ]

        # Extract HTML tables
        for t_idx, tbl_match in enumerate(re.finditer(r"<table[^>]*>(.*?)</table>", raw, flags=re.DOTALL | re.IGNORECASE)):
            tbl_html = tbl_match.group(1)
            parsed_rows: list[list[str]] = []
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl_html, flags=re.DOTALL | re.IGNORECASE):
                cells = [
                    re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c)).strip()
                    for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.DOTALL | re.IGNORECASE)
                ]
                if any(cells):
                    parsed_rows.append(cells)
            if parsed_rows:
                headers = parsed_rows[0]
                rows = parsed_rows[1:] if len(parsed_rows) > 1 else []
                doc.tables.append({
                    "table_index": t_idx,
                    "page_number": 1,
                    "headers": headers,
                    "rows": rows[:500],
                    "row_count": len(rows),
                    "markdown": _table_to_markdown(headers, rows),
                })

        cleaned = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<style[^>]*>.*?</style>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"</?(?:p|div|h[1-6]|li|br|tr|blockquote|section|article)[^>]*>", "\n", cleaned, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", cleaned)
        text = (
            text.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
        )
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        doc.extracted_text = "\n".join(lines)
        doc.sections = [
            {"heading": h_txt, "level": lvl, "text": "", "page_number": 1}
            for lvl, h_txt in headings if h_txt
        ] or [{
            "heading": doc.filename,
            "level": 1,
            "text": doc.extracted_text,
            "page_number": 1,
        }]
        doc.pages = [{
            "page_number": 1,
            "text": doc.extracted_text,
            "ocr_used": False,
            "ocr_confidence": None,
            "headings": [h for _, h in headings if h],
        }]

    def _parse_text_or_markdown(self, path: Path, doc: NormalizedDocument) -> None:
        """Parse Markdown and plain-text documents preserving heading sections."""
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        ext = path.suffix.lower()
        doc.parser_used = "markdown" if ext in {".md", ".markdown"} else "text"
        doc.page_count = 1
        doc.extracted_text = text
        if not text:
            return
        sections = self._extract_markdown_sections(text)
        doc.sections = sections or [{
            "heading": doc.filename,
            "level": 1,
            "text": text,
            "page_number": 1,
        }]
        doc.pages = [{
            "page_number": 1,
            "text": text,
            "ocr_used": False,
            "ocr_confidence": None,
            "headings": [s["heading"] for s in doc.sections],
        }]

    def _parse_zip(self, path: Path, doc: NormalizedDocument) -> None:
        """Inspect and recursively parse supported files inside a ZIP archive."""
        doc.parser_used = "zip"
        if not zipfile.is_zipfile(path):
            doc.extraction_status = "error"
            doc.errors.append("Invalid or corrupted ZIP archive")
            return

        max_bytes = int(getattr(settings, "AEGIS_DOCUMENT_MAX_BYTES", 100 * 1024 * 1024))
        text_parts: list[str] = []
        page_idx = 0

        with zipfile.ZipFile(path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir() or member.filename.startswith("__MACOSX"):
                    continue
                if member.file_size <= 0 or member.file_size > max_bytes:
                    continue
                member_ext = Path(member.filename).suffix.lower()
                if member_ext == ".zip":
                    continue
                with tempfile.NamedTemporaryFile(suffix=member_ext, prefix="aegis_zip_", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    tmp.write(archive.read(member))
                try:
                    sub_doc = self.parse(tmp_path, filename=Path(member.filename).name)
                    if sub_doc.extracted_text.strip():
                        page_idx += 1
                        header = f"[Archive Member: {member.filename}]"
                        merged_txt = f"{header}\n{sub_doc.extracted_text}"
                        text_parts.append(merged_txt)
                        doc.sections.append({
                            "heading": member.filename,
                            "level": 2,
                            "text": sub_doc.extracted_text,
                            "page_number": page_idx,
                        })
                        doc.tables.extend(sub_doc.tables)
                finally:
                    tmp_path.unlink(missing_ok=True)

        doc.page_count = max(1, page_idx)
        doc.extracted_text = "\n\n".join(text_parts).strip()

    @staticmethod
    def _extract_markdown_sections(text: str) -> list[dict[str, Any]]:
        """Split Markdown-formatted text by `#` headings into structured sections."""
        sections: list[dict[str, Any]] = []
        current_heading = "Introduction"
        current_level = 1
        current_lines: list[str] = []

        for line in text.splitlines():
            m = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if m:
                if current_lines:
                    sections.append({
                        "heading": current_heading,
                        "level": current_level,
                        "text": "\n".join(current_lines).strip(),
                        "page_number": 1,
                    })
                    current_lines = []
                current_level = len(m.group(1))
                current_heading = m.group(2).strip()
            else:
                current_lines.append(line)

        if current_lines and "\n".join(current_lines).strip():
            sections.append({
                "heading": current_heading,
                "level": current_level,
                "text": "\n".join(current_lines).strip(),
                "page_number": 1,
            })
        return sections


_ROUTER: DocumentParserRouter | None = None
_ROUTER_LOCK = threading.Lock()


def get_parser_router() -> DocumentParserRouter:
    """Return the process-wide shared :class:`DocumentParserRouter`."""
    global _ROUTER
    if _ROUTER is not None:
        return _ROUTER
    with _ROUTER_LOCK:
        if _ROUTER is None:
            _ROUTER = DocumentParserRouter()
        return _ROUTER

