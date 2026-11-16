# backend/aegis/tools/multi_loader.py
"""Multi-format document loader — supports PDF, DOCX, XLS, XLSX, TXT, CSV, MD, ZIP, URLs, and Git repos.

All loaders return a list of text chunks suitable for embedding and storage in Qdrant.
"""

from __future__ import annotations

import csv
import os
import re
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse
import zipfile

from aegis.core.config import settings
from aegis.core.logging import get_logger
from aegis.ingestion.chunker import chunk_text as chunk_documents
from aegis.ingestion.chunker import chunk_text_adaptive as chunk_adaptive
from aegis.tools.document_loader import get_document_loader

log = get_logger(__name__)

# ── Supported extensions and their loaders ───────────────────────────────

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".pdf": "pdf_docx",
    ".docx": "pdf_docx",
    ".doc": "pdf_docx",
    ".pptx": "pdf_docx",
    ".xls": "excel",
    ".xlsx": "excel",
    ".txt": "text",
    ".md": "text",
    ".html": "text",
    ".htm": "text",
    ".json": "json",
    ".csv": "csv",
    ".zip": "zip",
    # Image formats — processed via OCR
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".tif": "image",
    ".bmp": "image",
    ".webp": "image",
    ".gif": "image",
}

SUPPORTED_MIME_PREFIXES: dict[str, str] = {
    "application/pdf": "pdf_docx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "pdf_docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pdf_docx",
    "application/json": "json",
    "text/html": "text",
    "text/plain": "text",
    "text/markdown": "text",
    "text/csv": "csv",
    "application/zip": "zip",
    "application/x-zip-compressed": "zip",
}

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB per file
MAX_CHUNK_SIZE = 900  # words per chunk (default 900 words, 10% = 90 words overlap)

REPO_CODE_EXTENSIONS: set[str] = {
    ".py",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
    ".toml",
    ".rst",
    ".sql",
    ".sh",
    ".go",
    ".rs",
    ".java",
    ".cpp",
    ".c",
    ".h",
}


def _chunk_size_for(file_size_bytes: int) -> int:
    """The chunk size in words for a file of the given size (default 900 words)."""
    base = int(getattr(settings, "AEGIS_CHUNK_SIZE", MAX_CHUNK_SIZE))
    if file_size_bytes > 20 * 1024 * 1024:
        return min(base * 2, 1800)
    if file_size_bytes > 10 * 1024 * 1024:
        return int(base * 1.5)
    if file_size_bytes > 5 * 1024 * 1024:
        return int(base * 1.25)
    return base


def classify_source(filename: str, mime_type: str | None = None) -> str:
    """Classify a file into a loader category."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]
    if mime_type:
        for prefix, category in SUPPORTED_MIME_PREFIXES.items():
            if mime_type.startswith(prefix):
                return category
    return "unknown"


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks.

    Delegates to the shared chunker so there is exactly one chunking
    implementation (previously this module, ``document_loader`` and
    ``ingestion.chunker`` each had their own, and they disagreed).
    """
    return chunk_documents(text, chunk_size=chunk_size, overlap=overlap)


def chunk_text_for_size(text: str, file_size_bytes: int = 0) -> list[str]:
    """Adaptive chunking: larger files get larger chunks to reduce count."""
    return chunk_adaptive(text, file_size_bytes=file_size_bytes)


# ── Individual file loaders ─────────────────────────────────────────────


def load_image_file(path: str, file_size_bytes: int = 0) -> list[str]:
    """Load an image file via OCR (PaddleOCR fallback in DocumentLoader)."""
    loader = get_document_loader()
    text = loader.load(path)
    if not text or not text.strip():
        log.warning("No text extracted from image via OCR", path=path)
        return []
    return chunk_text_for_size(text, file_size_bytes=file_size_bytes)


def load_pdf_docx(path: str, file_size_bytes: int = 0) -> list[str]:
    """Load PDF or DOCX via Docling."""
    chunk_size = _chunk_size_for(file_size_bytes)
    loader = get_document_loader()
    chunks = loader.load_and_chunk(path, chunk_size=chunk_size)
    if not chunks:
        log.warning("No text extracted from PDF/DOCX", path=path)
    return chunks


def load_text_file(path: str, file_size_bytes: int = 0) -> list[str]:
    """Load a plain text or markdown file."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if not text.strip():
        log.warning("Empty text file", path=path)
        return []
    return chunk_text_for_size(text, file_size_bytes=file_size_bytes)


def load_csv_file(path: str, file_size_bytes: int = 0) -> list[str]:
    """Load a CSV file — each row becomes a chunk with headers preserved."""
    chunks: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    headers = rows[0]
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        pairs = [f"{h}: {v}" for h, v in zip(headers, row) if v.strip()]
        if pairs:
            chunks.append(" | ".join(pairs))

    return chunks if chunks else []


# ── Excel loader ─────────────────────────────────────────────────────────


def load_excel_file(path: str) -> list[str]:
    """Load an XLS or XLSX file — each sheet/row becomes a chunk."""
    ext = os.path.splitext(path)[1].lower()
    chunks: list[str] = []

    if ext == ".xls":
        try:
            import xlrd
            log.info("Loading .xls Excel file via xlrd", path=path)
            book = xlrd.open_workbook(path)
            for sheet_idx in range(book.nsheets):
                sheet = book.sheet_by_index(sheet_idx)
                if sheet.nrows == 0:
                    continue
                headers = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]
                for r in range(1, sheet.nrows):
                    row_vals = [sheet.cell_value(r, c) for c in range(sheet.ncols)]
                    pairs = [
                        f"{headers[c]}: {str(row_vals[c]).strip()}"
                        for c in range(min(len(headers), len(row_vals)))
                        if str(row_vals[c]).strip()
                    ]
                    if pairs:
                        chunk = f"[Sheet: {sheet.name}] " + " | ".join(pairs)
                        chunks.append(chunk)
                if sheet.nrows <= 1:
                    headers_str = ", ".join(h for h in headers if h)
                    if headers_str:
                        chunks.append(f"[Sheet: {sheet.name}] Columns: {headers_str}")
            log.info(".xls loaded", path=path, sheets=book.nsheets, chunks=len(chunks))
            return chunks
        except Exception as exc:
            log.warning("Failed to open .xls file via xlrd", path=path, error=str(exc))

    try:
        import openpyxl
    except ImportError:
        log.warning("openpyxl not installed — cannot load Excel files. Run: pip install openpyxl")
        return chunks

    log.info("Loading Excel file via openpyxl", path=path)
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as e:
        log.warning("Failed to open Excel file", path=path, error=str(e))
        return chunks

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        # First non-empty row as "headers"
        headers = []
        for cell in rows[0]:
            headers.append(str(cell) if cell is not None else "")

        for row in rows[1:]:
            if not any(cell is not None for cell in row):
                continue
            pairs = [
                f"{headers[i]}: {str(cell)}"
                for i, cell in enumerate(row)
                if cell is not None and str(cell).strip() and i < len(headers)
            ]
            if pairs:
                chunk = f"[Sheet: {sheet_name}] " + " | ".join(pairs)
                chunks.append(chunk)

        # If no data rows, still capture sheet name
        if len(rows) <= 1:
            headers_str = ", ".join(h for h in headers if h)
            if headers_str:
                chunks.append(f"[Sheet: {sheet_name}] Columns: {headers_str}")

    wb.close()
    log.info("Excel loaded", path=path, sheets=len(wb.sheetnames), chunks=len(chunks))
    return chunks


def load_zip_file(path: str, file_size_bytes: int = 0) -> list[str]:
    """Load supported files from a ZIP archive and index their extracted text.

    Each member is loaded through the loader for its own type, so each one is
    chunked with the sizing that suits it.
    """
    if not zipfile.is_zipfile(path):
        log.warning("Invalid ZIP file", path=path)
        return []

    all_chunks: list[str] = []
    try:
        with zipfile.ZipFile(path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                if member.file_size > MAX_FILE_SIZE:
                    log.warning("Skipping oversized ZIP entry", file=member.filename)
                    continue

                ext = os.path.splitext(member.filename)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                try:
                    with archive.open(member) as fh:
                        tmp_name = f"{uuid.uuid4()}{ext}"
                        tmp_path = Path("/tmp") / tmp_name
                        tmp_path.write_bytes(fh.read())
                        chunks = load_single_file(str(tmp_path), member.filename, file_size_bytes=member.file_size)
                        if chunks:
                            all_chunks.extend(chunks)
                        try:
                            tmp_path.unlink()
                        except Exception:
                            pass
                except Exception:
                    log.warning("Failed to load ZIP entry", file=member.filename)
    except Exception as e:
        log.warning("ZIP extraction failed", path=path, error=str(e))
        return []

    if not all_chunks:
        log.warning("No text extracted from ZIP", path=path)

    return all_chunks


# ── URL loader ──────────────────────────────────────────────────────────


def load_url(source_url: str) -> list[str]:
    """Fetch an HTTP/HTTPS URL (HTML, Markdown, JSON, or PDF) and extract text chunks safely."""
    import ssl
    import urllib.error
    import urllib.request

    cleaned_url = source_url.strip()
    parsed = urlparse(cleaned_url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        log.warning("Rejected non-HTTP(S) URL", url=source_url)
        return []

    # Automatically convert GitHub file viewer URLs (/blob/) to raw file URLs
    if parsed.netloc.lower() == "github.com" and "/blob/" in parsed.path:
        raw_path = parsed.path.replace("/blob/", "/", 1)
        cleaned_url = f"https://raw.githubusercontent.com{raw_path}"

    log.info("Loading URL", url=cleaned_url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain,text/markdown,*/*;q=0.9",
    }

    raw_bytes = b""
    content_type = ""
    try:
        req = urllib.request.Request(cleaned_url, headers=headers)
        with urllib.request.urlopen(req, timeout=25) as resp:
            content_type = str(resp.headers.get("Content-Type", "") or "").lower()
            raw_bytes = resp.read(MAX_FILE_SIZE)
    except urllib.error.HTTPError as e:
        log.warning("URL fetch HTTP error", url=cleaned_url, code=e.code)
        return []
    except Exception as e:
        # Retry once with relaxed SSL verification only on local macOS cert store issues
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            try:
                ctx = ssl._create_unverified_context()
                req = urllib.request.Request(cleaned_url, headers=headers)
                with urllib.request.urlopen(req, timeout=25, context=ctx) as resp:
                    content_type = str(resp.headers.get("Content-Type", "") or "").lower()
                    raw_bytes = resp.read(MAX_FILE_SIZE)
            except Exception as e2:
                log.warning("URL SSL retry error", url=cleaned_url, error=str(e2))
                return []
        else:
            log.warning("URL fetch error", url=cleaned_url, error=str(e))
            return []

    if not raw_bytes:
        return []

    # Direct PDF URL support
    if "application/pdf" in content_type or parsed.path.lower().endswith(".pdf") or raw_bytes[:4] == b"%PDF":
        tmp_pdf = Path(tempfile.gettempdir()) / f"url_{uuid.uuid4().hex}.pdf"
        try:
            tmp_pdf.write_bytes(raw_bytes)
            return load_pdf_docx(str(tmp_pdf), file_size_bytes=len(raw_bytes))
        finally:
            tmp_pdf.unlink(missing_ok=True)

    raw_text = raw_bytes.decode("utf-8", errors="replace")
    if "html" in content_type or "<html" in raw_text[:1000].lower() or "<body" in raw_text[:1000].lower():
        text = _html_to_text(raw_text)
    else:
        text = raw_text

    if not text.strip():
        log.warning("No text extracted from URL", url=cleaned_url)
        return []

    return chunk_text(text)


def _html_to_text(html: str) -> str:
    """Clean HTML-to-text extraction using BeautifulSoup when available, with regex fallback."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "aside"]):
            tag.decompose()
        extracted = soup.get_text(separator="\n")
        lines = [line.strip() for line in extracted.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        if cleaned:
            return cleaned
    except Exception:
        pass

    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"</?(?:p|div|h[1-6]|li|br|tr|blockquote)[^>]*>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", html)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


# ── Git repo loader ─────────────────────────────────────────────────────


def load_git_repo(repo_url: str) -> list[str]:
    """Clone a public git repo (all directories) and extract text from supported docs & code."""
    import subprocess

    log.info("Loading git repo", url=repo_url)

    clean_url = _sanitize_repo_url(repo_url)
    if not clean_url:
        log.warning("Invalid git repo URL", url=repo_url)
        return []

    ignored_dirs = {
        ".git",
        "node_modules",
        ".next",
        "dist",
        "build",
        ".venv",
        "venv",
        "__pycache__",
        "vendor",
        "target",
    }

    with tempfile.TemporaryDirectory(prefix="aegis_repo_") as tmpdir:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", clean_url, tmpdir],
                capture_output=True,
                timeout=180,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            log.warning("Git clone failed", url=repo_url, clean_url=clean_url, error=e.stderr.decode()[:500])
            return []
        except Exception as e:
            log.warning("Git clone error", url=repo_url, clean_url=clean_url, error=str(e))
            return []

        all_chunks: list[str] = []
        files_processed = 0
        for root, dirs, files in os.walk(tmpdir):
            dirs[:] = [d for d in dirs if d not in ignored_dirs and not d.startswith(".")]
            for fname in sorted(files):
                if files_processed >= 250:
                    break
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS and ext not in REPO_CODE_EXTENSIONS:
                    continue
                try:
                    fsize = os.path.getsize(fpath)
                except OSError:
                    continue
                if fsize == 0 or fsize > MAX_FILE_SIZE:
                    continue
                rel_path = os.path.relpath(fpath, tmpdir)
                try:
                    if ext in REPO_CODE_EXTENSIONS:
                        file_chunks = load_text_file(fpath, file_size_bytes=fsize)
                    else:
                        file_chunks = load_single_file(fpath, fname, file_size_bytes=fsize)
                    if file_chunks:
                        all_chunks.extend(f"[File: {rel_path}]\n{c}" for c in file_chunks)
                        files_processed += 1
                except Exception:
                    log.warning("Failed to load repo file", file=rel_path)

        if not all_chunks:
            log.warning("No text extracted from repo", url=repo_url)

        return all_chunks


def _sanitize_repo_url(repo_url: str) -> str | None:
    """Normalize a repo URL for git clone by removing tracking params and /tree/<branch> suffixes."""
    try:
        parsed = urlparse((repo_url or "").strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        path = parsed.path.rstrip("/")
        if parsed.netloc.lower() in {"github.com", "gitlab.com"}:
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 4 and parts[2] in {"tree", "blob", "-"}:
                path = "/" + "/".join(parts[:2])
        cleaned = parsed._replace(path=path or parsed.path, query="", fragment="")
        return urlunparse(cleaned)
    except Exception:
        return None


# ── Dispatch ────────────────────────────────────────────────────────────


def load_single_file(
    path: str,
    filename: str,
    mime_type: str | None = None,
    file_size_bytes: int = 0,
) -> list[str]:
    """Load a single file by auto-detecting its type."""
    category = classify_source(filename, mime_type)
    if category == "pdf_docx":
        return load_pdf_docx(path, file_size_bytes=file_size_bytes)
    elif category == "text":
        return load_text_file(path, file_size_bytes=file_size_bytes)
    elif category == "json":
        loader = get_document_loader()
        return loader.load_and_chunk(path)
    elif category == "csv":
        return load_csv_file(path, file_size_bytes=file_size_bytes)
    elif category == "excel":
        return load_excel_file(path)
    elif category == "zip":
        return load_zip_file(path, file_size_bytes=file_size_bytes)
    elif category == "image":
        return load_image_file(path, file_size_bytes=file_size_bytes)
    else:
        log.warning("Unsupported file type", filename=filename)
        return []


def load_from_source(source: str, source_type: str = "file") -> list[str]:
    """Load content from a file path, URL, or git repo URL.

    *source_type* can be ``"file"``, ``"url"``, or ``"repo"``.
    """
    if source_type == "url":
        return load_url(source)
    elif source_type == "repo":
        return load_git_repo(source)
    else:
        # File — auto-detect based on extension
        if not os.path.isfile(source):
            log.warning("File not found", path=source)
            return []
        return load_single_file(source, os.path.basename(source))
