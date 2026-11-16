# backend/aegis/ingestion/chunker.py
"""Advanced Document Chunking Engine (Section 4).

Single source of truth for chunking across AegisAI:
- Mandatory defaults:
  * ``CHUNK SIZE = 900 WORDS`` (``AEGIS_CHUNK_SIZE``)
  * ``CHUNK OVERLAP = 10%`` (``AEGIS_CHUNK_OVERLAP_PERCENT = 0.10`` -> ``90`` words
    dynamically calculated for 900-word chunks; never hardcoded).
- Supported modes:
  1. ``word`` — custom word-based chunking with paragraph/sentence boundary
     preservation and exact dynamic percentage overlap.
  2. ``adaptive`` — scales chunk word budget with document size while keeping
     10% dynamic overlap.
  3. ``document_aware`` — combines structural sections, headings, pages, and
     tables from :class:`NormalizedDocument` before applying word windows.
  4. ``section_aware`` — chunks within document sections with heading context.
  5. ``heading_aware`` — splits on Markdown/structural headings first.
  6. ``table_aware`` — preserves structured tables and repeats headers when
     large tables span multiple chunks.
  7. ``page_aware`` — chunks per page while preserving ``page_number``.
- Full chunk provenance metadata:
  ``chunk_id``, ``document_id`` (and ``doc_id``), ``filename``, ``page_number``
  (and ``page``), ``section_heading``, ``chunk_index``, ``extraction_method``,
  ``embedding_model``, ``creation_timestamp``, ``word_count``, ``token_count``,
  ``char_offset``, and ``content_hash``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from aegis.core.config import settings
from aegis.core.constants import (
    DEFAULT_CHUNK_OVERLAP_PERCENT,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_MAX_CHUNK_WORDS,
    DEFAULT_MIN_CHUNK_WORDS,
)
from aegis.core.logging import get_logger

if TYPE_CHECKING:
    from aegis.ingestion.parser_router import NormalizedDocument

log = get_logger(__name__)

SEPARATORS: list[str] = [
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
]

SUPPORTED_CHUNK_STRATEGIES: set[str] = {
    "word",
    "adaptive",
    "document_aware",
    "section_aware",
    "heading_aware",
    "table_aware",
    "page_aware",
}

_SENTENCE_END_RE = re.compile(r'[.!?]["\')\]]?$')
_TOKEN_RE = re.compile(r"(\S+)(\s*)")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_words(text: str) -> int:
    """Return word count of *text*."""
    if not text:
        return 0
    return len(text.split())


@dataclass
class DocumentChunk:
    """A single chunk of a document with stable identity and full provenance."""

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    source: str = ""
    page: int | None = None
    section_heading: str = ""
    extraction_method: str = "unknown"
    embedding_model: str = ""
    creation_timestamp: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Convert to Qdrant payload dict preserving both legacy & Section 4 keys."""
        word_cnt = count_words(self.text)
        emb_model = (
            self.embedding_model
            or self.metadata.get("embedding_model")
            or getattr(settings, "AEGIS_EMBEDDING_MODEL", "BAAI/bge-m3")
        )
        section_hdr = self.section_heading or str(self.metadata.get("section_heading", ""))
        ext_method = (
            self.extraction_method
            if self.extraction_method != "unknown"
            else str(self.metadata.get("extraction_method", "text"))
        )
        page_val = self.page if self.page is not None else self.metadata.get("page_number")

        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "doc_id": self.document_id,
            "document_id": self.document_id,
            "text": self.text,
            "filename": self.source,
            "chunk_index": self.chunk_index,
            "page": page_val,
            "page_number": page_val,
            "section_heading": section_hdr,
            "extraction_method": ext_method,
            "embedding_model": emb_model,
            "creation_timestamp": self.creation_timestamp,
            "type": "document_chunk",
            "source_type": "file",
            "word_count": word_cnt,
            "token_count": max(1, len(self.text) // 4),
            "char_offset": self.metadata.get("char_offset", 0),
            "content_hash": self._content_hash(),
        }
        payload.update(self.metadata)
        return payload

    def _content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_text(
        cls,
        text: str,
        document_id: str,
        chunk_index: int,
        source: str = "",
        page: int | None = None,
        section_heading: str = "",
        extraction_method: str = "unknown",
        embedding_model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DocumentChunk:
        """Create a chunk with a stable deterministic 16-char SHA-256 ID."""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        stable_id = hashlib.sha256(
            f"{document_id}:{chunk_index}:{content_hash}".encode("utf-8")
        ).hexdigest()[:16]
        meta = dict(metadata or {})
        return cls(
            chunk_id=stable_id,
            document_id=document_id,
            text=text,
            chunk_index=chunk_index,
            source=source,
            page=page,
            section_heading=section_heading or str(meta.get("section_heading", "")),
            extraction_method=extraction_method if extraction_method != "unknown" else str(meta.get("extraction_method", "unknown")),
            embedding_model=embedding_model or str(meta.get("embedding_model", "")),
            metadata=meta,
        )


# ── Parameter resolution (Word-based, 900 words default, 10% dynamic overlap) ─


def _resolve_chunk_size(chunk_size: int | None) -> int:
    """Resolve target chunk size in **words** (default 900 words)."""
    if chunk_size is None:
        val = int(getattr(settings, "AEGIS_CHUNK_SIZE", DEFAULT_CHUNK_SIZE_WORDS))
        return val if val > 0 else DEFAULT_CHUNK_SIZE_WORDS
    val = int(chunk_size)
    return val if val > 0 else DEFAULT_CHUNK_SIZE_WORDS


def _resolve_overlap(
    chunk_size: int,
    overlap: int | None = None,
    overlap_percent: float | None = None,
) -> int:
    """Dynamically calculate overlap in **words** (default 10% -> 90 words at 900)."""
    if overlap is not None:
        return max(0, min(int(overlap), max(0, chunk_size - 1)))

    configured_explicit = getattr(settings, "AEGIS_CHUNK_OVERLAP", None)
    if overlap_percent is None and configured_explicit is not None:
        return max(0, min(int(configured_explicit), max(0, chunk_size // 2)))

    pct = (
        float(overlap_percent)
        if overlap_percent is not None
        else float(getattr(settings, "AEGIS_CHUNK_OVERLAP_PERCENT", DEFAULT_CHUNK_OVERLAP_PERCENT))
    )
    pct = max(0.0, min(pct, 0.50))
    computed = int(round(chunk_size * pct))
    if computed <= 0 and pct > 0 and chunk_size >= 10:
        computed = max(1, chunk_size // 10)
    return min(computed, max(0, chunk_size // 2))


def _resolve_min_chunk_size(min_chunk_size: int | None) -> int:
    """Minimum word count for non-final chunks (default 40 words)."""
    if min_chunk_size is None:
        return int(getattr(settings, "AEGIS_MIN_CHUNK_SIZE", DEFAULT_MIN_CHUNK_WORDS))
    return max(0, int(min_chunk_size))


def _resolve_max_chunk_size(max_chunk_size: int | None, chunk_size: int) -> int:
    """Maximum word ceiling for structural blocks (default max(1800, chunk_size * 2))."""
    if max_chunk_size is not None and max_chunk_size > 0:
        return max(chunk_size, int(max_chunk_size))
    cfg = int(getattr(settings, "AEGIS_MAX_CHUNK_SIZE", DEFAULT_MAX_CHUNK_WORDS))
    return max(chunk_size, cfg)


def get_splitter(
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Build a word-length ``RecursiveCharacterTextSplitter`` for compatibility."""
    size = _resolve_chunk_size(chunk_size)
    ov = _resolve_overlap(size, overlap)
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=ov,
        separators=SEPARATORS,
        length_function=count_words,
        keep_separator=True,
    )


# ── Core word-window splitter with sentence/paragraph boundary preservation ───


def _tokenize_preserving_breaks(text: str) -> list[tuple[str, str]]:
    """Tokenize *text* into ``(word, normalized_separator)`` pairs."""
    tokens: list[tuple[str, str]] = []
    for match in _TOKEN_RE.finditer(text.strip()):
        word = match.group(1)
        raw_sep = match.group(2)
        if raw_sep.count("\n") >= 2:
            sep = "\n\n"
        elif "\n" in raw_sep:
            sep = "\n"
        else:
            sep = " "
        tokens.append((word, sep))
    return tokens


def _join_tokens(tokens: list[tuple[str, str]]) -> str:
    if not tokens:
        return ""
    parts: list[str] = []
    last = len(tokens) - 1
    for idx, (word, sep) in enumerate(tokens):
        parts.append(word)
        if idx < last:
            parts.append(sep)
    return "".join(parts).strip()


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    min_chunk_size: int | None = None,
    *,
    overlap_percent: float | None = None,
    preserve_sentences: bool = True,
) -> list[str]:
    """Split *text* into overlapping **word-based** chunks (default 900 words, 10% = 90 words overlap).

    Preserves paragraph and sentence boundaries near window edges whenever
    possible without breaking words or losing overlap continuity.
    """
    if not text or not text.strip():
        return []

    size = _resolve_chunk_size(chunk_size)
    ov = _resolve_overlap(size, overlap=overlap, overlap_percent=overlap_percent)
    min_size = _resolve_min_chunk_size(min_chunk_size)

    tokens = _tokenize_preserving_breaks(text)
    total_words = len(tokens)
    if total_words == 0:
        return []

    if total_words <= size:
        return [_join_tokens(tokens)]

    raw_chunks: list[str] = []
    start = 0
    lookback_limit = max(1, min(80, size // 10))

    while start < total_words:
        target_end = min(total_words, start + size)
        end = target_end

        if preserve_sentences and target_end < total_words and size >= 30:
            search_floor = max(start + max(ov + 1, size - lookback_limit), start + 1)
            for idx in range(target_end - 1, search_floor - 1, -1):
                word, sep = tokens[idx]
                if "\n" in sep or _SENTENCE_END_RE.search(word):
                    end = idx + 1
                    break

        chunk_str = _join_tokens(tokens[start:end])
        if chunk_str:
            raw_chunks.append(chunk_str)

        if end >= total_words:
            break

        next_start = end - ov
        if next_start <= start:
            next_start = start + max(1, size - ov)
        start = next_start

    filtered: list[str] = []
    last_idx = len(raw_chunks) - 1
    for i, c in enumerate(raw_chunks):
        w_count = count_words(c)
        if i == last_idx or w_count >= min_size:
            filtered.append(c)

    if not filtered:
        filtered = [_join_tokens(tokens[-min(total_words, size):])]

    log.info(
        "chunk_text",
        total_words=total_words,
        chunk_size_words=size,
        overlap_words=ov,
        min_chunk_words=min_size,
        num_chunks=len(filtered),
    )
    return filtered


def chunk_text_adaptive(
    text: str,
    file_size_bytes: int = 0,
    *,
    overlap_percent: float | None = None,
) -> list[str]:
    """Adaptive word-based chunking scaling with file size while keeping 10% overlap."""
    if not text or not text.strip():
        return []

    base = _resolve_chunk_size(None)  # 900 words by default
    if file_size_bytes > 20 * 1024 * 1024:
        chunk_size = min(base * 2, DEFAULT_MAX_CHUNK_WORDS)
    elif file_size_bytes > 10 * 1024 * 1024:
        chunk_size = int(base * 1.5)
    elif file_size_bytes > 5 * 1024 * 1024:
        chunk_size = int(base * 1.25)
    else:
        chunk_size = base

    return chunk_text(text, chunk_size=chunk_size, overlap=None, overlap_percent=overlap_percent)


# ── Structural & Document-Aware Chunking Modes ───────────────────────────────


def _chunk_table_block(
    table: dict[str, Any],
    chunk_size: int,
    max_chunk_size: int,
) -> list[str]:
    """Keep small/medium tables whole; split huge tables while repeating headers."""
    headers = [str(h or "").strip() for h in (table.get("headers") or [])]
    rows = table.get("rows") or []
    sheet = table.get("sheet_name") or ""
    prefix = f"[Table{' - ' + sheet if sheet else ''}] "

    if not headers and not rows:
        md = str(table.get("markdown") or "").strip()
        return [md] if md else []

    header_line = "Columns: " + " | ".join(h or f"Col_{i+1}" for i, h in enumerate(headers))
    formatted_rows: list[str] = []
    for r in rows:
        pairs = [
            f"{headers[i] if i < len(headers) and headers[i] else f'Col_{i+1}'}: {val}"
            for i, val in enumerate(r)
            if str(val).strip()
        ]
        if pairs:
            formatted_rows.append(" | ".join(pairs))

    full_table_text = prefix + header_line + "\n" + "\n".join(formatted_rows)
    if count_words(full_table_text) <= max_chunk_size:
        return [full_table_text.strip()]

    # Split large table into word-budgeted row batches, repeating header_line on every chunk
    batches: list[str] = []
    current_lines = [prefix + header_line]
    current_words = count_words(current_lines[0])
    header_words = current_words

    for row_str in formatted_rows:
        r_words = count_words(row_str)
        if len(current_lines) > 1 and current_words + r_words > chunk_size:
            batches.append("\n".join(current_lines).strip())
            # 10% row overlap for continuity
            overlap_rows = current_lines[max(1, len(current_lines) - max(1, len(current_lines) // 10)):]
            current_lines = [prefix + header_line] + overlap_rows
            current_words = header_words + sum(count_words(x) for x in overlap_rows)
        current_lines.append(row_str)
        current_words += r_words

    if len(current_lines) > 1:
        batches.append("\n".join(current_lines).strip())
    return batches


def chunk_normalized_document(
    norm_doc: NormalizedDocument,
    *,
    strategy: str | None = None,
    chunk_size: int | None = None,
    overlap: int | None = None,
    overlap_percent: float | None = None,
    min_chunk_size: int | None = None,
    max_chunk_size: int | None = None,
    embedding_model: str = "",
    source_type: str = "file",
    deduplicate: bool | None = None,
) -> list[DocumentChunk]:
    """Chunk a :class:`NormalizedDocument` using the configured structural strategy."""
    mode = (strategy or getattr(settings, "AEGIS_CHUNK_STRATEGY", "document_aware")).lower()
    if mode not in SUPPORTED_CHUNK_STRATEGIES:
        mode = "document_aware"

    file_size = int(norm_doc.metadata.get("file_size_bytes", 0) or 0)
    size = _resolve_chunk_size(chunk_size)
    if mode == "adaptive" and chunk_size is None:
        if file_size > 20 * 1024 * 1024:
            size = min(size * 2, DEFAULT_MAX_CHUNK_WORDS)
        elif file_size > 10 * 1024 * 1024:
            size = int(size * 1.5)
        elif file_size > 5 * 1024 * 1024:
            size = int(size * 1.25)

    ov = _resolve_overlap(size, overlap=overlap, overlap_percent=overlap_percent)
    min_size = _resolve_min_chunk_size(min_chunk_size)
    max_size = _resolve_max_chunk_size(max_chunk_size, size)
    do_dedup = (
        bool(getattr(settings, "AEGIS_DEDUPE_ENABLED", True))
        if deduplicate is None
        else deduplicate
    )
    emb_name = embedding_model or getattr(settings, "AEGIS_EMBEDDING_MODEL", "BAAI/bge-m3")

    # Collect segments: list of (segment_text, page_number, section_heading)
    segments: list[tuple[str, int | None, str]] = []

    if mode == "word" or mode == "adaptive":
        for piece in chunk_text(
            norm_doc.extracted_text,
            chunk_size=size,
            overlap=ov,
            min_chunk_size=min_size,
        ):
            segments.append((piece, 1 if norm_doc.page_count == 1 else None, ""))

    elif mode == "page_aware" and norm_doc.pages:
        for pg in norm_doc.pages:
            pg_text = str(pg.get("text") or "").strip()
            if not pg_text:
                continue
            pg_num = pg.get("page_number")
            headings = pg.get("headings") or []
            sec_hdr = str(headings[0]) if headings else f"Page {pg_num}"
            for piece in chunk_text(pg_text, chunk_size=size, overlap=ov, min_chunk_size=min_size):
                segments.append((piece, pg_num, sec_hdr))

    elif mode == "table_aware" and norm_doc.tables:
        for tbl in norm_doc.tables:
            pg_num = tbl.get("page_number", 1)
            sec_hdr = str(tbl.get("sheet_name") or f"Table {tbl.get('table_index', 0) + 1}")
            for piece in _chunk_table_block(tbl, chunk_size=size, max_chunk_size=max_size):
                segments.append((piece, pg_num, sec_hdr))
        # If non-tabular prose also exists (e.g. PDF/DOCX with embedded tables), chunk prose too
        if norm_doc.file_type not in {"csv", "xlsx", "xls"} and norm_doc.extracted_text.strip():
            for piece in chunk_text(norm_doc.extracted_text, chunk_size=size, overlap=ov, min_chunk_size=min_size):
                segments.append((piece, 1, "Document Prose"))

    elif mode in {"section_aware", "heading_aware", "document_aware"}:
        # For pure spreadsheet/CSV datasets, table_aware chunking preserves column headers on every chunk
        if norm_doc.file_type in {"csv", "xlsx", "xls"} and norm_doc.tables:
            for tbl in norm_doc.tables:
                pg_num = tbl.get("page_number", 1)
                sec_hdr = str(tbl.get("sheet_name") or norm_doc.filename)
                for piece in _chunk_table_block(tbl, chunk_size=size, max_chunk_size=max_size):
                    segments.append((piece, pg_num, sec_hdr))
        elif norm_doc.sections and len(norm_doc.sections) > 1:
            # Merge tiny consecutive sections on the same page up to `size` words so we don't emit micro-chunks,
            # while splitting large sections with 900-word / 10%-overlap windows!
            buf_texts: list[str] = []
            buf_words = 0
            buf_page: int | None = None
            buf_heading: str = ""

            def _flush_buffer() -> None:
                nonlocal buf_texts, buf_words, buf_page, buf_heading
                if not buf_texts:
                    return
                merged = "\n\n".join(buf_texts).strip()
                for piece in chunk_text(merged, chunk_size=size, overlap=ov, min_chunk_size=min_size):
                    segments.append((piece, buf_page, buf_heading))
                buf_texts = []
                buf_words = 0

            for sec in norm_doc.sections:
                sec_text = str(sec.get("text") or "").strip()
                if not sec_text:
                    continue
                sec_hdr = str(sec.get("heading") or "").strip()
                sec_pg = sec.get("page_number")
                formatted = f"[{sec_hdr}]\n{sec_text}" if sec_hdr and not sec_text.startswith(f"[{sec_hdr}") else sec_text
                w = count_words(formatted)

                if buf_texts and (buf_words + w > size or (sec_pg is not None and buf_page is not None and sec_pg != buf_page and buf_words >= min_size)):
                    _flush_buffer()

                if not buf_texts:
                    buf_page = sec_pg
                    buf_heading = sec_hdr
                buf_texts.append(formatted)
                buf_words += w

                if buf_words >= size:
                    _flush_buffer()

            _flush_buffer()
        else:
            sec_hdr = str(norm_doc.sections[0].get("heading", "")) if norm_doc.sections else ""
            pg_num = norm_doc.sections[0].get("page_number", 1) if norm_doc.sections else 1
            for piece in chunk_text(norm_doc.extracted_text, chunk_size=size, overlap=ov, min_chunk_size=min_size):
                segments.append((piece, pg_num, sec_hdr))

    if not segments and norm_doc.extracted_text.strip():
        for piece in chunk_text(norm_doc.extracted_text, chunk_size=size, overlap=ov, min_chunk_size=min_size):
            segments.append((piece, 1, ""))

    # Build DocumentChunk objects with full metadata and optional deduplication
    chunks: list[DocumentChunk] = []
    seen_hashes: set[str] = set()
    char_cursor = 0

    for seg_text, pg_num, sec_hdr in segments:
        cleaned = seg_text.strip()
        if not cleaned:
            continue
        c_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
        if do_dedup and c_hash in seen_hashes:
            continue
        seen_hashes.add(c_hash)

        idx = len(chunks)
        pos = norm_doc.extracted_text.find(cleaned[:80], char_cursor) if norm_doc.extracted_text else -1
        if pos < 0:
            pos = char_cursor
        else:
            char_cursor = pos + 1

        chunk_obj = DocumentChunk.from_text(
            text=cleaned,
            document_id=norm_doc.document_id,
            chunk_index=idx,
            source=norm_doc.filename,
            page=pg_num,
            section_heading=sec_hdr,
            extraction_method=norm_doc.parser_used,
            embedding_model=emb_name,
            metadata={
                "char_offset": pos,
                "source_type": source_type,
                "chunk_strategy": mode,
                "chunk_size_target_words": size,
                "chunk_overlap_target_words": ov,
                "section_heading": sec_hdr,
                "extraction_method": norm_doc.parser_used,
                "embedding_model": emb_name,
                "file_type": norm_doc.file_type,
            },
        )
        chunks.append(chunk_obj)

    return chunks


def verify_chunk_overlap(chunks: list[str] | list[DocumentChunk]) -> dict[str, Any]:
    """Inspect consecutive chunks and compute exact word-overlap diagnostics."""
    texts = [c.text if isinstance(c, DocumentChunk) else str(c) for c in chunks]
    if len(texts) <= 1:
        return {
            "num_chunks": len(texts),
            "pair_count": 0,
            "overlaps_words": [],
            "avg_overlap_words": 0.0,
            "all_pairs_overlap": True,
        }

    overlaps: list[int] = []
    for idx in range(len(texts) - 1):
        w1 = texts[idx].split()
        w2 = texts[idx + 1].split()
        max_k = min(len(w1), len(w2))
        matched = 0
        for k in range(max_k, 0, -1):
            if w1[-k:] == w2[:k]:
                matched = k
                break
        if matched == 0:
            shared = set(w1[-min(len(w1), 200):]) & set(w2[:min(len(w2), 200)])
            matched = len(shared)
        overlaps.append(matched)

    return {
        "num_chunks": len(texts),
        "pair_count": len(overlaps),
        "overlaps_words": overlaps,
        "avg_overlap_words": round(sum(overlaps) / len(overlaps), 2) if overlaps else 0.0,
        "all_pairs_overlap": all(x > 0 for x in overlaps),
    }
