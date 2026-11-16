# backend/aegis/api/rag.py
"""Minimal RAG API layer: ``POST /upload``, ``POST /query``, ``GET /health``.

Thin wrappers around the *existing* working pipeline — no component is
replaced or reimplemented:

    upload → aegis.ingestion.pipeline.ingest_document_async
             (parse → chunk → embed → Qdrant)
    query  → aegis.memory.retrieval.RetrievalService.retrieve
             (embed query → Qdrant similarity search)
           → aegis.llm.factory.get_default_llm().generate
             (grounded answer from retrieved context)
    health → service liveness probe

All credentials are read from ``.env`` / environment variables via
``aegis.core.config.Settings`` (pydantic-settings). No API keys are
hardcoded here or anywhere in the package (``aegis.core.config`` is the
single source of truth).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from aegis.api.routes import (
    SUPPORTED_EXTENSIONS,
    _document_registry,
    stable_document_id,
    stage_upload_file,
    update_document_ingestion,
)
from aegis.api.schemas import QueryRequest, QueryResponse, QuerySource, UploadResponse
from aegis.core.config import settings
from aegis.core.logging import get_logger
from aegis.ingestion.pipeline import ingest_document_async
from aegis.llm.factory import get_default_llm
from aegis.memory.retrieval import RetrievalService
from aegis.schemas.agent import RetrievedChunk

log = get_logger(__name__)
router = APIRouter()

_SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer the user's question "
    "using ONLY the context passages below. Cite passages inline as [1], "
    "[2], ... matching the passage numbers. If the context does not contain "
    "the answer, say so explicitly instead of guessing."
)


# ── Lazy singletons (constructed on first use; tests can monkeypatch) ──────

_retrieval_service: RetrievalService | None = None
_llm: Any = None


def _get_retrieval() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service


def _get_llm() -> Any:
    global _llm
    if _llm is None:
        _llm = get_default_llm()
    return _llm


def _to_source(chunk: RetrievedChunk) -> QuerySource:
    return QuerySource(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        source=chunk.source,
        score=chunk.score,
        snippet=chunk.text[:300],
        page=chunk.page,
        chunk_index=chunk.chunk_index,
    )


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe for the RAG service."""
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a document and run it through the existing ingestion pipeline.

    parse → chunk → embed → store in Qdrant. The document is also recorded
    in the shared registry so it appears in ``GET /api/documents``.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    temp_path = Path("/tmp") / f"{uuid.uuid4()}{ext}"
    try:
        file_size, content_hash = await stage_upload_file(file, temp_path)
        doc_id = stable_document_id(file.filename, content_hash)
        _document_registry[doc_id] = {
            "id": doc_id,
            "filename": file.filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "status": "processing",
            "chunks": 0,
            "error": "",
            "source_type": "file",
            "file_size_bytes": file_size,
            "content_hash": content_hash,
        }

        result = await ingest_document_async(
            source=temp_path,
            document_id=doc_id,
            filename=file.filename,
            source_type="file",
            file_size_bytes=file_size,
        )

        update_document_ingestion(doc_id, result)
        error_text = _document_registry[doc_id]["error"]

        if result.status != "ready":
            return UploadResponse(
                success=False,
                document_id=doc_id,
                filename=file.filename,
                status=result.status,
                error=error_text or "Ingestion failed",
            )

        return UploadResponse(
            success=True,
            document_id=doc_id,
            filename=file.filename,
            chunks=result.stored_chunks,
            deduped=result.deduped_chunks,
            status=result.status,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("upload failed")
        return UploadResponse(
            success=False,
            filename=file.filename or "",
            error=str(exc)[:300],
        )
    finally:
        await file.close()
        await asyncio.to_thread(temp_path.unlink, missing_ok=True)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Classic RAG query: retrieve context from Qdrant, then answer with LLM.

    Both the vector search and the LLM call run on worker threads so the
    event loop stays responsive (established convention in this codebase).
    """
    log.info("rag_query", query=request.query[:80], top_k=request.top_k, doc_id=request.doc_id)

    # 1. Retrieve — embeds the query and searches Qdrant.
    try:
        chunks = await asyncio.to_thread(
            _get_retrieval().retrieve,
            request.query,
            top_k=request.top_k,
            doc_id=request.doc_id,
        )
    except Exception as exc:
        log.exception("rag_query: retrieval failed")
        return QueryResponse(success=False, error=f"Retrieval failed: {str(exc)[:200]}")

    # Drop empty-text chunks (legacy data) — they carry no grounding value.
    chunks = [c for c in chunks if c.text.strip()]

    if not chunks:
        return QueryResponse(
            success=True,
            answer="No relevant documents were found for this query.",
            chunks_retrieved=0,
        )

    # 2. Build grounded prompt from retrieved context.
    context = "\n\n".join(
        f"[{i}] {c.source or c.document_id or 'unknown'} "
        f"(page={c.page if c.page is not None else 'n/a'}, score={c.score:.3f})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )
    prompt = f"Context passages:\n{context}\n\nQuestion: {request.query}"

    # 3. Generate the answer with the configured LLM.
    try:
        answer = await asyncio.to_thread(
            _get_llm().generate,
            prompt,
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.2,
        )
    except Exception as exc:
        log.exception("rag_query: LLM generation failed")
        return QueryResponse(
            success=False,
            sources=[_to_source(c) for c in chunks],
            chunks_retrieved=len(chunks),
            error=f"LLM generation failed: {str(exc)[:200]}",
        )

    return QueryResponse(
        success=True,
        answer=answer,
        sources=[_to_source(c) for c in chunks],
        chunks_retrieved=len(chunks),
    )