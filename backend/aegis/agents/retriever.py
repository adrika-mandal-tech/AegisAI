# backend/aegis/agents/retriever.py
"""Retriever agent — single- & multi-document Qdrant hybrid search with provenance (Section 7.4)."""

from __future__ import annotations

import inspect
import uuid
from datetime import datetime, timezone

from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider
from aegis.llm.factory import get_default_llm
from aegis.memory.retrieval import RetrievalService
from aegis.schemas.agent import Evidence, ResearchResult

log = get_logger(__name__)

_SHARED_RETRIEVAL_SERVICE: RetrievalService | None = None


def _get_shared_retrieval() -> RetrievalService:
    global _SHARED_RETRIEVAL_SERVICE
    if _SHARED_RETRIEVAL_SERVICE is None:
        _SHARED_RETRIEVAL_SERVICE = RetrievalService()
    return _SHARED_RETRIEVAL_SERVICE


class Retriever:
    """Queries Qdrant vector store with hybrid RRF ranking and full chunk provenance."""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        retrieval: RetrievalService | None = None,
    ) -> None:
        self._llm = llm or get_default_llm()
        self._retrieval = retrieval or _get_shared_retrieval()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        doc_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> ResearchResult:
        """Search Qdrant across one or multiple documents for *query*."""
        log.info(
            "Retriever.retrieve",
            query_preview=(query or "")[:80],
            top_k=top_k,
            doc_id=doc_id,
            document_ids=document_ids,
        )
        try:
            return self._retrieve_impl(query, top_k=top_k, doc_id=doc_id, document_ids=document_ids)
        except Exception as exc:
            log.warning("Retriever fallback to empty result due to error", error=str(exc)[:200])
            return ResearchResult(
                query=query,
                content="",
                source="vector_store",
                relevance_score=0.0,
            )

    def _retrieve_impl(
        self,
        query: str,
        *,
        top_k: int = 10,
        doc_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> ResearchResult:
        retrieve_fn = self._retrieval.retrieve
        kwargs: dict[str, object] = {
            "top_k": top_k,
            "doc_id": doc_id,
            "enable_dedup": True,
            "enable_contextual_headers": False,
        }
        # Inspect signature so custom/mocked RetrievalService objects remain 100% compatible
        try:
            sig = inspect.signature(retrieve_fn)
            params = sig.parameters
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if accepts_var_kw or "enable_hybrid" in params:
                kwargs["enable_hybrid"] = True
            if (accepts_var_kw or "document_ids" in params) and document_ids:
                kwargs["document_ids"] = document_ids
        except Exception:
            pass

        chunks = retrieve_fn(query, **kwargs)  # type: ignore[arg-type]

        if not chunks:
            log.info("Retriever: no document results from Qdrant")
            return ResearchResult(
                query=query,
                content="",
                source="vector_store",
                relevance_score=0.0,
            )

        content_parts: list[str] = []
        evidence_items: list[Evidence] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for chunk in chunks:
            text = chunk.text or ""
            score = float(chunk.score or 0.0)
            source = chunk.source or "unknown"
            meta = getattr(chunk, "metadata", {}) or {}
            sec_heading = getattr(chunk, "section_heading", "") or str(meta.get("section_heading", ""))
            ext_method = getattr(chunk, "extraction_method", "") or str(meta.get("extraction_method", ""))
            emb_model = getattr(chunk, "embedding_model", "") or str(meta.get("embedding_model", ""))

            source_metadata = {
                "page": chunk.page,
                "page_number": chunk.page,
                "section_heading": sec_heading,
                "chunk_index": chunk.chunk_index,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "extraction_method": ext_method,
                "embedding_model": emb_model,
                "score": score,
            }
            loc_tag = f"source={source}"
            if chunk.page is not None:
                loc_tag += f" p.{chunk.page}"
            if sec_heading:
                loc_tag += f" §{sec_heading}"
            content_parts.append(f"[{loc_tag} | score={score:.3f}] {text}")

            claim_text = text[:220] if text else f"Relevant passage for: {query[:100]}"
            evidence = Evidence(
                id=chunk.chunk_id or str(uuid.uuid4())[:8],
                claim=claim_text,
                source=source,
                supporting_passage=text[:700],
                source_metadata=source_metadata,
                confidence=min(max(score, 0.0) * 1.15, 1.0),
                relevance=min(max(score, 0.0) * 1.15, 1.0),
                timestamp=now_iso,
            )
            evidence_items.append(evidence)

        combined = "\n\n".join(content_parts)
        return ResearchResult(
            query=query,
            content=combined,
            source="qdrant_vector_store",
            relevance_score=float(chunks[0].score) if chunks else 0.0,
            evidence_items=[e.model_dump() for e in evidence_items],
        )
