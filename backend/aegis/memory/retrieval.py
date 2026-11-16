# backend/aegis/memory/retrieval.py
"""High-level Qdrant Vector & Hybrid Retrieval Engine (Section 6).

Single reusable interface for:
- Embedding queries with strict dimension & finiteness validation
- Single-document & multi-document filtered Qdrant vector search
- Hybrid dense + BM25 lexical Reciprocal Rank Fusion (RRF)
- Deduplication, contextual header injection, reranking, and context compression
- Diagnostics tracking (latency, unique document coverage, score distribution)
"""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from typing import Any

from aegis.core.constants import QDRANT_RETRIEVAL_LIMIT
from aegis.core.logging import get_logger
from aegis.memory.embeddings import EmbeddingModel, get_embedding_model
from aegis.memory.qdrant import QdrantDB
from aegis.retrieval_enhancer import (
    QueryEnhancer,
    assemble_compressed_context,
    deduplicate_chunks,
    hybrid_rrf_rerank,
    inject_contextual_headers,
    rerank_chunks,
)
from aegis.schemas.agent import RetrievedChunk

log = get_logger(__name__)


class RetrievalService:
    """End-to-end retrieval: embed query -> search Qdrant -> hybrid rank -> typed chunks."""

    def __init__(
        self,
        embedder: EmbeddingModel | None = None,
        vector_db: QdrantDB | None = None,
        llm: Any | None = None,
    ) -> None:
        self._embedder = embedder or get_embedding_model()
        self._db = vector_db or QdrantDB()
        self._query_enhancer = QueryEnhancer(llm=llm)
        self.last_diagnostics: dict[str, Any] = {}

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = QDRANT_RETRIEVAL_LIMIT,
        score_threshold: float = 0.25,
        doc_id: str | None = None,
        document_ids: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        enable_hybrid: bool = False,
        enable_hyde: bool = False,
        enable_multi_query: bool = False,
        enable_dedup: bool = True,
        enable_contextual_headers: bool = False,
        enable_rerank: bool = False,
        enable_corrective: bool = False,
        include_memory_entries: bool = False,
    ) -> list[RetrievedChunk]:
        """Embed *query* and return the top-k relevant chunks from Qdrant."""
        from qdrant_client.http import models as qm

        start = time.time()
        log.info(
            "RetrievalService.retrieve",
            query_preview=query[:80],
            top_k=top_k,
            doc_id=doc_id,
            document_ids=document_ids,
            hybrid=enable_hybrid,
            hyde=enable_hyde,
            multi_query=enable_multi_query,
            rerank=enable_rerank,
            corrective=enable_corrective,
        )

        # 1. Build query variants
        queries = [query]
        if enable_multi_query:
            queries = self._query_enhancer.rewrite(query)
        hyde_query = None
        if enable_hyde:
            hyde_query = self._query_enhancer.hyde(query)

        # 2. Embed all query variants & validate finiteness/dimensions
        all_vectors: list[list[float]] = []
        for q in queries:
            vec = self._embedder.embed(q)
            if vec:
                if not all(math.isfinite(value) for value in vec):
                    raise ValueError("Query embedding contains a non-finite value")
                all_vectors.append(vec)
        if hyde_query and hyde_query != query:
            vec = self._embedder.embed(hyde_query)
            if vec:
                if not all(math.isfinite(value) for value in vec):
                    raise ValueError("Query embedding contains a non-finite value")
                all_vectors.append(vec)
        if not all_vectors:
            log.warning("RetrievalService: empty embeddings for all query variants")
            raise RuntimeError("Embedding provider returned no query vectors")

        expected_size = len(all_vectors[0])
        if any(len(vector) != expected_size for vector in all_vectors):
            raise ValueError("Query embeddings have inconsistent dimensions")

        # 3. Ensure collection exists for the active vector dimension
        ok = self._db.ensure_collection(vector_size=expected_size)
        if not ok:
            raise RuntimeError(
                "Qdrant collection is unavailable for the query embedding dimensions"
            )

        # 4. Build metadata / single-doc / multi-doc filter
        must_conditions: list[Any] = []
        target_doc_ids = [d for d in (document_ids or []) if d]
        if doc_id and doc_id not in target_doc_ids:
            target_doc_ids.append(doc_id)

        if len(target_doc_ids) == 1:
            must_conditions.append(
                qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=target_doc_ids[0]))
            )
        elif len(target_doc_ids) > 1:
            must_conditions.append(
                qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=target_doc_ids))
            )

        if metadata_filter:
            for m_key, m_val in metadata_filter.items():
                if m_val is not None:
                    must_conditions.append(
                        qm.FieldCondition(key=str(m_key), match=qm.MatchValue(value=m_val))
                    )

        filter_condition = qm.Filter(must=must_conditions) if must_conditions else None

        # 5. Search Qdrant across query vectors
        seen_ids: set[str] = set()
        merged: list[dict[str, Any]] = []
        candidate_limit = max(top_k * 3, 20)
        try:
            for vector in all_vectors:
                raw_results = self._db.search(
                    query_vector=vector,
                    limit=candidate_limit,
                    score_threshold=score_threshold,
                    filter_condition=filter_condition,
                )
                for r in raw_results:
                    payload = r.get("payload") or {}
                    if not include_memory_entries and payload.get("type") == "memory_entry":
                        continue
                    point_id = str(r.get("id", ""))
                    if point_id and point_id not in seen_ids:
                        seen_ids.add(point_id)
                        merged.append(r)
        except Exception as exc:
            log.error("RetrievalService: Qdrant search failed", error=str(exc)[:200])
            raise RuntimeError("Qdrant search failed") from exc

        # If threshold was too strict on explicit document-scoped queries, retry once with 0.05 floor
        if not merged and filter_condition is not None and score_threshold > 0.05:
            try:
                raw_results = self._db.search(
                    query_vector=all_vectors[0],
                    limit=candidate_limit,
                    score_threshold=0.05,
                    filter_condition=filter_condition,
                )
                for r in raw_results:
                    payload = r.get("payload") or {}
                    if not include_memory_entries and payload.get("type") == "memory_entry":
                        continue
                    point_id = str(r.get("id", ""))
                    if point_id and point_id not in seen_ids:
                        seen_ids.add(point_id)
                        merged.append(r)
            except Exception:
                pass

        if not merged:
            self.last_diagnostics = {
                "retrieved_count": 0,
                "unique_documents": 0,
                "avg_score": 0.0,
                "top_score": 0.0,
                "latency_ms": int((time.time() - start) * 1000),
                "embedding_model": getattr(self._embedder, "model_name", ""),
                "vector_dimensions": expected_size,
            }
            return []

        merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        merged = merged[:candidate_limit]

        # 6. Corrective retrieval
        if enable_corrective and merged:
            top_score = merged[0].get("score", 0.0)
            if top_score < score_threshold:
                reason = f"top_score={top_score:.3f} below threshold={score_threshold:.3f}"
                reformulated = self._query_enhancer.reformulate(query, reason=reason)
                if reformulated and reformulated.lower() != query.lower():
                    try:
                        reformulated_vec = self._embedder.embed(reformulated)
                        if reformulated_vec:
                            retry_results = self._db.search(
                                query_vector=reformulated_vec,
                                limit=candidate_limit,
                                score_threshold=min(score_threshold, 0.15),
                                filter_condition=filter_condition,
                            )
                            for r in retry_results:
                                payload = r.get("payload") or {}
                                if not include_memory_entries and payload.get("type") == "memory_entry":
                                    continue
                                point_id = str(r.get("id", ""))
                                if point_id and point_id not in seen_ids:
                                    seen_ids.add(point_id)
                                    merged.append(r)
                            merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)
                    except Exception as exc:
                        log.warning("Corrective retrieval failed", error=str(exc)[:180])

        # 7. Convert to typed RetrievedChunk objects preserving full provenance
        chunks: list[RetrievedChunk] = []
        for r in merged:
            payload = r.get("payload", {}) or {}
            chunk = RetrievedChunk(
                chunk_id=str(payload.get("chunk_id", r.get("id", ""))),
                document_id=str(payload.get("doc_id", payload.get("document_id", ""))),
                text=str(payload.get("text", "")),
                score=float(r.get("score", 0.0)),
                source=str(payload.get("filename", payload.get("source", "unknown"))),
                page=payload.get("page", payload.get("page_number")),
                chunk_index=int(payload.get("chunk_index", 0)),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k not in {
                        "chunk_id", "doc_id", "text", "filename", "source",
                        "page", "chunk_index", "type", "source_type",
                        "token_count", "char_offset", "content_hash",
                    }
                },
            )
            chunks.append(chunk)

        # 8. Hybrid RRF & optional LLM reranking
        if enable_rerank:
            chunks = rerank_chunks(query, chunks, llm=self._query_enhancer._llm, top_k=top_k * 2)
        elif enable_hybrid:
            chunks = hybrid_rrf_rerank(query, chunks, top_k=top_k * 2)

        # 9. Post-retrieval deduplication & header enrichment
        if enable_dedup:
            chunks = deduplicate_chunks(chunks, max_results=top_k)
        else:
            chunks = chunks[:top_k]

        if enable_contextual_headers:
            chunks = inject_contextual_headers(chunks)

        elapsed = int((time.time() - start) * 1000)
        unique_docs = len({c.document_id or c.source for c in chunks})
        avg_score = round(sum(c.score for c in chunks) / len(chunks), 4) if chunks else 0.0
        self.last_diagnostics = {
            "retrieved_count": len(chunks),
            "unique_documents": unique_docs,
            "avg_score": avg_score,
            "top_score": round(chunks[0].score, 4) if chunks else 0.0,
            "latency_ms": elapsed,
            "embedding_model": getattr(self._embedder, "model_name", ""),
            "vector_dimensions": expected_size,
            "hybrid_enabled": enable_hybrid or enable_rerank,
        }
        log.info(
            "RetrievalService: retrieved",
            count=len(chunks),
            unique_docs=unique_docs,
            top_score=chunks[0].score if chunks else 0.0,
            latency_ms=elapsed,
        )
        return chunks

    def assemble_context(self, chunks: list[RetrievedChunk], *, max_words: int = 3600) -> str:
        """Compress & assemble retrieved chunks into citation-tagged prompt context."""
        return assemble_compressed_context(chunks, max_words=max_words)

    def store(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Embed *text* and store it in Qdrant with model metadata."""
        from qdrant_client.http import models as qm

        vector = self._embedder.embed(text)
        if not vector:
            raise RuntimeError("Embedding provider returned an empty vector")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding contains a non-finite value")

        vector_size = len(vector)
        if not self._db.ensure_collection(vector_size=vector_size):
            raise RuntimeError(
                f"Qdrant collection is unavailable for {vector_size}-dimensional vectors"
            )
        payload = dict(metadata or {})
        if "chunk_index" in payload or "text" in payload:
            payload.setdefault("embedding_model", getattr(self._embedder, "model_name", ""))
            payload.setdefault("embedding_dimensions", vector_size)

        point_id = int(time.time() * 1000000) & ((1 << 63) - 1)
        point = qm.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )
        if not self._db.upsert([point]):
            raise RuntimeError("Qdrant failed to store the vector")

    def store_batch(
        self,
        texts: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> int:
        """Embed and store multiple texts in batch with strict dimension validation."""
        from qdrant_client.http import models as qm

        if not texts:
            return 0
        if metadata_list is not None and len(metadata_list) != len(texts):
            raise ValueError(
                f"Received {len(metadata_list)} metadata records for {len(texts)} texts"
            )

        vectors = self._embedder.embed_batch(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding provider returned {len(vectors)} vectors for {len(texts)} texts"
            )
        if not vectors or not vectors[0]:
            raise RuntimeError("Embedding provider returned an empty vector")

        vector_size = len(vectors[0])
        for index, vector in enumerate(vectors):
            if len(vector) != vector_size:
                raise ValueError(
                    f"Embedding {index} has {len(vector)} dimensions; expected {vector_size}"
                )
            if not all(math.isfinite(value) for value in vector):
                raise ValueError(f"Embedding {index} contains a non-finite value")

        emb_model_name = getattr(self._embedder, "model_name", "")
        points: list[qm.PointStruct] = []
        for i, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
            meta = dict(metadata_list[i]) if metadata_list and i < len(metadata_list) else {}
            if "chunk_index" in meta or "text" in meta:
                meta.setdefault("embedding_model", emb_model_name)
                meta.setdefault("embedding_dimensions", vector_size)
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            stable_key = (
                f"{meta.get('doc_id', '')}:"
                f"{meta.get('chunk_id') or meta.get('chunk_index', i)}:"
                f"{content_hash}"
            )
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=meta,
                )
            )

        if not points:
            return 0

        ok = self._db.ensure_collection(vector_size=vector_size)
        if not ok:
            raise RuntimeError(
                f"Qdrant collection is unavailable for {vector_size}-dimensional vectors"
            )

        if not self._db.upsert(points):
            raise RuntimeError("Qdrant failed to upsert the embedded batch")
        return len(points)

    def get_collection_stats(self) -> dict[str, Any]:
        """Return Qdrant collection statistics."""
        try:
            count = self._db.get_count()
            collection_info = self._db._client.get_collection(self._db._collection)
            return {
                "collection": self._db._collection,
                "points_count": count,
                "vector_size": collection_info.config.params.vectors.size,
                "embedding_model": getattr(self._embedder, "model_name", ""),
                "status": "ok",
            }
        except Exception as exc:
            log.error("RetrievalService: failed to get stats", error=str(exc)[:200])
            return {"collection": self._db._collection, "status": "error", "error": str(exc)}


class RetrievalPipeline(RetrievalService):
    """Backward-compatible wrapper around RetrievalService."""
    pass
