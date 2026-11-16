# backend/aegis/memory/qdrant.py
"""Qdrant vector-database client wrapper with resilience.

Falls back to in-memory Qdrant when the remote server is unreachable,
so the entire pipeline works without Docker.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from aegis.core.config import settings
from aegis.core.constants import EMBEDDING_VECTOR_SIZES, QDRANT_VECTOR_SIZE
from aegis.core.logging import get_logger

log = get_logger(__name__)

_ORIG_QDRANT_CLIENT = _QdrantClient
_memory_client: _QdrantClient | None = None
_file_client: _QdrantClient | None = None
_FILE_STORAGE_PATH: str | None = None
_remote_client: _QdrantClient | None = None
_remote_client_url: str | None = None
_remote_down_until: float = 0.0

_CANONICAL_BACKEND_DATA = Path(__file__).resolve().parents[2] / "data"


def _is_test_runtime() -> bool:
    return (
        os.environ.get("ENVIRONMENT", "").lower() == "test"
        or "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
    )


def _resolve_qdrant_storage_path(raw_path: str) -> str:
    cleaned = (raw_path or "").strip()
    if not cleaned or cleaned in ("./data/qdrant", "data/qdrant"):
        return str((_CANONICAL_BACKEND_DATA / "qdrant").resolve())
    p = Path(cleaned)
    if p.is_absolute():
        return str(p)
    return str((_CANONICAL_BACKEND_DATA.parent / p).resolve())


def _get_memory_client() -> _QdrantClient:
    global _memory_client
    if _memory_client is None:
        _memory_client = _QdrantClient(location=":memory:", timeout=5)
        log.info("Qdrant shared in-memory client initialized")
    return _memory_client


def _get_file_client(storage_path: str) -> _QdrantClient:
    """Get or create a singleton file-based Qdrant client."""
    global _file_client, _FILE_STORAGE_PATH
    resolved_path = _resolve_qdrant_storage_path(storage_path)
    if _file_client is not None and _FILE_STORAGE_PATH == resolved_path:
        return _file_client
    os.makedirs(resolved_path, exist_ok=True)
    _file_client = _QdrantClient(path=resolved_path, timeout=5)
    _FILE_STORAGE_PATH = resolved_path
    log.info("Qdrant shared file-based client initialized", path=resolved_path)
    return _file_client


class QdrantDB:
    """Thin wrapper around Qdrant client with persistent file-based & in-memory fallback."""

    def __init__(self, url: str = "", collection: str = "") -> None:
        global _remote_client, _remote_client_url, _remote_down_until

        explicit_url = bool(url)
        self._url = url or settings.AEGIS_QDRANT_URL
        self._collection = collection or settings.AEGIS_QDRANT_COLLECTION
        # The collection writes currently target. It differs from
        # ``_collection`` when a fallback embedder has a different vector size.
        self._active_collection = self._collection
        self._client: _QdrantClient | None = None

        self._use_memory = self._url in (
            ":memory:", "memory:", "inmemory:", "in-memory:",
        )

        is_patched_client = _QdrantClient is not _ORIG_QDRANT_CLIENT

        if self._use_memory:
            self._client = _get_memory_client()
            log.debug("Qdrant client initialized in-memory (explicit)")
        elif not self._url:
            if _is_test_runtime() and not is_patched_client:
                self._client = _get_memory_client()
            else:
                try:
                    self._client = _get_file_client(settings.AEGIS_QDRANT_STORAGE_PATH)
                except Exception as file_exc:
                    log.error(
                        "Failed to initialize file-based Qdrant storage: %s. Falling back to in-memory.",
                        str(file_exc)[:200],
                    )
                    self._client = _get_memory_client()
        elif (
            not explicit_url
            and not is_patched_client
            and _remote_client is not None
            and _remote_client_url == self._url
        ):
            self._client = _remote_client
        elif (
            not explicit_url
            and not is_patched_client
            and _remote_down_until > time.monotonic()
        ):
            if _is_test_runtime():
                self._client = _get_memory_client()
            else:
                try:
                    self._client = _get_file_client(settings.AEGIS_QDRANT_STORAGE_PATH)
                except Exception:
                    self._client = _get_memory_client()
        else:
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*Failed to obtain server version.*",
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=".*Client version .* is incompatible.*",
                    )
                    client = _QdrantClient(
                        url=self._url,
                        api_key=settings.AEGIS_QDRANT_API_KEY or None,
                        timeout=5,
                    )
                client.get_collections()
                self._client = client
                if not explicit_url and not is_patched_client:
                    _remote_client = client
                    _remote_client_url = self._url
                    _remote_down_until = 0.0
                log.info("Qdrant client connected to remote", url=self._url)
            except Exception as exc:
                if not explicit_url and not is_patched_client:
                    _remote_down_until = time.monotonic() + 30.0
                if _is_test_runtime() and not is_patched_client:
                    self._client = _get_memory_client()
                else:
                    log.warning(
                        "Qdrant remote unreachable (%s). Using persistent file-based storage at %s.",
                        str(exc)[:100],
                        _resolve_qdrant_storage_path(settings.AEGIS_QDRANT_STORAGE_PATH),
                    )
                    try:
                        self._client = _get_file_client(settings.AEGIS_QDRANT_STORAGE_PATH)
                    except Exception as file_exc:
                        log.error(
                            "Failed to initialize file-based Qdrant storage: %s. "
                            "Falling back to in-memory.",
                            str(file_exc)[:200],
                        )
                        self._client = _get_memory_client()

    @staticmethod
    def get_vector_size(provider: str) -> int:
        """Get the vector size for a given embedding provider."""
        return EMBEDDING_VECTOR_SIZES.get(provider, QDRANT_VECTOR_SIZE)

    def _collection_exists(self, name: str) -> bool:
        """Return True when *name* already exists in Qdrant."""
        if not self._client:
            return False
        try:
            self._client.get_collection(name)
            return True
        except Exception:
            return False

    def collection_for_dimensions(self, vector_size: int) -> str:
        """The collection that holds vectors of *vector_size*.

        The base collection is used when it has that dimension, or when
        nothing sized for it exists yet.  Otherwise the size-suffixed
        collection is used.  Reads and writes must agree on this, or a
        document is written to one collection and looked for in another.
        """
        if not self._client:
            return self._collection
        try:
            info = self._client.get_collection(self._collection)
            if info.config.params.vectors.size == vector_size:
                return self._collection
        except Exception:
            return self._collection
        suffixed = f"{self._collection}_{vector_size}"
        if self._collection_exists(suffixed):
            return suffixed
        return self._collection

    def ensure_collection(self, vector_size: int = QDRANT_VECTOR_SIZE) -> bool:
        """Create the collection if it does not exist. Returns True on success.

        A fallback embedder can have a different vector size from the primary
        one (Gemini is 3072, the local models are 384 / 1024 / 2048).  Rather
        than refusing to store anything — which silently dropped every chunk
        when the primary provider was rate-limited — a size-suffixed
        collection is used, e.g. ``aegisai_384``.  Retrieval searches the base
        collection *and* every suffixed one, so documents stored through a
        fallback are still found.

        Existing collections are **never** deleted or recreated: a size-suffixed
        collection that already holds points is reused as-is.  This prevents
        the data loss that ``recreate_collection`` caused whenever a
        suffixed collection was touched a second time.
        """
        if not self._client:
            return False
        try:
            collection_info = self._client.get_collection(self._collection)
            current_size = collection_info.config.params.vectors.size
            if current_size != vector_size:
                target = f"{self._collection}_{vector_size}"
                log.warning(
                    "Collection vector size mismatch — using a size-specific collection",
                    name=target,
                    old_size=current_size,
                    new_size=vector_size,
                )
                if not self._create_collection(target, vector_size):
                    return False
                self._active_collection = target
                return True
            self._active_collection = self._collection
            # Ensure payload indexes exist
            self._ensure_payload_indexes()
            return True
        except (UnexpectedResponse, RuntimeError, ValueError):
            # Base collection does not exist yet — create it.
            if not self._create_collection(self._collection, vector_size):
                return False
            self._active_collection = self._collection
            return True
        except Exception as e:
            log.error("Failed to check Qdrant collection", error=str(e))
            return False

    def _create_collection(self, name: str, vector_size: int) -> bool:
        """Create a collection only when it does not already exist.

        ``recreate_collection`` is intentionally *not* used: it deletes the
        existing collection and every point in it, which destroyed previously
        ingested vectors whenever a size-suffixed collection was (re)created.
        If the collection already exists with a compatible size it is reused;
        if it exists with an incompatible size it cannot be resized without
        data loss, so we report failure instead of destroying data.
        """
        try:
            if self._collection_exists(name):
                info = self._client.get_collection(name)
                existing_size = info.config.params.vectors.size
                if existing_size == vector_size:
                    self._ensure_payload_indexes(name)
                    log.debug("Reusing existing collection", name=name, vector_size=vector_size)
                    return True
                log.error(
                    "Collection exists with incompatible vector size",
                    name=name,
                    existing=existing_size,
                    requested=vector_size,
                )
                return False
            self._client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(
                    size=vector_size, distance=qm.Distance.COSINE,
                ),
            )
            log.info("Collection created", name=name, vector_size=vector_size)
            self._ensure_payload_indexes(name)
            return True
        except Exception as e:
            log.error("Failed to create Qdrant collection", name=name, error=str(e))
            return False

    def _searchable_collections(self, vector_size: int | None = None) -> list[str]:
        """The collections that can hold a vector of this size.

        The base collection plus any size-suffixed variant. When a vector
        size is given, only the collections whose dimension matches are
        returned — querying a collection with the wrong dimension is a
        server-side error, not an empty result.
        """
        names = [self._collection]
        if not self._client:
            return names
        try:
            for collection in self._client.get_collections().collections:
                name = getattr(collection, "name", "")
                if name.startswith(f"{self._collection}_") and name not in names:
                    names.append(name)
        except Exception:
            pass

        if vector_size is None:
            return names

        matching: list[str] = []
        for name in names:
            try:
                info = self._client.get_collection(name)
                size = info.config.params.vectors.size
            except Exception:
                continue
            if size == vector_size:
                matching.append(name)
        return matching

    def _ensure_payload_indexes(self, collection_name: str | None = None) -> None:
        """Create payload indexes for efficient filtering."""
        if not self._client:
            return
        target = collection_name or self._active_collection or self._collection
        index_fields = [
            ("doc_id", "keyword"),
            ("document_id", "keyword"),
            ("filename", "keyword"),
            ("section_heading", "keyword"),
            ("embedding_model", "keyword"),
            ("source_type", "keyword"),
            ("content_hash", "keyword"),
            ("type", "keyword"),
        ]
        for field_name, field_type in index_fields:
            try:
                self._client.create_payload_index(
                    collection_name=target,
                    field_name=field_name,
                    field_schema=field_type,
                )
                log.debug("Payload index created", field=field_name)
            except Exception:
                # Index may already exist
                pass

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_condition: qm.Filter | None = None,
    ) -> list[dict[str, Any]]:
        """Search for vectors with optional filter, failing gracefully.

        Every collection that could hold a vector of this size is searched —
        the base one and any size-suffixed variant — and the results are
        merged and re-sorted.
        """
        if not self._client or not query_vector:
            return []
        merged: list[dict[str, Any]] = []
        collections = self._searchable_collections(len(query_vector))
        if not collections:
            try:
                self._client.get_collections()
            except Exception as exc:
                log.error("Qdrant collection lookup failed", error=str(exc)[:200])
                raise RuntimeError("Qdrant collection lookup failed") from exc
            return []

        failures: list[Exception] = []
        for name in collections:
            try:
                response = self._client.query_points(
                    collection_name=name,
                    query=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=filter_condition,
                    with_payload=True,
                )
                points = response.points if hasattr(response, "points") else []
                merged.extend(
                    {
                        "id": str(p.id),
                        "score": p.score,
                        "payload": p.payload or {},
                    }
                    for p in points
                )
            except Exception as e:
                log.warning("Qdrant search failed for a collection", collection=name, error=str(e))
                failures.append(e)
        if failures:
            raise RuntimeError(
                f"Qdrant search failed for {len(failures)} of "
                f"{len(collections)} compatible collections"
            ) from failures[0]
        merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return merged[:limit]

    def scroll(
        self,
        *,
        scroll_filter: qm.Filter,
        limit: int = 10_000,
        with_payload: bool | list[str] = True,
        with_vectors: bool = False,
        vector_size: int | None = None,
    ) -> list[Any]:
        """Read matching points from compatible collections.

        Payload-only document views need not assume the active embedding
        dimension. When a size is supplied, only matching collections are
        scanned, which is useful for idempotent ingestion checks.
        """
        if not self._client or limit <= 0:
            return []

        points: list[Any] = []
        for name in self._searchable_collections(vector_size):
            offset = None
            try:
                while len(points) < limit:
                    batch, offset = self._client.scroll(
                        collection_name=name,
                        scroll_filter=scroll_filter,
                        limit=min(1000, limit - len(points)),
                        with_payload=with_payload,
                        with_vectors=with_vectors,
                        offset=offset,
                    )
                    points.extend(batch)
                    if offset is None:
                        break
            except Exception as exc:
                log.warning(
                    "Qdrant scroll failed for a collection",
                    collection=name,
                    error=str(exc)[:200],
                )
        return points[:limit]

    def delete_by_filter(self, filter_condition: qm.Filter) -> int:
        """Delete points matching a filter. Returns count of deleted points.

        Every collection that could hold the data is swept, including the
        size-suffixed ones used by fallback embedders.
        """
        if not self._client:
            return 0
        deleted = 0
        for name in self._searchable_collections():
            try:
                # Qdrant's delete() returns an UpdateResult without a deleted
                # count, so measure the match set before deleting.
                matching = self._client.count(
                    collection_name=name,
                    count_filter=filter_condition,
                    exact=True,
                ).count
                self._client.delete(
                    collection_name=name,
                    points_selector=qm.FilterSelector(filter=filter_condition),
                    wait=True,
                )
                deleted += int(matching)
            except Exception as e:
                log.warning("Qdrant delete by filter failed", collection=name, error=str(e))
        return deleted

    def delete_collection(self) -> None:
        """Remove the collection, failing gracefully."""
        if not self._client:
            return
        for name in self._searchable_collections():
            try:
                self._client.delete_collection(collection_name=name)
                log.warning("Collection deleted", name=name)
            except Exception as e:
                log.warning("Qdrant delete_collection failed", name=name, error=str(e))

    def upsert(self, points: list[qm.PointStruct]) -> bool:
        """Insert or update a batch of vectors; report whether Qdrant accepted it."""
        if not self._client:
            return False
        try:
            self._client.upsert(
                collection_name=self._active_collection, points=points, wait=True,
            )
            log.debug("Upserted vectors", count=len(points))
            return True
        except Exception as e:
            log.error("Qdrant upsert failed", error=str(e))
            return False

    def get_count(self, vector_size: int | None = None) -> int:
        """The number of points in the collection for this vector size."""
        if not self._client:
            return 0
        try:
            name = (
                self.collection_for_dimensions(vector_size)
                if vector_size
                else self._active_collection
            )
            return self._client.get_collection(name).points_count or 0
        except Exception:
            return 0
