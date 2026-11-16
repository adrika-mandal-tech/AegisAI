# backend/aegis/agents/memory_manager.py
"""Memory Manager agent — stores and retrieves interaction context.

Uses Qdrant (via the ``memory/`` package) for persistent vector storage
of conversation history and completed workflow reports.
Falls back to in-memory dict when Qdrant is unavailable.
"""

from __future__ import annotations

from typing import Any

from aegis.core.logging import get_logger
from aegis.memory.retrieval import RetrievalService

log = get_logger(__name__)


class MemoryManager:
    """Manages short-term and long-term memory for the agent workflow."""

    def __init__(self, retrieval: RetrievalService | None = None) -> None:
        self._retrieval = retrieval or RetrievalService()
        self._store: dict[str, list[dict[str, Any]]] = {}

    def store(self, key: str, data: dict[str, Any]) -> None:
        """Persist *data* under *key* in both memory and Qdrant."""
        self._store.setdefault(key, []).append(data)
        log.debug("MemoryManager.store (memory)", key=key)

        try:
            text = str(data)
            metadata = {"key": key, "type": "memory_entry"}
            self._retrieval.store(text, metadata=metadata)
            log.debug("MemoryManager.store (qdrant)", key=key)
        except Exception:
            log.warning("Failed to persist to Qdrant", exc_info=True)

    def retrieve(self, key: str) -> list[dict[str, Any]]:
        """Return all entries stored under *key*."""
        return self._store.get(key, [])

    def clear(self, key: str) -> None:
        """Remove all entries for *key*."""
        self._store.pop(key, None)
