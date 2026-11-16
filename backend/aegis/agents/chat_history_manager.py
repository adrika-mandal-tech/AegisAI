# backend/aegis/agents/chat_history_manager.py
"""Chat history manager for persistent conversation and session storage."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.core.config import settings
from aegis.core.logging import get_logger

log = get_logger(__name__)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatHistoryManager:
    """Manages session-isolated chat history with optional JSON disk persistence.

    When ``storage_path`` is omitted (e.g. unit tests), operates cleanly in-memory.
    The global singleton uses ``data/chat_history.json`` so conversations survive
    backend restarts.
    """

    def __init__(self, storage_path: str | Path | None = None):
        self._lock = threading.RLock()
        self._conversations: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}
        self._storage_path: Path | None = Path(storage_path) if storage_path else None
        if self._storage_path is not None:
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
            self._conversations = dict(payload.get("conversations", {}))
            self._messages = {
                k: list(v) for k, v in (payload.get("messages", {}) or {}).items()
            }
        except Exception as exc:
            log.warning("ChatHistoryManager: failed to load persisted history", error=str(exc))

    def _save_to_disk(self) -> None:
        if self._storage_path is None:
            return
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "conversations": self._conversations,
                "messages": self._messages,
                "saved_at": _utc_iso(),
            }
            self._storage_path.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("ChatHistoryManager: failed to save persisted history", error=str(exc))

    def create_conversation(
        self, title: str = "", metadata: dict[str, Any] | None = None
    ) -> str:
        """Create a new conversation and return its ID."""
        with self._lock:
            conversation_id = str(uuid.uuid4())[:8]
            now = _utc_iso()
            conversation = {
                "id": conversation_id,
                "title": title or f"Conversation {len(self._conversations) + 1}",
                "created_at": now,
                "updated_at": now,
                "metadata": metadata or {},
                "message_count": 0,
            }
            self._conversations[conversation_id] = conversation
            self._messages[conversation_id] = []
            self._save_to_disk()
            log.info("ChatHistoryManager.create_conversation", conversation_id=conversation_id)
            return conversation_id

    def ensure_session(self, session_id: str, title: str = "") -> str:
        """Ensure a session/conversation exists for session isolation."""
        with self._lock:
            if session_id in self._conversations:
                return session_id
            now = _utc_iso()
            self._conversations[session_id] = {
                "id": session_id,
                "title": title or f"Session {session_id}",
                "created_at": now,
                "updated_at": now,
                "metadata": {"session_id": session_id},
                "message_count": 0,
            }
            self._messages[session_id] = []
            self._save_to_disk()
            return session_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Add a message to a conversation."""
        with self._lock:
            if conversation_id not in self._conversations:
                log.warning(
                    "ChatHistoryManager.add_message: conversation not found",
                    conversation_id=conversation_id,
                )
                return None

            now = _utc_iso()
            message = {
                "id": str(uuid.uuid4())[:8],
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "timestamp": now,
                "metadata": metadata or {},
            }

            self._messages[conversation_id].append(message)
            self._conversations[conversation_id]["updated_at"] = now
            self._conversations[conversation_id]["message_count"] += 1
            self._save_to_disk()

            log.info(
                "ChatHistoryManager.add_message",
                conversation_id=conversation_id,
                message_id=message["id"],
            )
            return message

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Get conversation details."""
        with self._lock:
            return self._conversations.get(conversation_id)

    def get_messages(
        self,
        conversation_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get messages from a conversation with pagination."""
        with self._lock:
            if conversation_id not in self._messages:
                return []
            messages = self._messages[conversation_id]
            start = offset
            end = offset + limit if limit > 0 else len(messages)
            return list(messages[start:end])

    def list_conversations(
        self,
        limit: int = 20,
        offset: int = 0,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List conversations with optional session filter and pagination."""
        with self._lock:
            conversations = list(self._conversations.values())
            if session_id:
                conversations = [
                    c
                    for c in conversations
                    if c.get("id") == session_id
                    or (c.get("metadata") or {}).get("session_id") == session_id
                ]
            conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            start = offset
            end = offset + limit if limit > 0 else len(conversations)
            return conversations[start:end]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its messages."""
        with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                self._messages.pop(conversation_id, None)
                self._save_to_disk()
                log.info("ChatHistoryManager.delete_conversation", conversation_id=conversation_id)
                return True
            return False

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update conversation title."""
        with self._lock:
            if conversation_id in self._conversations:
                self._conversations[conversation_id]["title"] = title
                self._conversations[conversation_id]["updated_at"] = _utc_iso()
                self._save_to_disk()
                return True
            return False

    def search_messages(
        self,
        query: str,
        conversation_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search messages by content."""
        with self._lock:
            query_lower = query.lower()
            results: list[dict[str, Any]] = []
            search_convos = [conversation_id] if conversation_id else list(self._messages.keys())

            for convo_id in search_convos:
                if convo_id not in self._messages:
                    continue
                for message in self._messages[convo_id]:
                    if query_lower in message.get("content", "").lower():
                        results.append(message)
                        if len(results) >= limit:
                            return results
            return results

    def get_conversation_summary(self, conversation_id: str) -> dict[str, Any] | None:
        """Get a summary of a conversation."""
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                return None
            messages = self._messages.get(conversation_id, [])
            return {
                "id": conversation["id"],
                "title": conversation["title"],
                "created_at": conversation["created_at"],
                "updated_at": conversation["updated_at"],
                "message_count": len(messages),
                "first_message": messages[0] if messages else None,
                "last_message": messages[-1] if messages else None,
            }

    def export_conversation(self, conversation_id: str) -> str | None:
        """Export conversation as JSON string."""
        with self._lock:
            if conversation_id not in self._conversations:
                return None
            conversation = self._conversations[conversation_id]
            messages = self._messages.get(conversation_id, [])
            export_data = {
                "conversation": conversation,
                "messages": messages,
                "exported_at": _utc_iso(),
            }
            return json.dumps(export_data, indent=2)

    def import_conversation(self, json_data: str) -> str | None:
        """Import conversation from JSON string."""
        with self._lock:
            try:
                data = json.loads(json_data)
                conversation = data.get("conversation", {})
                messages = data.get("messages", [])
                conversation_id = conversation.get("id", str(uuid.uuid4())[:8])
                self._conversations[conversation_id] = conversation
                self._messages[conversation_id] = messages
                self._save_to_disk()
                log.info("ChatHistoryManager.import_conversation", conversation_id=conversation_id)
                return conversation_id
            except Exception as exc:
                log.error("ChatHistoryManager.import_conversation failed", error=str(exc))
                return None

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stored conversations."""
        with self._lock:
            total_messages = sum(len(msgs) for msgs in self._messages.values())
            return {
                "total_conversations": len(self._conversations),
                "total_messages": total_messages,
                "avg_messages_per_conversation": (
                    total_messages / len(self._conversations) if self._conversations else 0
                ),
            }


_default_storage = Path(settings.AEGIS_DOCUMENT_ROOT).parent / "chat_history.json"
_chat_history_manager = ChatHistoryManager(storage_path=_default_storage)


def get_chat_history_manager() -> ChatHistoryManager:
    """Get the global persistent chat history manager instance."""
    return _chat_history_manager
