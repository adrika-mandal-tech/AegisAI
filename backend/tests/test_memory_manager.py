from unittest.mock import MagicMock

from aegis.agents.memory_manager import MemoryManager


class TestMemoryManager:
    def test_store_and_retrieve(self, mock_retrieval: MagicMock):
        mm = MemoryManager(retrieval=mock_retrieval)
        mm.store("key1", {"msg": "hello"})
        entries = mm.retrieve("key1")
        assert len(entries) == 1
        assert entries[0]["msg"] == "hello"

    def test_retrieve_nonexistent(self, mock_retrieval: MagicMock):
        mm = MemoryManager(retrieval=mock_retrieval)
        entries = mm.retrieve("nonexistent")
        assert entries == []

    def test_clear(self, mock_retrieval: MagicMock):
        mm = MemoryManager(retrieval=mock_retrieval)
        mm.store("key1", {"msg": "hello"})
        mm.clear("key1")
        entries = mm.retrieve("key1")
        assert entries == []

    def test_multiple_store(self, mock_retrieval: MagicMock):
        mm = MemoryManager(retrieval=mock_retrieval)
        mm.store("key1", {"i": 1})
        mm.store("key1", {"i": 2})
        assert len(mm.retrieve("key1")) == 2

    def test_store_calls_qdrant(self, mock_retrieval: MagicMock):
        mm = MemoryManager(retrieval=mock_retrieval)
        mm.store("key1", {"msg": "hello"})
        assert mock_retrieval.store.called
