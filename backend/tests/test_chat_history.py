"""Test chat history manager."""

from __future__ import annotations

import pytest

from aegis.agents.chat_history_manager import ChatHistoryManager, get_chat_history_manager


class TestChatHistoryManager:
    """Test chat history management functionality."""

    def setup_method(self):
        self.manager = ChatHistoryManager()

    def test_create_conversation(self):
        """Test creating a new conversation."""
        conversation_id = self.manager.create_conversation("Test Conversation")
        
        assert conversation_id is not None
        assert len(conversation_id) > 0
        
        conversation = self.manager.get_conversation(conversation_id)
        assert conversation is not None
        assert conversation["title"] == "Test Conversation"
        assert conversation["message_count"] == 0

    def test_create_conversation_with_metadata(self):
        """Test creating conversation with metadata."""
        metadata = {"user_id": "123", "session_id": "abc"}
        conversation_id = self.manager.create_conversation("Test", metadata=metadata)
        
        conversation = self.manager.get_conversation(conversation_id)
        assert conversation["metadata"] == metadata

    def test_add_message(self):
        """Test adding messages to conversation."""
        conversation_id = self.manager.create_conversation()
        
        message = self.manager.add_message(conversation_id, "user", "Hello")
        assert message is not None
        assert message["role"] == "user"
        assert message["content"] == "Hello"
        
        conversation = self.manager.get_conversation(conversation_id)
        assert conversation["message_count"] == 1

    def test_get_messages(self):
        """Test retrieving messages from conversation."""
        conversation_id = self.manager.create_conversation()
        
        self.manager.add_message(conversation_id, "user", "Message 1")
        self.manager.add_message(conversation_id, "assistant", "Message 2")
        self.manager.add_message(conversation_id, "user", "Message 3")
        
        messages = self.manager.get_messages(conversation_id)
        assert len(messages) == 3
        assert messages[0]["content"] == "Message 1"
        assert messages[1]["content"] == "Message 2"
        assert messages[2]["content"] == "Message 3"

    def test_get_messages_with_pagination(self):
        """Test message pagination."""
        conversation_id = self.manager.create_conversation()
        
        for i in range(10):
            self.manager.add_message(conversation_id, "user", f"Message {i}")
        
        # Get first 5
        messages = self.manager.get_messages(conversation_id, limit=5)
        assert len(messages) == 5
        
        # Get next 5
        messages = self.manager.get_messages(conversation_id, limit=5, offset=5)
        assert len(messages) == 5
        assert messages[0]["content"] == "Message 5"

    def test_list_conversations(self):
        """Test listing conversations."""
        self.manager.create_conversation("Conv 1")
        self.manager.create_conversation("Conv 2")
        self.manager.create_conversation("Conv 3")
        
        conversations = self.manager.list_conversations()
        assert len(conversations) == 3
        # Most recently updated should be first
        assert conversations[0]["title"] == "Conv 3"
        assert conversations[1]["title"] == "Conv 2"
        assert conversations[2]["title"] == "Conv 1"

    def test_delete_conversation(self):
        """Test deleting a conversation."""
        conversation_id = self.manager.create_conversation()
        
        assert self.manager.delete_conversation(conversation_id) is True
        assert self.manager.get_conversation(conversation_id) is None
        assert self.manager.get_messages(conversation_id) == []
        
        # Deleting non-existent conversation should return False
        assert self.manager.delete_conversation("nonexistent") is False

    def test_update_conversation_title(self):
        """Test updating conversation title."""
        conversation_id = self.manager.create_conversation("Old Title")
        
        assert self.manager.update_conversation_title(conversation_id, "New Title") is True
        
        conversation = self.manager.get_conversation(conversation_id)
        assert conversation["title"] == "New Title"

    def test_search_messages(self):
        """Test searching messages."""
        conversation_id = self.manager.create_conversation()
        
        self.manager.add_message(conversation_id, "user", "Hello world")
        self.manager.add_message(conversation_id, "assistant", "How are you?")
        self.manager.add_message(conversation_id, "user", "World peace")
        
        results = self.manager.search_messages("world")
        assert len(results) == 2  # "Hello world" and "World peace"
        
        results = self.manager.search_messages("hello")
        assert len(results) == 1

    def test_search_messages_in_conversation(self):
        """Test searching messages within a specific conversation."""
        conv1 = self.manager.create_conversation("Conv 1")
        conv2 = self.manager.create_conversation("Conv 2")
        
        self.manager.add_message(conv1, "user", "apple")
        self.manager.add_message(conv2, "user", "banana")
        self.manager.add_message(conv2, "user", "apple pie")
        
        # Search in conv2 only
        results = self.manager.search_messages("apple", conversation_id=conv2)
        assert len(results) == 1
        assert results[0]["content"] == "apple pie"

    def test_export_import_conversation(self):
        """Test exporting and importing conversations."""
        conversation_id = self.manager.create_conversation("Export Test")
        self.manager.add_message(conversation_id, "user", "Test message")
        
        # Export
        json_data = self.manager.export_conversation(conversation_id)
        assert json_data is not None
        assert "Test message" in json_data
        
        # Import into new manager
        new_manager = ChatHistoryManager()
        imported_id = new_manager.import_conversation(json_data)
        assert imported_id is not None
        assert imported_id == conversation_id
        
        # Verify imported data
        messages = new_manager.get_messages(imported_id)
        assert len(messages) == 1
        assert messages[0]["content"] == "Test message"

    def test_get_conversation_summary(self):
        """Test getting conversation summary."""
        conversation_id = self.manager.create_conversation("Summary Test")
        self.manager.add_message(conversation_id, "user", "First message")
        self.manager.add_message(conversation_id, "assistant", "Second message")
        
        summary = self.manager.get_conversation_summary(conversation_id)
        
        assert summary is not None
        assert summary["title"] == "Summary Test"
        assert summary["message_count"] == 2
        assert summary["first_message"]["content"] == "First message"
        assert summary["last_message"]["content"] == "Second message"

    def test_get_stats(self):
        """Test getting chat statistics."""
        self.manager.create_conversation("Conv 1")
        self.manager.create_conversation("Conv 2")
        
        conv1 = list(self.manager._conversations.keys())[0]
        self.manager.add_message(conv1, "user", "Message 1")
        self.manager.add_message(conv1, "user", "Message 2")
        
        stats = self.manager.get_stats()
        
        assert stats["total_conversations"] == 2
        assert stats["total_messages"] == 2
        assert stats["avg_messages_per_conversation"] == 1.0

    def test_global_instance(self):
        """Test getting global chat history manager."""
        manager = get_chat_history_manager()
        assert isinstance(manager, ChatHistoryManager)

    def test_empty_conversation(self):
        """Test operations on empty/non-existent conversation."""
        assert self.manager.get_conversation("nonexistent") is None
        assert self.manager.get_messages("nonexistent") == []
        assert self.manager.get_conversation_summary("nonexistent") is None

    def test_message_metadata(self):
        """Test adding messages with metadata."""
        conversation_id = self.manager.create_conversation()
        
        metadata = {"timestamp": "2024-01-01", "user_id": "123"}
        message = self.manager.add_message(
            conversation_id, "user", "Hello", metadata=metadata
        )
        
        assert message["metadata"] == metadata