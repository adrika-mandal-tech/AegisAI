# backend/tests/conftest.py
"""Shared fixtures for AegisAI test suite."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aegis.api.schemas import AgentRequest
from aegis.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """FastAPI test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_llm() -> MagicMock:
    """Return a MagicMock that mimics ``GeminiClient.generate_model``."""
    mock = MagicMock()

    def fake_generate_model(
        prompt: str,
        model_cls: type,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> object:
        return model_cls()

    def fake_generate_model_safe(
        prompt: str,
        model_cls: type,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> object:
        return model_cls()

    mock.generate_model.side_effect = fake_generate_model
    mock.generate_model_safe.side_effect = fake_generate_model_safe
    return mock


@pytest.fixture
def mock_retrieval() -> MagicMock:
    """Return a MagicMock that mimics ``RetrievalPipeline``."""
    mock = MagicMock()
    mock.retrieve.return_value = []
    return mock


@pytest.fixture
def sample_request() -> AgentRequest:
    """A basic ``AgentRequest`` for endpoint tests."""
    return AgentRequest(input="test task")
