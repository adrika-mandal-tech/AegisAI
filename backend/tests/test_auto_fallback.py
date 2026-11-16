from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from aegis.llm.auto_fallback import AutoFallbackLLM, _is_empty_or_placeholder, _is_provider_ready
from aegis.llm.base import LLMUnavailableError

pytestmark = pytest.mark.unit


class DummyModel(BaseModel):
    summary: str = ""
    status: str = "pending"


class PopulatedModel(BaseModel):
    summary: str
    status: str


def test_generate_raises_when_all_providers_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = object.__new__(AutoFallbackLLM)
    provider._primary = MagicMock()
    provider._primary.generate.side_effect = RuntimeError("quota exhausted")
    provider._fallback = None
    provider._using_fallback = False
    provider._fallback_attempts = 0
    provider._max_fallback_attempts = 2
    monkeypatch.setattr("aegis.llm.auto_fallback.time.sleep", lambda _delay: None)

    with pytest.raises(RuntimeError, match="All configured LLM providers failed"):
        provider.generate("grounded prompt")


def test_auto_fallback_switches_to_fallback_on_error() -> None:
    provider = object.__new__(AutoFallbackLLM)
    provider._primary = MagicMock()
    provider._primary.generate.side_effect = LLMUnavailableError("Rate limit 429")
    provider._fallback = MagicMock()
    provider._fallback.generate.return_value = "Fallback response"
    provider._using_fallback = False
    provider._fallback_until = 0.0
    provider._fallback_attempts = 0
    provider._max_fallback_attempts = 1000

    res = provider.generate("test prompt")
    assert res == "Fallback response"
    assert provider._using_fallback is True
    assert provider.is_using_fallback is True


def test_auto_fallback_switches_when_primary_returns_dry_run() -> None:
    provider = object.__new__(AutoFallbackLLM)
    provider._primary = MagicMock()
    provider._primary.generate.return_value = "[dry-run] Gemini response placeholder"
    provider._primary._clients = []
    provider._fallback = MagicMock()
    provider._fallback._client = MagicMock()
    provider._fallback.generate.return_value = "Groq response"
    provider._using_fallback = False
    provider._fallback_until = 0.0
    provider._fallback_attempts = 0
    provider._max_fallback_attempts = 1000

    res = provider.generate("test prompt")
    assert res == "Groq response"


def test_generate_model_safe_switches_on_empty_model() -> None:
    provider = object.__new__(AutoFallbackLLM)
    provider._primary = MagicMock()
    # Primary returns empty default model
    provider._primary.generate_model_safe.return_value = DummyModel()
    provider._fallback = MagicMock()
    provider._fallback.generate_model_safe.return_value = DummyModel(summary="Real result", status="success")
    provider._using_fallback = False
    provider._fallback_until = 0.0
    provider._fallback_attempts = 0

    res = provider.generate_model_safe("summarize this", DummyModel)
    assert res.summary == "Real result"
    assert res.status == "success"
    assert provider.is_using_fallback is True


def test_generate_model_raw_method_adaptation() -> None:
    """When a provider only implements generate_model, generate_model_raw adapts seamlessly."""
    mock_provider = MagicMock()
    del mock_provider.generate_model_raw  # Provider lacks generate_model_raw
    mock_provider.generate_model_safe.return_value = PopulatedModel(summary="Adapted data", status="verified")

    provider = AutoFallbackLLM(primary=mock_provider, fallback=None)
    raw = provider.generate_model_raw("give data", PopulatedModel)
    assert raw == {"summary": "Adapted data", "status": "verified"}


def test_bidirectional_failover_recovers_to_primary() -> None:
    """If fallback provider fails while active and primary is ready, it restores primary."""
    provider = object.__new__(AutoFallbackLLM)
    provider._primary = MagicMock()
    provider._primary.generate.return_value = "Primary recovered"
    provider._fallback = MagicMock()
    provider._fallback.generate.side_effect = LLMUnavailableError("Groq 503 service unavailable")
    provider._using_fallback = True  # Start in fallback mode
    provider._fallback_until = time.monotonic() + 100.0
    provider._fallback_attempts = 1

    res = provider.generate("query")
    assert res == "Primary recovered"
    assert provider.is_using_fallback is False


def test_fallback_cooldown_restores_primary_after_expiry() -> None:
    """Active provider reverts to primary after cooldown window passes."""
    provider = object.__new__(AutoFallbackLLM)
    provider._primary = MagicMock()
    provider._fallback = MagicMock()
    provider._using_fallback = True
    provider._fallback_until = time.monotonic() - 1.0  # Expired in past

    assert provider._get_active() is provider._primary
    assert provider.is_using_fallback is False


def test_manual_switch_controls() -> None:
    """switch_to_fallback and switch_to_primary operate deterministically."""
    provider = object.__new__(AutoFallbackLLM)
    provider._primary = MagicMock()
    provider._fallback = MagicMock()
    provider._using_fallback = False
    provider._fallback_until = 0.0

    provider.switch_to_fallback("testing switch")
    assert provider.is_using_fallback is True
    assert provider.active_provider is provider._fallback

    provider.switch_to_primary()
    assert provider.is_using_fallback is False
    assert provider.active_provider is provider._primary


def test_is_empty_or_placeholder_helpers() -> None:
    assert _is_empty_or_placeholder(None) is True
    assert _is_empty_or_placeholder("") is True
    assert _is_empty_or_placeholder("   ") is True
    assert _is_empty_or_placeholder("[dry-run] Gemini response placeholder") is True
    assert _is_empty_or_placeholder("[DRY-RUN] Groq placeholder") is True
    assert _is_empty_or_placeholder({}) is True
    assert _is_empty_or_placeholder([]) is True
    assert _is_empty_or_placeholder(DummyModel()) is True

    assert _is_empty_or_placeholder("Valid response") is False
    assert _is_empty_or_placeholder({"key": "val"}) is False
    assert _is_empty_or_placeholder(PopulatedModel(summary="content", status="ok")) is False
