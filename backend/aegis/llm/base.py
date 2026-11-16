# backend/aegis/llm/base.py
"""Abstract LLM provider interface used by agents."""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping

from pydantic import BaseModel


class LLMUnavailableError(Exception):
    """Raised when an LLM provider fails on every configured key.

    Carriers of this error are expected to trigger provider fallback
    in :class:`aegis.llm.auto_fallback.AutoFallbackLLM`.
    """


class LLMProvider:
    """Minimal interface that all LLM provider clients must implement."""

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        response_schema: type[Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def generate_model(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> BaseModel:
        raise NotImplementedError

    def generate_model_raw(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def generate_model_safe(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> BaseModel:
        raise NotImplementedError
