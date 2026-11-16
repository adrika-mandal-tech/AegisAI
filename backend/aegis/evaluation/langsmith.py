# backend/aegis/evaluation/langsmith.py
"""LangSmith client initialisation for tracing and evaluation.

Tracing is enabled automatically when ``LANGCHAIN_API_KEY`` and
``LANGCHAIN_TRACING_V2=true`` are set in the environment.
"""

from __future__ import annotations

import os

from langsmith import Client as LangSmithClient

from aegis.core.config import settings
from aegis.core.logging import get_logger

log = get_logger(__name__)


def get_langsmith_client() -> LangSmithClient | None:
    """Return a configured LangSmith client, or ``None`` if no API key set."""
    api_key = settings.AEGIS_LANGSMITH_API_KEY or os.environ.get("AEGIS_LANGSMITH_API_KEY")
    if not api_key:
        log.warning("AEGIS_LANGSMITH_API_KEY not set — LangSmith tracing disabled.")
        return None

    # The SDK reads LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT, etc.
    # from the environment automatically when they are set.
    try:
        client = LangSmithClient(api_key=api_key)
        log.info("LangSmith client initialised")
        return client
    except Exception:
        log.exception("Failed to initialise LangSmith client")
        return None
