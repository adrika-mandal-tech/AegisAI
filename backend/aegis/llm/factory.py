# backend/aegis/llm/factory.py
"""LLM provider factory.

Primary LLM: Google Gemini 2.5 Flash (``GeminiClient``).
Modular Secondary / Gateway Providers: ``OpenLMClient`` and ``GroqClient`` via
``AutoFallbackLLM``.
"""

from __future__ import annotations

from aegis.core.config import settings
from aegis.core.logging import get_logger
from aegis.llm.auto_fallback import AutoFallbackLLM
from aegis.llm.client import GeminiClient
from aegis.llm.groq_client import GroqClient
from aegis.llm.openlm_client import OpenLMClient

log = get_logger(__name__)


def get_default_llm():
    """Return the configured LLM provider instance with fallback support."""
    provider = getattr(settings, "AEGIS_LLM_PROVIDER", "auto").lower()
    if provider == "groq":
        return GroqClient()
    if provider == "gemini":
        return GeminiClient()
    if provider == "openlm":
        return OpenLMClient()
    return AutoFallbackLLM()
