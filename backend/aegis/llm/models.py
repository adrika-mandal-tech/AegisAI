# backend/aegis/llm/models.py
"""LLM model constants and sanitizer.

Primary LLM: Google Gemini 2.5 Flash (``gemini-2.5-flash``) via ``google-genai``.
"""

from __future__ import annotations

from collections.abc import Mapping

from aegis.core.config import settings

# ── Gemini models (Section 2 & Section 8: Gemini 2.5 Flash primary) ───────
GEMINI_MODELS: Mapping[str, str] = {
    "pro": "gemini-2.5-flash",
    "flash": "gemini-2.5-flash",
    "pro-exp": "gemini-2.5-flash",
}

# Known retired legacy model aliases mapped cleanly to gemini-2.5-flash
_INVALID_GEMINI_ALIASES: set[str] = {
    "gemini-1.5-flash",
    "gemini-pro",
}

# ── Groq models ────────────────────────────────────────────────────────────
GROQ_MODELS: Mapping[str, str] = {
    "pro": "openai/gpt-oss-120b",
    "flash": "openai/gpt-oss-20b",
    "mixtral": "openai/gpt-oss-120b",
}


def resolve_gemini_model(configured: str | None = None) -> str:
    """Return a valid Gemini model ID, defaulting to ``gemini-2.5-flash``."""
    candidate = (configured or getattr(settings, "AEGIS_GEMINI_MODEL", "") or "").strip()
    if not candidate or candidate.lower() in _INVALID_GEMINI_ALIASES:
        return "gemini-2.5-flash"
    return candidate


def _default_model() -> str:
    provider = getattr(settings, "AEGIS_LLM_PROVIDER", "auto").lower()
    if provider == "groq":
        return GROQ_MODELS["pro"]
    if provider == "openlm":
        return getattr(settings, "AEGIS_OPENLM_MODEL", "openlm-research/open_llama_7b")
    return resolve_gemini_model()


DEFAULT_MODEL: str = _default_model()
FAST_MODEL: str = GROQ_MODELS["flash"]
