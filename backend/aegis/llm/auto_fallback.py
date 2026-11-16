# backend/aegis/llm/auto_fallback.py
"""Auto-fallback LLM provider.

If the primary provider fails with quota exhaustion, timeouts, or empty responses,
automatically falls back to the configured fallback provider without changing call sites.
Provides bidirectional failover, automatic recovery after cooldown, method adaptation,
and complete thread-safe state management.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from aegis.core.config import settings
from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider, LLMUnavailableError
from aegis.llm.client import GeminiClient
from aegis.llm.groq_client import DEFAULT_GROQ_MODEL, GroqClient
from aegis.llm.openlm_client import OpenLMClient

log = get_logger(__name__)

_FALLBACK_COOLDOWN_SECONDS = 45.0
T = TypeVar("T", bound=BaseModel)


def _is_provider_ready(provider: LLMProvider | None) -> bool:
    """Return True if the provider is instantiated and has active credentials."""
    if provider is None:
        return False
    # If explicitly set as a boolean or method on instance/class (except MagicMock auto-created)
    if "is_configured" in provider.__dict__:
        val = provider.__dict__["is_configured"]
        return bool(val() if callable(val) else val)
    if "is_configured" in type(provider).__dict__:
        try:
            val = getattr(provider, "is_configured")
            return bool(val() if callable(val) else val)
        except Exception:
            pass
    if isinstance(provider, GeminiClient):
        return len(getattr(provider, "_clients", [])) > 0
    if isinstance(provider, GroqClient):
        return getattr(provider, "_client", None) is not None
    if isinstance(provider, OpenLMClient):
        return getattr(provider, "is_configured", False)
    if isinstance(provider, AutoFallbackLLM):
        return provider.is_configured
    return True


def _is_empty_or_placeholder(result: Any) -> bool:
    """Return True if the result represents an unpopulated or dry-run response."""
    if result is None:
        return True
    if isinstance(result, str):
        s = result.strip()
        if not s:
            return True
        s_lower = s.lower()
        return s_lower.startswith("[dry-run]") or "[dry-run]" in s_lower[:40]
    if isinstance(result, dict):
        return len(result) == 0
    if isinstance(result, (list, tuple)):
        return len(result) == 0
    if isinstance(result, BaseModel):
        d = result.model_dump()
        if not d:
            return True
        # Compare with default instance if constructible without required args
        try:
            default_instance = type(result)()
            if result == default_instance:
                return True
        except Exception:
            pass
        return not any(
            v is not None and v != "" and v != [] and v != {}
            for v in d.values()
        )
    return False


def _invoke_provider_method(
    provider: LLMProvider,
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Invoke a method on the provider, adapting between LLMProvider methods if needed."""
    # 1. Direct call if available
    fn = getattr(provider, method_name, None)
    if callable(fn):
        try:
            return fn(*args, **kwargs)
        except (NotImplementedError, AttributeError):
            pass

    # 2. Adaptation for generate_model_safe / generate_model
    if method_name in ("generate_model_safe", "generate_model"):
        alt_name = "generate_model" if method_name == "generate_model_safe" else "generate_model_safe"
        alt_fn = getattr(provider, alt_name, None)
        if callable(alt_fn):
            return alt_fn(*args, **kwargs)

        raw_fn = getattr(provider, "generate_model_raw", None)
        if callable(raw_fn):
            raw = raw_fn(*args, **kwargs)
            model_cls = args[1] if len(args) > 1 else kwargs.get("model_cls")
            if isinstance(raw, BaseModel):
                return raw
            if isinstance(raw, dict) and model_cls is not None:
                return model_cls(**raw) if raw else model_cls()

        json_fn = getattr(provider, "generate_json", None)
        if callable(json_fn):
            prompt = args[0] if args else kwargs.get("prompt", "")
            model_cls = args[1] if len(args) > 1 else kwargs.get("model_cls")
            raw = json_fn(
                prompt,
                system_instruction=kwargs.get("system_instruction"),
                temperature=kwargs.get("temperature", 0.7),
            )
            if isinstance(raw, dict) and model_cls is not None:
                return model_cls(**raw) if raw else model_cls()

    # 3. Adaptation for generate_model_raw
    elif method_name == "generate_model_raw":
        for cand_name in ("generate_model_safe", "generate_model"):
            cand_fn = getattr(provider, cand_name, None)
            if callable(cand_fn):
                res = cand_fn(*args, **kwargs)
                if isinstance(res, BaseModel):
                    return res.model_dump()
                if isinstance(res, dict):
                    return res

        json_fn = getattr(provider, "generate_json", None)
        if callable(json_fn):
            prompt = args[0] if args else kwargs.get("prompt", "")
            return json_fn(
                prompt,
                system_instruction=kwargs.get("system_instruction"),
                temperature=kwargs.get("temperature", 0.7),
            )

    # 4. Adaptation for generate_json
    elif method_name == "generate_json":
        gen_fn = getattr(provider, "generate", None)
        if callable(gen_fn):
            text = gen_fn(*args, **kwargs)
            if isinstance(text, str):
                cleaned = text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
                match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        pass
            return {}

    raise AttributeError(f"Provider {type(provider).__name__} does not implement {method_name}")


class AutoFallbackLLM(LLMProvider):
    """Primary LLM (Gemini 2.5/3.5/3.8 Flash) with instant fallback to Groq / OpenLM."""

    def __init__(
        self,
        primary: LLMProvider | None = None,
        fallback: LLMProvider | None = None,
    ) -> None:
        provider = getattr(settings, "AEGIS_LLM_PROVIDER", "auto").lower()
        fallback_cfg = getattr(settings, "AEGIS_LLM_FALLBACK_PROVIDER", "groq").lower()

        if primary is not None:
            self._primary = primary
        elif provider == "groq":
            self._primary = GroqClient()
        elif provider == "openlm":
            self._primary = OpenLMClient()
        else:
            self._primary = GeminiClient()

        if fallback is not None:
            self._fallback = fallback
        elif provider == "groq":
            if fallback_cfg == "openlm":
                self._fallback = OpenLMClient()
            elif fallback_cfg == "gemini":
                self._fallback = GeminiClient()
            else:
                self._fallback = None
        elif provider == "openlm":
            if fallback_cfg == "groq":
                self._fallback = GroqClient(model_id=DEFAULT_GROQ_MODEL)
            else:
                self._fallback = GeminiClient()
        else:
            if fallback_cfg == "openlm":
                openlm = OpenLMClient()
                self._fallback = openlm if openlm.is_configured else GroqClient(model_id=DEFAULT_GROQ_MODEL)
            elif fallback_cfg == "groq":
                self._fallback = GroqClient(model_id=DEFAULT_GROQ_MODEL)
            else:
                self._fallback = None

        if self._fallback is self._primary:
            self._fallback = None

        self._using_fallback = False
        self._fallback_until: float = 0.0
        self._fallback_attempts = 0
        self._max_fallback_attempts = 1000
        self._lock_obj = threading.RLock()
        log.info(
            "AutoFallbackLLM configured",
            primary=type(self._primary).__name__,
            fallback=type(self._fallback).__name__ if self._fallback else None,
        )

    @property
    def _lock(self) -> threading.RLock:
        if not hasattr(self, "_lock_obj"):
            self._lock_obj = threading.RLock()
        return self._lock_obj

    @property
    def primary(self) -> LLMProvider:
        return self._primary

    @property
    def fallback(self) -> LLMProvider | None:
        return self._fallback

    @property
    def active_provider(self) -> LLMProvider:
        return self._get_active()

    @property
    def is_using_fallback(self) -> bool:
        with self._lock:
            active = self._get_active()
            return (
                getattr(self, "_fallback", None) is not None
                and active is getattr(self, "_fallback", None)
                and getattr(self, "_fallback", None) is not getattr(self, "_primary", None)
            )

    @property
    def model_id(self) -> str:
        """Return the active provider's model ID."""
        active = self._get_active()
        return getattr(active, "_model_id", getattr(active, "model_id", "auto"))

    @property
    def is_configured(self) -> bool:
        """Return True if either primary or fallback provider is configured."""
        prim = getattr(self, "_primary", None)
        fb = getattr(self, "_fallback", None)
        return _is_provider_ready(prim) or _is_provider_ready(fb)

    def switch_to_fallback(self, reason: str = "") -> None:
        """Explicitly switch traffic to the fallback provider."""
        with self._lock:
            if getattr(self, "_fallback", None) is None:
                log.warning("Cannot switch to fallback: no fallback provider configured")
                return
            self._using_fallback = True
            self._fallback_until = time.monotonic() + _FALLBACK_COOLDOWN_SECONDS
            log.info("Switched to fallback LLM provider", reason=reason)

    def switch_to_primary(self) -> None:
        """Explicitly restore traffic to the primary provider."""
        with self._lock:
            self._using_fallback = False
            self._fallback_until = 0.0
            self._fallback_attempts = 0
            log.info("Switched to primary LLM provider")

    def _get_active(self) -> LLMProvider:
        with self._lock:
            using_fb = getattr(self, "_using_fallback", False)
            fb = getattr(self, "_fallback", None)
            prim = getattr(self, "_primary", None)

            if using_fb and fb is not None:
                until = getattr(self, "_fallback_until", 0.0)
                if until > 0.0 and time.monotonic() >= until:
                    if _is_provider_ready(prim):
                        log.info("Fallback cooldown elapsed — restoring primary provider")
                        self._using_fallback = False
                        self._fallback_until = 0.0
                        return prim
                    else:
                        # Primary is still not ready; extend cooldown
                        self._fallback_until = time.monotonic() + _FALLBACK_COOLDOWN_SECONDS
                return fb

            # If primary has no credentials / is dry-run, but fallback is ready, route to fallback
            if prim is not None and not _is_provider_ready(prim) and fb is not None and _is_provider_ready(fb):
                return fb

            return prim if prim is not None else fb

    def _handle_error(self, exc: Exception, failing_client: LLMProvider | None = None) -> bool:
        """Return True if failover to alternate provider was performed and should be retried."""
        error_str = str(exc).lower()
        exc_type = type(exc).__name__.lower()
        is_provider_error = (
            isinstance(exc, (LLMUnavailableError, TimeoutError, ConnectionError, NotImplementedError))
            or any(
                token in exc_type
                for token in [
                    "ratelimit",
                    "timeout",
                    "connection",
                    "unavailable",
                    "quota",
                    "overloaded",
                    "apierror",
                ]
            )
            or any(
                token in error_str
                for token in [
                    "429",
                    "404",
                    "500",
                    "502",
                    "503",
                    "504",
                    "not_found",
                    "resource_exhausted",
                    "quota",
                    "rate limit",
                    "internal",
                    "service unavailable",
                    "unavailable",
                    "timeout",
                    "timed out",
                    "deadline_exceeded",
                    "deadline exceeded",
                    "deadline expired",
                    "gemini api error",
                    "gemini call failed",
                    "exhausted",
                    "connection",
                ]
            )
        )
        if not is_provider_error:
            return False

        with self._lock:
            prim = getattr(self, "_primary", None)
            fb = getattr(self, "_fallback", None)

            # If failing_client is primary, switch to fallback
            if failing_client is not None and failing_client is prim:
                can_fallback = (
                    fb is not None
                    and fb is not prim
                    and _is_provider_ready(fb)
                )
                if can_fallback:
                    attempts = getattr(self, "_fallback_attempts", 0)
                    log.warning(
                        "Primary LLM error/timeout — switching immediately to fallback provider.",
                        attempt=attempts + 1,
                        error=error_str[:160],
                    )
                    self._using_fallback = True
                    self._fallback_until = time.monotonic() + _FALLBACK_COOLDOWN_SECONDS
                    self._fallback_attempts = attempts + 1
                    return True

            # If failing_client is fallback, switch back to primary
            elif failing_client is not None and failing_client is fb:
                can_primary = (
                    prim is not None
                    and prim is not fb
                    and _is_provider_ready(prim)
                )
                if can_primary:
                    log.warning(
                        "Fallback LLM error/timeout — switching back to primary provider.",
                        error=error_str[:160],
                    )
                    self._using_fallback = False
                    self._fallback_until = 0.0
                    return True

            # Generic case when failing_client is not explicitly passed
            else:
                if not getattr(self, "_using_fallback", False) and fb is not None:
                    if _is_provider_ready(fb):
                        attempts = getattr(self, "_fallback_attempts", 0)
                        log.warning(
                            "LLM error/timeout — switching immediately to fallback provider.",
                            attempt=attempts + 1,
                            error=error_str[:160],
                        )
                        self._using_fallback = True
                        self._fallback_until = time.monotonic() + _FALLBACK_COOLDOWN_SECONDS
                        self._fallback_attempts = attempts + 1
                        return True
                elif getattr(self, "_using_fallback", False) and prim is not None:
                    if _is_provider_ready(prim):
                        log.warning(
                            "LLM error/timeout — restoring primary provider.",
                            error=error_str[:160],
                        )
                        self._using_fallback = False
                        self._fallback_until = 0.0
                        return True

        return False

    def _call_with_retry(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the active client with zero-sleep instant provider failover."""
        max_retries = 3
        base_delay = 0.25

        for attempt in range(max_retries):
            client = self._get_active()
            try:
                result = _invoke_provider_method(client, method_name, *args, **kwargs)

                # If primary returned empty or dry-run placeholder, failover to fallback
                if _is_empty_or_placeholder(result) and attempt < max_retries - 1:
                    prim = getattr(self, "_primary", None)
                    fb = getattr(self, "_fallback", None)

                    if client is prim and fb is not None and _is_provider_ready(fb):
                        log.warning(
                            "Primary LLM returned empty/dry-run placeholder — failing over to fallback provider",
                            method=method_name,
                            attempt=attempt + 1,
                        )
                        with self._lock:
                            self._using_fallback = True
                            self._fallback_until = time.monotonic() + _FALLBACK_COOLDOWN_SECONDS
                            self._fallback_attempts = getattr(self, "_fallback_attempts", 0) + 1
                        continue
                    elif client is fb and prim is not None and _is_provider_ready(prim):
                        log.warning(
                            "Fallback LLM returned empty/dry-run placeholder — restoring primary provider",
                            method=method_name,
                            attempt=attempt + 1,
                        )
                        with self._lock:
                            self._using_fallback = False
                            self._fallback_until = 0.0
                        continue

                # If primary succeeded with valid content, reset fallback attempts
                if client is getattr(self, "_primary", None) and not _is_empty_or_placeholder(result):
                    with self._lock:
                        self._fallback_attempts = 0

                return result

            except Exception as exc:
                if attempt < max_retries - 1 and self._handle_error(exc, failing_client=client):
                    log.info("Retrying immediately on alternate provider", attempt=attempt + 1)
                    continue
                elif attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    log.info(f"Retrying same provider after {delay}s", attempt=attempt + 1, error=str(exc)[:160])
                    time.sleep(delay)
                    continue

                log.error("All LLM providers failed", method=method_name, error=str(exc)[:300])
                raise RuntimeError(
                    f"All configured LLM providers failed during {method_name}"
                ) from exc

        raise RuntimeError("Max retries exceeded")

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int | None = None,
    ) -> str:
        return self._call_with_retry(
            "generate",
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        response_schema: type[Any] | None = None,
    ) -> dict[str, Any]:
        return self._call_with_retry(
            "generate_json",
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            response_schema=response_schema,
        )

    def generate_model(
        self,
        prompt: str,
        model_cls: type[T],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> T:
        return self._call_with_retry(
            "generate_model",
            prompt,
            model_cls,
            system_instruction=system_instruction,
            temperature=temperature,
        )

    def generate_model_safe(
        self,
        prompt: str,
        model_cls: type[T],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> T:
        return self._call_with_retry(
            "generate_model_safe",
            prompt,
            model_cls,
            system_instruction=system_instruction,
            temperature=temperature,
        )

    def generate_model_raw(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        return self._call_with_retry(
            "generate_model_raw",
            prompt,
            model_cls,
            system_instruction=system_instruction,
            temperature=temperature,
        )
