# backend/aegis/llm/client.py
"""Production Gemini 2.5 Flash client with multi-key rotation, timeouts, retries,
structured Pydantic outputs, and token usage tracking (Section 8).
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel

from aegis.core.config import settings
from aegis.core.constants import DEFAULT_TEMPERATURE
from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider, LLMUnavailableError
from aegis.llm.models import resolve_gemini_model

log = get_logger(__name__)


def _strip_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove ``additionalProperties`` unsupported by Gemini proto schema."""
    if not isinstance(schema, dict):
        return schema
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        if isinstance(value, dict):
            result[key] = _strip_additional_properties(value)
        elif isinstance(value, list):
            result[key] = [
                _strip_additional_properties(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _build_generate_config(
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int | None = None,
    system_instruction: str | None = None,
    response_mime_type: str | None = None,
    response_schema: Any = None,
    thinking_budget: int | None = None,
) -> genai_types.GenerateContentConfig:
    """Build GenerateContentConfig for sub-2s latency across Gemini 2.5 and 3.x."""
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "system_instruction": system_instruction,
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    if response_mime_type is not None:
        kwargs["response_mime_type"] = response_mime_type
    if response_schema is not None:
        kwargs["response_schema"] = response_schema

    if thinking_budget is not None and thinking_budget > 0 and hasattr(genai_types, "ThinkingConfig"):
        try:
            kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        except Exception:
            pass

    return genai_types.GenerateContentConfig(**kwargs)


class GeminiClient(LLMProvider):
    """Thread-safe wrapper around ``google.genai.Client`` targeting ``gemini-2.5-flash``."""

    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = resolve_gemini_model(model_id)
        self._clients: list[Any] = []
        self._key_index = 0
        self._lock = threading.Lock()
        self.usage_stats: dict[str, Any] = {
            "provider": "gemini",
            "model_id": self._model_id,
            "total_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "last_latency_ms": 0,
        }
        self._init_clients()

    def _init_clients(self) -> None:
        keys = settings.gemini_api_keys
        if not keys:
            log.warning(
                "No Gemini API keys configured — dry-run mode. "
                "Set AEGIS_GEMINI_API_KEYS in your .env file."
            )
            return

        configured_s = int(getattr(settings, "AEGIS_GEMINI_TIMEOUT_SECONDS", 20) or 20)
        timeout_ms = min(max(configured_s * 1000, 8000), 22000)
        http_options = genai_types.HttpOptions(timeout=timeout_ms)

        self._clients = []
        for key in keys:
            try:
                client = genai.Client(api_key=key, http_options=http_options)
                self._clients.append(client)
            except Exception as exc:
                log.warning("Failed to create Gemini client for a key", error=str(exc)[:100])

        log.info(
            "GeminiClient initialised",
            model_id=self._model_id,
            key_count=len(self._clients),
            timeout_ms=timeout_ms,
        )

    def _get_client(self) -> Any | None:
        with self._lock:
            if not self._clients:
                return None
            client = self._clients[self._key_index % len(self._clients)]
            self._key_index += 1
            return client

    def _record_usage(self, response: Any, latency_ms: int) -> None:
        with self._lock:
            self.usage_stats["total_calls"] += 1
            self.usage_stats["last_latency_ms"] = latency_ms
            self.usage_stats["model_id"] = self._model_id
            meta = getattr(response, "usage_metadata", None)
            if meta is not None:
                p_tok = int(getattr(meta, "prompt_token_count", 0) or 0)
                c_tok = int(getattr(meta, "candidates_token_count", 0) or 0)
                t_tok = int(getattr(meta, "total_token_count", 0) or (p_tok + c_tok))
                self.usage_stats["prompt_tokens"] += p_tok
                self.usage_stats["completion_tokens"] += c_tok
                self.usage_stats["total_tokens"] += t_tok

    def _execute_with_retry(self, call_fn: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._clients:
            log.warning("GeminiClient dry-run — no keys configured.")
            return None

        from google.genai import errors as genai_errors  # noqa: PLC0415

        max_retries = min(2, max(1, len(self._clients)))
        last_error: Exception | None = None
        for attempt in range(max_retries):
            client = self._get_client()
            if client is None:
                break

            start = time.time()
            try:
                return call_fn(client, start, *args, **kwargs)
            except genai_errors.APIError as exc:
                last_error = exc
                err_str = str(exc)
                err_upper = err_str.upper()
                # Auto-heal retired/missing Gemini Flash model IDs on 404 NOT_FOUND
                if "404" in err_upper or "NOT_FOUND" in err_upper:
                    alt_model = "gemini-3.5-flash-lite"
                    log.warning(
                        "Configured Gemini model returned 404; switching automatically",
                        previous_model=self._model_id,
                        fallback_model=alt_model,
                    )
                    self._model_id = alt_model
                    return call_fn(client, start, *args, **kwargs)
                if any(
                    tok in err_upper
                    for tok in (
                        "429",
                        "RESOURCE_EXHAUSTED",
                        "503",
                        "UNAVAILABLE",
                        "500",
                        "INTERNAL",
                        "504",
                        "DEADLINE_EXCEEDED",
                        "DEADLINE EXPIRED",
                        "TIMEOUT",
                        "TIMED OUT",
                    )
                ):
                    log.warning(
                        f"Gemini API transient error ({getattr(exc, 'code', 'err')}), rotating key",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        model=self._model_id,
                    )
                    if attempt < max_retries - 1:
                        time.sleep(0.15)
                        continue
                    raise LLMUnavailableError(f"Gemini API error: {err_str[:300]}") from exc
                raise LLMUnavailableError(f"Gemini API error: {err_str[:300]}") from exc
            except json.JSONDecodeError:
                raise
            except Exception as exc:
                last_error = exc
                err_upper = str(exc).upper()
                if attempt < max_retries - 1 and any(
                    tok in err_upper for tok in ("TIMEOUT", "TIMED OUT", "DEADLINE", "504", "503", "CONNECTION")
                ):
                    time.sleep(0.15)
                    continue
                log.warning("Gemini API call failed", error=str(exc)[:200])
                raise LLMUnavailableError(f"Gemini call failed: {str(exc)[:300]}") from exc

        raise LLMUnavailableError(
            f"All {max_retries} Gemini API keys exhausted. "
            f"Last error: {str(last_error)[:200] if last_error else 'unknown'}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int | None = None,
    ) -> str:
        if not self._clients:
            return "[dry-run] Gemini response placeholder"

        config = _build_generate_config(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )

        def _call(client: Any, t0: float) -> str:
            response = client.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=config,
            )
            self._record_usage(response, int((time.time() - t0) * 1000))
            return response.text or ""

        result = self._execute_with_retry(_call)
        return result if result is not None else "[dry-run] Gemini response placeholder"

    def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        response_schema: type[Any] | None = None,
    ) -> dict[str, Any]:
        if not self._clients:
            return {}

        config = _build_generate_config(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
            system_instruction=system_instruction,
        )

        def _call(client: Any, t0: float) -> dict[str, Any]:
            response = client.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=config,
            )
            self._record_usage(response, int((time.time() - t0) * 1000))
            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.strip("`").strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            return json.loads(text)

        result = self._execute_with_retry(_call)
        return result if isinstance(result, dict) else {}

    def generate_model(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> BaseModel:
        data = self.generate_model_safe(
            prompt,
            model_cls,
            system_instruction=system_instruction,
            temperature=temperature,
        )
        if isinstance(data, model_cls):
            return data
        if isinstance(data, dict):
            return model_cls(**data) if data else model_cls()
        return model_cls()

    def generate_model_safe(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> BaseModel:
        last_exc: Exception | None = None
        try:
            data = self.generate_model_raw(
                prompt,
                model_cls,
                system_instruction=system_instruction,
                temperature=temperature,
            )
            if isinstance(data, model_cls):
                return data
            return model_cls(**data) if data else model_cls()
        except LLMUnavailableError:
            # Propagate provider unavailability/504/429 immediately so AutoFallbackLLM
            # switches straight to Groq without firing a second stalled Gemini call.
            raise
        except Exception as exc:
            last_exc = exc
            log.warning("generate_model_raw parse failed, trying plain-text JSON recovery", error=str(exc)[:180])
            try:
                schema_hint = json.dumps(model_cls.model_json_schema(), indent=2)[:1500]
                plain = self.generate(
                    f"{prompt}\n\nReturn ONLY valid JSON matching this schema:\n{schema_hint}",
                    system_instruction=system_instruction,
                    temperature=temperature,
                )
                if plain:
                    parsed = self._parse_plain_text_to_dict(plain)
                    if parsed:
                        return model_cls(**parsed)
            except LLMUnavailableError:
                raise
            except Exception as exc2:
                last_exc = exc2
            if last_exc is not None:
                raise last_exc from None
            return model_cls()

    @staticmethod
    def _parse_plain_text_to_dict(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj
        except Exception:
            pass
        return {}

    def generate_model_raw(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict[str, Any]:
        if not self._clients:
            return {}

        # Pass model_cls directly to google.genai for native, valid proto schema compilation
        config = _build_generate_config(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=model_cls,
            system_instruction=system_instruction,
        )

        def _call(client: Any, t0: float) -> dict[str, Any]:
            response = client.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=config,
            )
            self._record_usage(response, int((time.time() - t0) * 1000))
            raw = (response.text or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)

        result = self._execute_with_retry(_call)
        return result if isinstance(result, dict) else {}
