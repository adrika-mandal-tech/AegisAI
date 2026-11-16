# backend/aegis/llm/groq_client.py
"""Groq LLM provider client with OpenAI-compatible structured output.

Groq free tier supports JSON mode via ``response_format`` and requires
no credit card.  This client implements the same ``LLMProvider`` interface
as ``GeminiClient`` so it can be swapped in without changing agents.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

from groq import Groq, APIError

from aegis.core.config import settings
from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider
from aegis.llm.models import GROQ_MODELS
from pydantic import BaseModel

log = get_logger(__name__)

DEFAULT_GROQ_MODEL = GROQ_MODELS["pro"]


def _strip_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return schema
    result = {}
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


class GroqClient(LLMProvider):
    """Thread-safe wrapper around ``groq.Groq`` with retry/backoff."""

    def __init__(self, model_id: str | None = None) -> None:
        env_model = getattr(settings, "AEGIS_GROQ_MODEL", "")
        candidate = model_id or env_model or DEFAULT_GROQ_MODEL
        if candidate.lower().startswith("gemini"):
            log.warning(
                "GroqClient received a Gemini model id '%s'; falling back to '%s'.",
                candidate,
                DEFAULT_GROQ_MODEL,
            )
            candidate = DEFAULT_GROQ_MODEL
        self._model_id = candidate
        self._client = self._init_client()
        log.info("GroqClient initialised", model_id=self._model_id)

    def _init_client(self) -> Groq | None:
        api_keys = getattr(settings, "AEGIS_GROQ_API_KEYS", "") or ""
        keys = [k.strip() for k in api_keys.split(",") if k.strip()]
        if not keys:
            single = getattr(settings, "AEGIS_GROQ_API_KEY", "") or ""
            if single:
                keys = [single]
        if not keys:
            log.warning(
                "No Groq API key configured — dry-run mode. "
                "Set AEGIS_GROQ_API_KEY or AEGIS_GROQ_API_KEYS in your .env file."
            )
            return None
        self._api_keys = keys
        self._key_index = random.randrange(len(keys))
        self._client = Groq(api_key=keys[self._key_index], timeout=15.0)
        log.info("GroqClient initialised", model_id=self._model_id, key_count=len(keys))
        return self._client

    def _rotate_key(self) -> None:
        """Rotate to the next Groq API key."""
        if not getattr(self, "_api_keys", None):
            return
        self._key_index = (self._key_index + 1) % len(self._api_keys)
        key = self._api_keys[self._key_index]
        try:
            self._client = Groq(api_key=key, timeout=15.0)
            log.info("Rotated Groq API key", index=self._key_index, total=len(self._api_keys))
        except Exception:
            self._client = None

    def _execute_with_retry(self, call_fn, *args, **kwargs) -> Any:
        if not self._client:
            log.warning("GroqClient dry-run — returning placeholder.")
            return None

        max_retries = min(max(settings.AEGIS_PROVIDER_RETRY_ATTEMPTS, 2), 3)

        for attempt in range(max_retries):
            try:
                return call_fn(self._client, *args, **kwargs)
            except APIError as e:
                status = getattr(e, "status_code", None)
                err_text = str(e).lower()
                if status in (400, 404) and ("model" in err_text or "decommissioned" in err_text or "not found" in err_text):
                    alt = "openai/gpt-oss-20b" if self._model_id != "openai/gpt-oss-20b" else "openai/gpt-oss-120b"
                    log.warning("Groq model unavailable, switching model", old=self._model_id, new=alt)
                    self._model_id = alt
                    continue
                if status in (429, 500, 502, 503, 504):
                    self._rotate_key()
                    wait_time = 0.2 * (attempt + 1)
                    log.warning(
                        f"Groq API error ({status}), rotating key and retrying in {wait_time:.2f}s",
                        attempt=attempt + 1,
                        model=self._model_id,
                        key_index=getattr(self, "_key_index", -1),
                    )
                    if attempt >= max_retries - 1:
                        log.error("All Groq retries exhausted.")
                        return None
                    time.sleep(wait_time)
                    continue
                log.warning("Unhandled Groq API error", error=str(e)[:200])
                return None
            except Exception:
                log.exception("Groq API call failed")
                return None

        return None

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        max_output_tokens: int | None = None,
    ) -> str:
        if not self._client:
            return "[dry-run] Groq response placeholder"

        def _call(client: Groq):
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "messages": messages,
                "temperature": temperature,
            }
            if max_output_tokens:
                kwargs["max_tokens"] = max_output_tokens

            response = client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()

        result = self._execute_with_retry(_call)
        return result if isinstance(result, str) else "[dry-run] Groq response placeholder"

    def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
        response_schema: type[Any] | None = None,
    ) -> dict[str, Any]:
        if not self._client:
            return {}

        def _call(client: Groq):
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }

            response = client.chat.completions.create(**kwargs)
            raw = (response.choices[0].message.content or "").strip()
            text = raw.strip()
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
        temperature: float = 0.7,
    ) -> BaseModel:
        data = self.generate_model_raw(
            prompt,
            model_cls,
            system_instruction=system_instruction,
            temperature=temperature,
        )
        return model_cls(**data) if data else model_cls()

    def generate_model_raw(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        if not self._client:
            return {}

        raw_schema = model_cls.model_json_schema()
        cleaned = _strip_additional_properties(raw_schema)
        schema_hint = json.dumps(cleaned)

        def _call(client: Groq):
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    f"Return ONLY valid JSON data matching this schema. Do NOT return the schema itself.\n"
                    f"Schema: {schema_hint}"
                ),
            })

            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }

            response = client.chat.completions.create(**kwargs)
            raw = (response.choices[0].message.content or "").strip()
            text = raw.strip()
            if text.startswith("```"):
                text = text.strip("`").strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            return json.loads(text)

        result = self._execute_with_retry(_call)
        return result if isinstance(result, dict) else {}

    def generate_model_safe(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = 0.7,
    ) -> BaseModel:
        """Like ``generate_model`` but falls back to plain text on JSON failure."""
        try:
            data = self.generate_model_raw(
                prompt,
                model_cls,
                system_instruction=system_instruction,
                temperature=temperature,
            )
            return model_cls(**data) if data else model_cls()
        except Exception as exc:
            log.warning("generate_model_raw failed, falling back to plain text", error=str(exc)[:200])
            try:
                plain = self.generate(
                    prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                )
                if plain:
                    import re
                    json_match = re.search(r"\{.*\}", plain, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        return model_cls(**data) if data else model_cls()
            except Exception as exc2:
                log.warning("Plain text fallback also failed", error=str(exc2)[:200])
            return model_cls()
