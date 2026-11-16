# backend/aegis/llm/openlm_client.py
"""Modular OpenLM Provider & Gateway Integration (Sections 2 & 16).

Actual Role & Architecture Determination:
- Earlier repository artifacts (``IMPLEMENTATION_SUMMARY.md`` and ``memory/embeddings.py``)
  mistakenly listed ``openlm`` (``openlm-research/openlm-1b``) as a 2048-d
  ``SentenceTransformers`` embedding model.
- ``openlm-research`` / ``mlfoundations/open_lm`` models (``openlm-1b``, ``open_llama_3b_v2``,
  ``open_llama_7b``) are **causal decoder-only Language Models**, and OpenLM / OmniRoute
  deployments expose an **OpenAI-compatible Chat/Completions HTTP gateway** or local
  ``transformers`` CausalLM model-serving runtime.
- Accordingly, ``OpenLMClient`` integrates OpenLM in its true supported capacity as a
  modular :class:`LLMProvider` layer (supporting both OpenAI-compatible HTTP gateways via
  ``AEGIS_OPENLM_BASE_URL`` and optional local HuggingFace ``AutoModelForCausalLM``
  serving when ``AEGIS_OPENLM_BASE_URL="local"``), while retaining **Gemini 2.5 Flash**
  as AegisAI's primary LLM.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import httpx
from pydantic import BaseModel

from aegis.core.config import settings
from aegis.core.constants import DEFAULT_TEMPERATURE
from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider, LLMUnavailableError

log = get_logger(__name__)

_LOCAL_CAUSAL_CACHE: dict[str, tuple[Any, Any]] = {}
_LOCAL_LOCK = threading.Lock()


class OpenLMClient(LLMProvider):
    """Compatible LLMProvider for OpenLM / OmniRoute gateways and local OpenLM CausalLMs."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_id: str | None = None,
        timeout_seconds: int | None = None,
    ) -> None:
        self._base_url = (
            base_url if base_url is not None else getattr(settings, "AEGIS_OPENLM_BASE_URL", "")
        ).strip()
        self._api_key = (
            api_key if api_key is not None else getattr(settings, "AEGIS_OPENLM_API_KEY", "")
        ).strip()
        self._model_id = (
            model_id
            if model_id is not None
            else getattr(settings, "AEGIS_OPENLM_MODEL", "openlm-research/open_llama_7b")
        ).strip()
        self._timeout = int(
            timeout_seconds
            if timeout_seconds is not None
            else getattr(settings, "AEGIS_OPENLM_TIMEOUT_SECONDS", 30)
        )
        self._enabled = bool(getattr(settings, "AEGIS_OPENLM_ENABLED", False)) or bool(self._base_url)

    @property
    def is_configured(self) -> bool:
        return bool(self._enabled and self._base_url)

    def get_status(self) -> dict[str, Any]:
        """Return honest configuration & readiness metadata for /api/health."""
        mode = "unconfigured"
        if self._base_url.lower() == "local" or self._base_url.lower().startswith("local://"):
            mode = "local_transformers_causal_lm"
        elif self._base_url:
            mode = "openai_compatible_gateway"
        return {
            "provider": "openlm",
            "role": "llm_provider_gateway",
            "is_embedding_model": False,
            "enabled": self._enabled,
            "configured": self.is_configured,
            "mode": mode,
            "base_url": self._base_url if self._base_url else None,
            "model_id": self._model_id,
        }

    def health_check(self) -> dict[str, Any]:
        """Verify gateway connectivity when a remote URL is configured."""
        status = self.get_status()
        if not self.is_configured:
            status["reachable"] = False
            status["detail"] = "Set AEGIS_OPENLM_ENABLED=true and AEGIS_OPENLM_BASE_URL to activate OpenLM gateway."
            return status

        if status["mode"] == "local_transformers_causal_lm":
            status["reachable"] = True
            status["detail"] = "Local HuggingFace AutoModelForCausalLM mode configured."
            return status

        models_url = self._base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(models_url, headers=headers)
                status["reachable"] = resp.status_code < 500
                status["http_status"] = resp.status_code
        except Exception as exc:
            status["reachable"] = False
            status["detail"] = str(exc)[:200]
        return status

    # ── Core generation ───────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int | None = None,
    ) -> str:
        if not self.is_configured:
            raise LLMUnavailableError(
                "OpenLMClient is not configured (set AEGIS_OPENLM_BASE_URL and AEGIS_OPENLM_ENABLED=true)."
            )

        if self._base_url.lower() == "local" or self._base_url.lower().startswith("local://"):
            return self._generate_local_causal(
                prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_output_tokens or 512,
            )

        return self._generate_gateway(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def _generate_gateway(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int | None = None,
    ) -> str:
        endpoint = self._base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            if endpoint.endswith("/v1"):
                endpoint = f"{endpoint}/chat/completions"
            else:
                endpoint = f"{endpoint}/v1/chat/completions"

        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        t0 = time.time()
        try:
            with httpx.Client(timeout=float(self._timeout)) as client:
                resp = client.post(endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise LLMUnavailableError("OpenLM gateway returned empty choices")
            msg = choices[0].get("message") or {}
            content = str(msg.get("content") or choices[0].get("text") or "")
            log.info(
                "OpenLMClient gateway generate OK",
                model=self._model_id,
                latency_ms=int((time.time() - t0) * 1000),
            )
            return content
        except Exception as exc:
            raise LLMUnavailableError(f"OpenLM gateway call failed: {str(exc)[:300]}") from exc

    def _generate_local_causal(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = 512,
    ) -> str:
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            with _LOCAL_LOCK:
                if self._model_id not in _LOCAL_CAUSAL_CACHE:
                    tok = AutoTokenizer.from_pretrained(self._model_id)
                    mdl = AutoModelForCausalLM.from_pretrained(self._model_id)
                    _LOCAL_CAUSAL_CACHE[self._model_id] = (tok, mdl)
                tokenizer, model = _LOCAL_CAUSAL_CACHE[self._model_id]

            full_prompt = f"{system_instruction}\n\n{prompt}" if system_instruction else prompt
            inputs = tokenizer(full_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_output_tokens,
                    do_sample=temperature > 0.05,
                    temperature=max(temperature, 0.05),
                )
            gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
            return str(tokenizer.decode(gen_tokens, skip_special_tokens=True)).strip()
        except Exception as exc:
            raise LLMUnavailableError(f"OpenLM local CausalLM failed: {str(exc)[:300]}") from exc

    def generate_json(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        response_schema: type[Any] | None = None,
    ) -> dict[str, Any]:
        json_prompt = f"{prompt}\n\nRespond ONLY with a valid JSON object."
        raw = self.generate(
            json_prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )
        return self._extract_json_dict(raw)

    def generate_model_raw(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict[str, Any]:
        schema_str = json.dumps(model_cls.model_json_schema(), indent=2)[:1800]
        full_prompt = f"{prompt}\n\nReturn ONLY valid JSON conforming to this schema:\n{schema_str}"
        raw = self.generate(
            full_prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )
        return self._extract_json_dict(raw)

    def generate_model_safe(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> BaseModel:
        data = self.generate_model_raw(
            prompt,
            model_cls,
            system_instruction=system_instruction,
            temperature=temperature,
        )
        return model_cls(**data) if data else model_cls()

    def generate_model(
        self,
        prompt: str,
        model_cls: type[BaseModel],
        *,
        system_instruction: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> BaseModel:
        return self.generate_model_safe(
            prompt,
            model_cls,
            system_instruction=system_instruction,
            temperature=temperature,
        )

    @staticmethod
    def _extract_json_dict(text: str) -> dict[str, Any]:
        cleaned = (text or "").strip()
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

