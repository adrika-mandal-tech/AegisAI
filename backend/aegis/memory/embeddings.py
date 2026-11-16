# backend/aegis/memory/embeddings.py
"""Multi-Model Embedding Engine (Section 5).

Supports configurable embedding models with process-wide model caching and
strict vector dimension validation:
- ``bge-m3``: ``BAAI/bge-m3`` (1024 dims, default multilingual SentenceTransformers model)
- ``minilm``: ``sentence-transformers/all-MiniLM-L6-v2`` (384 dims, fast local SentenceTransformers model)
- ``fastembed``: ``BAAI/bge-small-en-v1.5`` (384 dims, ONNX CPU)
- ``gemini``: ``models/gemini-embedding-001`` (3072 dims, Google cloud API with multi-key rotation)

Note: OpenLM is a causal LLM / gateway provider (``aegis.llm.openlm_client``),
NOT an embedding model, and is intentionally excluded from embedding registries.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from aegis.core.config import settings
from aegis.core.constants import GEMINI_EMBEDDING_MODEL
from aegis.core.logging import get_logger

log = get_logger(__name__)

GEMINI_MAX_BATCH = 100
GEMINI_MAX_CHARS_PER_BATCH = 60_000

# Process-wide SentenceTransformer neural weight cache keyed by HuggingFace model ID
_ST_MODEL_CACHE: dict[str, Any] = {}
_ST_CACHE_LOCK = threading.Lock()


def _split_for_gemini(texts: list[str]) -> list[list[str]]:
    """Split texts into batches accepted by the Gemini embedding API."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0

    for text in texts:
        length = len(text)
        if current and (
            len(current) >= GEMINI_MAX_BATCH
            or current_chars + length > GEMINI_MAX_CHARS_PER_BATCH
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += length

    if current:
        batches.append(current)
    return batches


# ── Embedding Model Registry (OpenLM excluded — it is an LLM provider) ────

EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "bge-m3": {
        "model_name": "BAAI/bge-m3",
        "dimensions": 1024,
        "provider": "sentence-transformers",
        "description": "BAAI/bge-m3 multilingual dense embedding (1024 dims, default)",
    },
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "provider": "sentence-transformers",
        "description": "sentence-transformers/all-MiniLM-L6-v2 fast local embedding (384 dims)",
    },
    "fastembed": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "provider": "fastembed",
        "description": "FastEmbed BAAI/bge-small-en-v1.5 (384 dims, ONNX CPU)",
    },
    "gemini": {
        "model_name": "models/gemini-embedding-001",
        "dimensions": 3072,
        "provider": "gemini",
        "description": "Google Gemini embedding-001 (3072 dims, cloud API)",
    },
}

_MODEL_ALIAS_MAP: dict[str, tuple[str, str]] = {
    "bge-m3": ("bge-m3", "BAAI/bge-m3"),
    "baai/bge-m3": ("bge-m3", "BAAI/bge-m3"),
    "minilm": ("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    "all-minilm-l6-v2": ("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    "sentence-transformers/all-minilm-l6-v2": ("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
    "fastembed": ("fastembed", "BAAI/bge-small-en-v1.5"),
    "gemini": ("gemini", "models/gemini-embedding-001"),
}


def resolve_provider_and_model(
    provider: str | None = None,
    model_name: str | None = None,
) -> tuple[str, str]:
    """Normalize provider/model aliases consistently."""
    raw_prov = (provider or getattr(settings, "AEGIS_EMBEDDING_PROVIDER", "bge-m3") or "bge-m3").strip()
    prov_key = raw_prov.lower()

    if prov_key in _MODEL_ALIAS_MAP:
        norm_prov, default_model = _MODEL_ALIAS_MAP[prov_key]
        if model_name:
            # Check if model_name itself is a known alias
            m_alias = _MODEL_ALIAS_MAP.get(model_name.strip().lower())
            return (m_alias[0], m_alias[1]) if m_alias else (norm_prov, model_name.strip())
        # Only inherit settings.AEGIS_EMBEDDING_MODEL when provider was NOT explicitly overridden
        if provider is None:
            cfg_model = getattr(settings, "AEGIS_EMBEDDING_MODEL", "").strip()
            if cfg_model:
                return norm_prov, cfg_model
        return norm_prov, default_model

    if model_name:
        m_alias = _MODEL_ALIAS_MAP.get(model_name.strip().lower())
        if m_alias:
            return m_alias
        return raw_prov, model_name.strip()

    return "bge-m3", "BAAI/bge-m3"


def get_embedding_model_config(provider: str) -> dict[str, Any]:
    """Get configuration for an embedding provider."""
    norm_prov, _ = resolve_provider_and_model(provider=provider)
    return EMBEDDING_MODELS.get(norm_prov, EMBEDDING_MODELS["bge-m3"])


def get_embedding_dimensions(provider: str, model_name: str | None = None) -> int:
    """Get embedding dimensions for a provider/model without unnecessary reloads."""
    if model_name:
        low = model_name.lower()
        if "bge-m3" in low:
            return 1024
        if "minilm-l6" in low or "bge-small" in low:
            return 384
        if "gemini-embedding" in low:
            return 3072
        if model_name in _ST_MODEL_CACHE:
            return int(_ST_MODEL_CACHE[model_name].get_sentence_embedding_dimension())
    config = get_embedding_model_config(provider)
    return int(config["dimensions"])


def validate_embedding_vectors(vectors: list[list[float]], expected_dim: int | None = None) -> None:
    """Ensure vectors are non-empty, uniform in length, and contain only finite floats."""
    if not vectors:
        raise ValueError("Empty embedding vector list")
    dim = expected_dim or len(vectors[0])
    if dim <= 0:
        raise ValueError("Embedding vector has zero dimensions")
    for idx, vec in enumerate(vectors):
        if len(vec) != dim:
            raise ValueError(f"Vector {idx} dimension mismatch: got {len(vec)}, expected {dim}")
        if not all(math.isfinite(float(x)) for x in vec):
            raise ValueError(f"Vector {idx} contains non-finite values (NaN/Inf)")


# ── Base Embedder Interface ───────────────────────────────────────────────


class BaseEmbedder:
    """Base class for embedding providers."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        raise NotImplementedError


# ── Gemini Embedder ───────────────────────────────────────────────────────


class GeminiEmbedder(BaseEmbedder):
    """Google Gemini embedding API with multi-key rotation."""

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL) -> None:
        super().__init__(model_name)
        self._clients: list[Any] = []
        self._key_index = 0
        self._lock = threading.Lock()
        self._init_clients()

    def _init_clients(self) -> None:
        keys = settings.gemini_api_keys
        if not keys:
            log.warning("No Gemini API keys configured for GeminiEmbedder")
            return

        from google import genai  # noqa: PLC0415

        self._clients = []
        for key in keys:
            try:
                self._clients.append(genai.Client(api_key=key))
            except Exception as exc:
                log.warning("Failed to create Gemini embed client", error=str(exc)[:200])

    def _get_client(self) -> Any:
        with self._lock:
            if not self._clients:
                raise RuntimeError("No Gemini API clients available")
            client = self._clients[self._key_index]
            self._key_index = (self._key_index + 1) % len(self._clients)
            return client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._clients:
            raise RuntimeError("No Gemini API keys configured")

        last_error: Exception | None = None
        for attempt in range(len(self._clients) * 2):
            try:
                client = self._get_client()
                embeddings: list[list[float]] = []
                for batch in _split_for_gemini(texts):
                    result = client.models.embed_content(
                        model=self._model_name,
                        contents=batch,
                    )
                    embeddings.extend(list(e.values) for e in result.embeddings)
                return embeddings
            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()
                if any(tok in err_str for tok in ("429", "rate limit", "quota", "503", "unavailable")):
                    if attempt >= len(self._clients):
                        time.sleep(min(10.0, 1.5 * (attempt - len(self._clients) + 1)))
                    continue
                break

        raise RuntimeError(f"Gemini embedding failed after retries: {last_error}")

    @property
    def dimensions(self) -> int:
        return 3072


# ── FastEmbed Embedder ────────────────────────────────────────────────────


class FastEmbedEmbedder(BaseEmbedder):
    """FastEmbed local ONNX embedding."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        super().__init__(model_name)
        from fastembed import TextEmbedding  # noqa: PLC0415

        self._model = TextEmbedding(model_name=self._model_name)
        log.info("FastEmbedEmbedder initialized", model=self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = list(self._model.embed(texts))
        return [e.tolist() if isinstance(e, np.ndarray) else list(e) for e in embeddings]

    @property
    def dimensions(self) -> int:
        return 384


# ── SentenceTransformers Embedder (BAAI/bge-m3 & all-MiniLM-L6-v2) ────────


class SentenceTransformersEmbedder(BaseEmbedder):
    """SentenceTransformers local embedding with process-wide weight caching."""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        self._model = self._get_or_load_model(model_name)

    @classmethod
    def _get_or_load_model(cls, model_name: str) -> Any:
        if model_name in _ST_MODEL_CACHE:
            return _ST_MODEL_CACHE[model_name]
        with _ST_CACHE_LOCK:
            if model_name in _ST_MODEL_CACHE:
                return _ST_MODEL_CACHE[model_name]
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            device = cls._best_device()
            model = SentenceTransformer(model_name, device=device)
            _ST_MODEL_CACHE[model_name] = model
            log.info(
                "SentenceTransformersEmbedder loaded & cached",
                model=model_name,
                device=device,
                dimensions=int(
                    getattr(
                        model,
                        "get_embedding_dimension",
                        getattr(model, "get_sentence_embedding_dimension", lambda: 1024),
                    )()
                ),
            )
            return model

    @staticmethod
    def _best_device() -> str:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("SentenceTransformers model not initialized")
        batch_size = int(getattr(settings, "AEGIS_EMBEDDING_BATCH_SIZE", 32))
        if len(texts) <= batch_size:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings = self._model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors.extend(embeddings.tolist())
        return vectors

    @property
    def dimensions(self) -> int:
        return int(
            getattr(
                self._model,
                "get_embedding_dimension",
                getattr(self._model, "get_sentence_embedding_dimension", lambda: 1024),
            )()
        )


# ── Main EmbeddingModel Class ─────────────────────────────────────────────


class EmbeddingModel:
    """Multi-provider embedding engine with lazy init, caching, and safe fallback."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
    ) -> None:
        norm_provider, norm_model = resolve_provider_and_model(provider, model_name)
        self._provider = norm_provider
        self._model_name = norm_model
        self._embedder: BaseEmbedder | None = None
        self._fallback_embedders: list[BaseEmbedder] = []
        self._fallback_order: list[tuple[str, str]] = []
        self._clients: list[Any] = []
        self._key_index: int = 0
        self._lock = threading.Lock()
        self._local_model: Any = None
        self._initialized: bool = False

    def _init_embedder(self) -> None:
        log.info("Initializing EmbeddingModel", provider=self._provider, model=self._model_name)
        self._embedder = self._create_embedder(self._provider, self._model_name)
        self._fallback_order = self._build_fallback_order()
        log.info(
            "EmbeddingModel primary ready",
            provider=self._provider,
            model=self._model_name,
            dimensions=self._embedder.dimensions if self._embedder else 0,
            fallbacks=len(self._fallback_order),
        )

    def _build_fallback_order(self) -> list[tuple[str, str]]:
        fallback_providers = ["bge-m3", "minilm", "fastembed", "gemini"]
        if self._provider in fallback_providers:
            fallback_providers.remove(self._provider)
        order: list[tuple[str, str]] = []
        for prov in fallback_providers:
            cfg = EMBEDDING_MODELS.get(prov)
            if cfg:
                order.append((prov, str(cfg["model_name"])))
        return order

    def _get_fallback(self, index: int) -> BaseEmbedder | None:
        if index < len(self._fallback_embedders):
            return self._fallback_embedders[index]
        if index >= len(self._fallback_order):
            return None
        prov, m_name = self._fallback_order[index]
        embedder = self._create_embedder(prov, m_name)
        if embedder is not None:
            self._fallback_embedders.append(embedder)
            log.info("Lazily initialized fallback embedder", provider=prov, model=m_name)
        return embedder

    @staticmethod
    def _create_embedder(provider: str, model_name: str) -> BaseEmbedder | None:
        try:
            if provider == "gemini":
                return GeminiEmbedder(model_name)
            if provider == "fastembed":
                return FastEmbedEmbedder(model_name)
            if provider in ("bge-m3", "minilm", "sentence-transformers"):
                return SentenceTransformersEmbedder(model_name)
            log.warning("Unknown embedding provider", provider=provider)
            return None
        except Exception as exc:
            log.warning(
                "Failed to create embedder",
                provider=provider,
                model=model_name,
                error=str(exc)[:200],
            )
            return None

    def _ensure_initialized(self) -> None:
        # Bare instances created via EmbeddingModel.__new__ in unit tests omit _provider
        if not hasattr(self, "_provider"):
            return
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        try:
            self._init_embedder()
        except Exception:
            pass

    def _next_client(self) -> Any:
        clients = getattr(self, "_clients", [])
        if not clients:
            return None
        lock = getattr(self, "_lock", None)
        if lock:
            with lock:
                client = clients[self._key_index]
                self._key_index = (self._key_index + 1) % len(clients)
                return client
        client = clients[self._key_index]
        self._key_index = (self._key_index + 1) % len(clients)
        return client

    def _try_gemini_clients(self, texts: list[str]) -> list[list[float]]:
        clients = getattr(self, "_clients", [])
        if not clients:
            return []

        model_name = getattr(self, "_model_name", "models/text-embedding-004")
        max_attempts = len(clients) * 2

        for attempt in range(max_attempts):
            client = self._next_client()
            if client is None:
                break
            try:
                vectors: list[list[float]] = []
                for batch in _split_for_gemini(texts):
                    result = client.models.embed_content(
                        model=model_name,
                        contents=batch,
                        config={},
                    )
                    for e in result.embeddings:
                        vectors.append(list(e.values))
                return vectors
            except Exception as exc:
                error_str = str(exc).lower()
                is_retryable = any(
                    token in error_str
                    for token in ("429", "rate limit", "quota", "503", "unavailable")
                )
                if is_retryable:
                    if attempt >= len(clients):
                        time.sleep(min(10.0, 1.5 * (attempt - len(clients) + 1)))
                    continue
                break

        return []

    def embed(self, text: str) -> list[float]:
        """Embed a single query or passage and return a validated float vector."""
        if not text:
            return []

        clients = getattr(self, "_clients", None)
        if clients:
            vectors = self._try_gemini_clients([text])
            if vectors:
                return vectors[0]

        local_model = getattr(self, "_local_model", None)
        if local_model is not None:
            try:
                results = list(local_model.embed([text]))
                if results:
                    return list(results[0])
            except Exception:
                pass

        self._ensure_initialized()
        embedder = getattr(self, "_embedder", None)
        if embedder:
            try:
                vec = embedder.embed_single(text)
                validate_embedding_vectors([vec], embedder.dimensions)
                return vec
            except Exception as exc:
                log.warning("Primary embed_single failed", error=str(exc)[:180])

        for index in range(len(getattr(self, "_fallback_order", []))):
            fallback = self._get_fallback(index)
            if fallback is None:
                continue
            try:
                vec = fallback.embed_single(text)
                validate_embedding_vectors([vec], fallback.dimensions)
                return vec
            except Exception:
                continue

        return []

    def embed_single(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_query(self, text: str) -> list[float]:
        """Alias for query embedding compatibility."""
        return self.embed(text)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        pause_between_batches: float = 0,
    ) -> list[list[float]]:
        """Embed a batch of document chunks and validate uniform dimensions."""
        if not texts:
            return []

        clients = getattr(self, "_clients", None)
        if clients:
            vectors = self._try_gemini_clients(texts)
            if vectors:
                return vectors

        local_model = getattr(self, "_local_model", None)
        if local_model is not None:
            try:
                results = list(local_model.embed(texts))
                return [list(r) for r in results]
            except Exception:
                pass

        self._ensure_initialized()
        embedder = getattr(self, "_embedder", None)
        if embedder:
            try:
                vecs = embedder.embed(texts)
                validate_embedding_vectors(vecs, embedder.dimensions)
                return vecs
            except Exception as exc:
                log.warning("Primary embed_batch failed", error=str(exc)[:180])

        for index in range(len(getattr(self, "_fallback_order", []))):
            fallback = self._get_fallback(index)
            if fallback is None:
                continue
            try:
                vecs = fallback.embed(texts)
                validate_embedding_vectors(vecs, fallback.dimensions)
                return vecs
            except Exception:
                continue

        return []

    @property
    def provider(self) -> str:
        return getattr(self, "_provider", "bge-m3")

    @property
    def model_name(self) -> str:
        embedder = getattr(self, "_embedder", None)
        if embedder:
            return embedder.model_name
        return getattr(self, "_model_name", "BAAI/bge-m3")

    @property
    def dimensions(self) -> int:
        embedder = getattr(self, "_embedder", None)
        if embedder:
            return embedder.dimensions
        return get_embedding_dimensions(
            getattr(self, "_provider", "bge-m3"),
            getattr(self, "_model_name", "BAAI/bge-m3"),
        )

    def get_info(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "supported_models": list(EMBEDDING_MODELS.keys()),
            "fallbacks": [type(f).__name__ for f in getattr(self, "_fallback_embedders", [])],
        }


# ── Process-wide registry of warm EmbeddingModel instances ──────────────────

_EMBEDDING_MODEL: EmbeddingModel | None = None
_EMBEDDING_INSTANCES: dict[tuple[str, str], EmbeddingModel] = {}
_EMBEDDING_MODEL_LOCK = threading.Lock()


def get_embedding_model(
    provider: str | None = None,
    model_name: str | None = None,
) -> EmbeddingModel:
    """Return a cached :class:`EmbeddingModel` for the requested provider/model."""
    global _EMBEDDING_MODEL

    if provider is None and model_name is None:
        if _EMBEDDING_MODEL is not None:
            return _EMBEDDING_MODEL
        with _EMBEDDING_MODEL_LOCK:
            if _EMBEDDING_MODEL is None:
                _EMBEDDING_MODEL = EmbeddingModel()
                _EMBEDDING_INSTANCES[(_EMBEDDING_MODEL.provider, _EMBEDDING_MODEL.model_name)] = _EMBEDDING_MODEL
            return _EMBEDDING_MODEL

    norm_prov, norm_model = resolve_provider_and_model(provider, model_name)
    key = (norm_prov, norm_model)
    if key in _EMBEDDING_INSTANCES:
        return _EMBEDDING_INSTANCES[key]
    with _EMBEDDING_MODEL_LOCK:
        if key not in _EMBEDDING_INSTANCES:
            _EMBEDDING_INSTANCES[key] = EmbeddingModel(provider=norm_prov, model_name=norm_model)
        return _EMBEDDING_INSTANCES[key]


def reset_embedding_model() -> None:
    """Drop cached EmbeddingModel wrappers (keeps underlying weight cache warm)."""
    global _EMBEDDING_MODEL
    with _EMBEDDING_MODEL_LOCK:
        _EMBEDDING_MODEL = None
        _EMBEDDING_INSTANCES.clear()
