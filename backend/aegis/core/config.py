# backend/aegis/core/config.py
"""Application configuration loaded from environment variables.

Uses pydantic-settings to read from .env / shell environment.
Every field has a sensible default so the app degrades gracefully
when a key is missing (the agent will log a warning but not crash).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the AegisAI backend.

    Values are read (in order of precedence):
    1. Explicit keyword argument
    2. Environment variable
    3. .env file
    4. Hard-coded default
    """

    # ── API Keys (AEGIS_* prefix matching .env.example) ───────────────────
    # Supports both single key and comma-separated multi-key rotation
    AEGIS_GEMINI_API_KEY: str = ""
    AEGIS_GEMINI_API_KEYS: str = ""  # Comma-separated: "key1,key2,key3"
    AEGIS_TAVILY_API_KEY: str = ""
    AEGIS_LANGSMITH_API_KEY: str = ""
    AEGIS_GROQ_API_KEY: str = ""
    AEGIS_GROQ_API_KEYS: str = ""  # Comma-separated: "key1,key2,key3"
    AEGIS_GROQ_MODEL: str = ""
    AEGIS_LLM_PROVIDER: str = "auto"
    AEGIS_LLM_FALLBACK_PROVIDER: str = "groq"

    @property
    def gemini_api_keys(self) -> list[str]:
        """Return list of all available Gemini API keys.

        Checks AEGIS_GEMINI_API_KEYS (comma-separated) first,
        falls back to AEGIS_GEMINI_API_KEY (single), returns [] if empty.
        """
        raw = self.AEGIS_GEMINI_API_KEYS or self.AEGIS_GEMINI_API_KEY
        if not raw:
            return []
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        return keys

    # Backward-compatible aliases for direct access
    @property
    def GOOGLE_API_KEY(self) -> str:
        keys = self.gemini_api_keys
        return keys[0] if keys else ""

    @property
    def TAVILY_API_KEY(self) -> str:
        return self.AEGIS_TAVILY_API_KEY

    @property
    def LANGCHAIN_API_KEY(self) -> str:
        return self.AEGIS_LANGSMITH_API_KEY

    # ── App environment ───────────────────────────────────────────────────
    AEGIS_APP_ENV: str = "development"
    AEGIS_DEBUG: bool = False

    # ── Gemini ────────────────────────────────────────────────────────────
    AEGIS_GEMINI_MODEL: str = "gemini-2.5-flash"
    AEGIS_GEMINI_TIMEOUT_SECONDS: int = 20

    @property
    def GEMINI_MODEL(self) -> str:
        """Return sanitized Gemini primary model (honors .env override such as gemini-3.8-flash)."""
        candidate = (self.AEGIS_GEMINI_MODEL or "").strip()
        if not candidate or candidate.lower() in {
            "gemini-1.5-flash",
            "gemini-pro",
        }:
            return "gemini-2.5-flash"
        return candidate

    # ── OpenLM (Modular LLM Gateway / Model-Serving Provider) ─────────────
    # Note: OpenLM is an LLM provider/gateway layer, NOT an embedding model.
    AEGIS_OPENLM_ENABLED: bool = False
    AEGIS_OPENLM_BASE_URL: str = ""
    AEGIS_OPENLM_API_KEY: str = ""
    AEGIS_OPENLM_MODEL: str = "openlm-research/open_llama_7b"
    AEGIS_OPENLM_TIMEOUT_SECONDS: int = 30

    # ── Provider defaults ─────────────────────────────────────────────────
    AEGIS_PROVIDER_RETRY_ATTEMPTS: int = 2
    AEGIS_PROVIDER_TIMEOUT_SECONDS: int = 20
    AEGIS_PROVIDER_CONCURRENCY: int = 6

    # ── Embedding ─────────────────────────────────────────────────────────
    # Supported providers: "bge-m3" (BAAI/bge-m3, 1024d default) |
    # "minilm" (sentence-transformers/all-MiniLM-L6-v2, 384d) |
    # "fastembed" | "gemini".
    AEGIS_EMBEDDING_PROVIDER: str = "bge-m3"
    AEGIS_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    AEGIS_EMBEDDING_DIMENSIONS: int = 1024
    # How many texts are embedded in one forward pass.
    AEGIS_EMBEDDING_BATCH_SIZE: int = 32

    # ── OCR ───────────────────────────────────────────────────────────────
    AEGIS_OCR_ENABLED: bool = True
    # "auto" uses PaddleOCR when its runtime is available and RapidOCR
    # (the ONNX build of the same PP-OCRv6 models) otherwise.
    AEGIS_OCR_ENGINE: str = "auto"
    # Read a PDF's text layer with PyMuPDF/PyPDF first and run OCR selectively
    # only on low-text/scanned pages instead of every page.
    AEGIS_PDF_FAST_TEXT_PATH: bool = True
    AEGIS_OCR_LANGUAGES: str = "en"

    # ── Chunking (Mandatory default: 900 words, 10% = 90 words overlap) ───
    AEGIS_CHUNK_SIZE: int = 900
    AEGIS_CHUNK_OVERLAP_PERCENT: float = 0.10  # 10% dynamic overlap (= 90 words at 900)
    AEGIS_CHUNK_OVERLAP: int | None = None  # Computed dynamically when None
    AEGIS_MIN_CHUNK_SIZE: int = 40  # Minimum words for non-final chunks
    AEGIS_MAX_CHUNK_SIZE: int = 1800  # Maximum words ceiling for structural blocks
    AEGIS_CHUNK_STRATEGY: str = "document_aware"  # word | adaptive | document_aware | section_aware | heading_aware | table_aware | page_aware
    AEGIS_DEDUPE_ENABLED: bool = True
    # Legacy character settings retained for compatibility helpers
    AEGIS_CHUNK_SIZE_CHARS: int = 5400
    AEGIS_MIN_CHUNK_CHARS: int = 200

    @property
    def CHUNK_SIZE(self) -> int:
        return int(self.AEGIS_CHUNK_SIZE)

    @property
    def CHUNK_OVERLAP_PCT(self) -> int:
        pct = self.AEGIS_CHUNK_OVERLAP_PERCENT
        return int(round(pct * 100)) if pct <= 1.0 else int(round(pct))

    @property
    def CHUNK_OVERLAP(self) -> int:
        if self.AEGIS_CHUNK_OVERLAP is not None:
            return int(self.AEGIS_CHUNK_OVERLAP)
        pct = self.AEGIS_CHUNK_OVERLAP_PERCENT
        ratio = pct if pct <= 1.0 else (pct / 100.0)
        return max(1, int(round(self.AEGIS_CHUNK_SIZE * ratio)))

    # ── LangSmith ─────────────────────────────────────────────────────────
    AEGIS_LANGSMITH_TRACING: bool = False
    AEGIS_LANGSMITH_PROJECT: str = "aegisai"
    AEGIS_LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # ── Tavily ────────────────────────────────────────────────────────────
    AEGIS_TAVILY_MAX_RESULTS: int = 5

    # ── Qdrant ────────────────────────────────────────────────────────────
    AEGIS_QDRANT_URL: str = ""
    AEGIS_QDRANT_API_KEY: str = ""
    AEGIS_QDRANT_COLLECTION: str = "aegisai"
    AEGIS_QDRANT_STORAGE_PATH: str = "./data/qdrant"

    # ── Document processing ──────────────────────────────────────────────
    AEGIS_DOCUMENT_ROOT: str = "./data/documents"
    AEGIS_DOCUMENT_MAX_BYTES: int = 500 * 1024 * 1024
    # How long an upload request waits for the ingestion to finish before
    # returning and letting it continue in the background.
    AEGIS_UPLOAD_WAIT_SECONDS: float = 60.0
    AEGIS_ARTIFACT_ROOT: str = "./data/artifacts"
    AEGIS_TASK_STORE_PATH: str = "./data/tasks"

    # ── Embedding ──────────────────────────────────────────────────────
    AEGIS_EMBED_BATCH_SIZE: int = 32
    AEGIS_EMBED_CONCURRENCY: int = 4

    # ── Storage ────────────────────────────────────────────────────────
    AEGIS_QDRANT_COLLECTION_STRATEGY: str = "shared"  # "shared", "per_doc"

    # ── Application metadata ──────────────────────────────────────────────
    APP_NAME: str = "AegisAI: AgentSwarm"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
