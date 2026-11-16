# backend/aegis/core/constants.py
"""Application-wide constants.

All magic numbers, default values, and enumerated option sets live here.
No imports from other *aegis* modules are permitted — this file is the
root of the dependency graph.
"""

# ── Retry & timeout defaults ──────────────────────────────────────────────
MAX_RETRIES: int = 3
DEFAULT_TIMEOUT_SECONDS: int = 60
LLM_TIMEOUT_SECONDS: int = 120

# ── LLM ───────────────────────────────────────────────────────────────────
GEMINI_MODEL_ID: str = "gemini-2.5-flash"
# gemini-embedding-001 is the current Gemini embedding model and supports
# configurable output dimensions.
GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
DEFAULT_MAX_TOKENS: int = 8192
DEFAULT_TEMPERATURE: float = 0.7

# ── Chunking defaults (Section 4: 900 words, 10% = 90 words overlap) ──────
DEFAULT_CHUNK_SIZE_WORDS: int = 900
DEFAULT_CHUNK_OVERLAP_PERCENT: float = 0.10
DEFAULT_CHUNK_OVERLAP_WORDS: int = 90
DEFAULT_MIN_CHUNK_WORDS: int = 40
DEFAULT_MAX_CHUNK_WORDS: int = 1800

# ── Qdrant ────────────────────────────────────────────────────────────────
QDRANT_DEFAULT_COLLECTION: str = "aegis_memory"
# Default vector size matches the default embedding provider (BAAI/bge-m3 = 1024).
QDRANT_VECTOR_SIZE: int = 1024
QDRANT_RETRIEVAL_LIMIT: int = 10

# Vector sizes per embedding provider (OpenLM is an LLM provider, NOT an embedding model)
EMBEDDING_VECTOR_SIZES: dict[str, int] = {
    "gemini": 3072,
    "fastembed": 384,
    "bge-m3": 1024,
    "BAAI/bge-m3": 1024,
    "minilm": 384,
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

# ── Tavily ────────────────────────────────────────────────────────────────
TAVILY_DEFAULT_MAX_RESULTS: int = 5

# ── Project metadata ──────────────────────────────────────────────────────
PROJECT_NAME: str = "AegisAI: AgentSwarm"
PROJECT_VERSION: str = "0.1.0"

# ── Workflow agent definitions (used by Supervisor) ───────────────────────
AGENT_NAMES: list[str] = [
    "supervisor",
    "planner",
    "researcher",
    "retriever",
    "validator",
    "reasoner",
    "critic",
    "report_generator",
]

# ── Health-check defaults ─────────────────────────────────────────────────
HEALTH_CHECK_TIMEOUT_SECONDS: int = 5

