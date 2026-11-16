# backend/aegis/langchain/embeddings.py
"""LangChain-compatible embeddings wrapper for AegisAI embedding providers."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from aegis.memory.embeddings import EmbeddingModel, get_embedding_model


class AegisEmbeddings(BaseModel, Embeddings):
    """LangChain-compatible embeddings using AegisAI's multi-model embedding engine.

    Supported providers:
    - ``bge-m3``: ``BAAI/bge-m3`` (1024 dims, default)
    - ``minilm``: ``sentence-transformers/all-MiniLM-L6-v2`` (384 dims)
    - ``fastembed``: ``BAAI/bge-small-en-v1.5`` (384 dims)
    - ``gemini``: ``models/gemini-embedding-001`` (3072 dims)
    """

    provider: str = Field(default="bge-m3", description="Embedding provider")
    model_name: str | None = Field(default=None, description="Specific HuggingFace / API model name")
    _embedder: EmbeddingModel | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._embedder = get_embedding_model(
            provider=self.provider,
            model_name=self.model_name,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        if not texts or self._embedder is None:
            return []
        return self._embedder.embed_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        if self._embedder is None:
            return []
        return self._embedder.embed(text)

    @property
    def dimensions(self) -> int:
        return self._embedder.dimensions if self._embedder else 1024

    def get_info(self) -> dict[str, Any]:
        return self._embedder.get_info() if self._embedder else {}


def create_embeddings(provider: str = "bge-m3", model_name: str | None = None) -> AegisEmbeddings:
    """Factory function to create AegisEmbeddings."""
    return AegisEmbeddings(provider=provider, model_name=model_name)
