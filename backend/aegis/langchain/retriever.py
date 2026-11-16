"""LangChain-compatible retriever for AegisAI Qdrant vector store."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from aegis.memory.retrieval import RetrievalService
from aegis.memory.embeddings import EmbeddingModel
from aegis.memory.qdrant import QdrantDB
from aegis.schemas.agent import RetrievedChunk


class AegisRetriever(BaseRetriever):
    """LangChain-compatible retriever using AegisAI's RetrievalService.

    Wraps the RetrievalService to provide LangChain's BaseRetriever interface.
    Supports all retrieval enhancements: HyDE, multi-query, dedup, rerank, etc.

    Example:
        >>> retriever = AegisRetriever(top_k=10, enable_rerank=True)
        >>> docs = retriever.invoke("search query")
    """

    top_k: int = Field(default=10, description="Number of documents to retrieve")
    score_threshold: float = Field(default=0.3, description="Minimum similarity score")
    doc_id: str | None = Field(default=None, description="Filter by document ID")
    enable_hyde: bool = Field(default=False, description="Enable HyDE query expansion")
    enable_multi_query: bool = Field(default=False, description="Enable multi-query retrieval")
    enable_dedup: bool = Field(default=True, description="Enable deduplication")
    enable_contextual_headers: bool = Field(default=False, description="Enable contextual headers")
    enable_rerank: bool = Field(default=False, description="Enable LLM reranking")
    enable_corrective: bool = Field(default=False, description="Enable corrective retrieval")

    _retrieval_service: RetrievalService | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._retrieval_service = RetrievalService()

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents for a query."""
        chunks = self._retrieval_service.retrieve(
            query,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            doc_id=self.doc_id,
            enable_hyde=self.enable_hyde,
            enable_multi_query=self.enable_multi_query,
            enable_dedup=self.enable_dedup,
            enable_contextual_headers=self.enable_contextual_headers,
            enable_rerank=self.enable_rerank,
            enable_corrective=self.enable_corrective,
        )

        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "score": chunk.score,
                    "page": chunk.page,
                    "source": chunk.source,
                    "metadata": chunk.metadata,
                },
            )
            documents.append(doc)

        return documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        """Async retrieve relevant documents."""
        # RetrievalService is sync, so just call sync version
        return self._get_relevant_documents(query, run_manager=run_manager)

    def get_retrieval_service(self) -> RetrievalService:
        """Get the underlying RetrievalService for advanced usage."""
        return self._retrieval_service


def create_retriever(
    top_k: int = 10,
    score_threshold: float = 0.3,
    doc_id: str | None = None,
    enable_hyde: bool = False,
    enable_multi_query: bool = False,
    enable_rerank: bool = False,
) -> AegisRetriever:
    """Factory function to create AegisRetriever."""
    return AegisRetriever(
        top_k=top_k,
        score_threshold=score_threshold,
        doc_id=doc_id,
        enable_hyde=enable_hyde,
        enable_multi_query=enable_multi_query,
        enable_rerank=enable_rerank,
    )