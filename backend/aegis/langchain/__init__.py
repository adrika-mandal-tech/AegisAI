"""LangChain integration for AegisAI.

Provides LangChain-compatible wrappers for:
- Embedding models (Gemini, FastEmbed, BGE-M3, MiniLM, OpenLM)
- Retrieval (Qdrant via RetrievalService)
- Document loading
- RAG chains
"""

from __future__ import annotations

# Re-export main classes
from aegis.langchain.embeddings import AegisEmbeddings
from aegis.langchain.retriever import AegisRetriever
from aegis.langchain.chains import create_rag_chain, create_qa_chain, create_summarization_chain

__all__ = [
    "AegisEmbeddings",
    "AegisRetriever",
    "create_rag_chain",
    "create_qa_chain",
    "create_summarization_chain",
]