from unittest.mock import MagicMock

from aegis.agents.retriever import Retriever
from aegis.schemas.agent import RetrievedChunk, ResearchResult


class TestRetriever:
    def test_retrieve_returns_result(self, mock_llm: MagicMock):
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = []
        r = Retriever(llm=mock_llm, retrieval=mock_retrieval)
        result = r.retrieve("test query")
        assert isinstance(result, ResearchResult)

    def test_retrieve_with_qdrant_results(self, mock_llm: MagicMock):
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = [
            RetrievedChunk(
                chunk_id="1",
                document_id="doc_1",
                text="test content",
                score=0.95,
                source="test.pdf",
            )
        ]
        r = Retriever(llm=mock_llm, retrieval=mock_retrieval)
        result = r.retrieve("test query")
        assert isinstance(result, ResearchResult)
        assert "test content" in result.content
        assert result.relevance_score == 0.95
