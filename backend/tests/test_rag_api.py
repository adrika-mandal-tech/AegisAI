"""Tests for the minimal RAG API layer: /upload, /query, /health.

The upload test marked ``integration`` exercises the real ingestion
pipeline (parse → chunk → embed → Qdrant). Query tests mock the
retrieval/LLM boundaries so they are deterministic and quota-free.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from aegis.api import rag
from aegis.schemas.agent import RetrievedChunk


class TestHealth:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"]
        assert data["environment"]


class TestUpload:
    def test_rejects_unsupported_extension(self, client: TestClient) -> None:
        response = client.post(
            "/upload",
            files={"file": ("malware.exe", b"MZ", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_rejects_missing_file(self, client: TestClient) -> None:
        response = client.post("/upload")
        assert response.status_code == 422  # FastAPI validation: file required

    @pytest.mark.integration
    def test_upload_md_runs_full_ingestion(self, client: TestClient) -> None:
        """Live path: parse → chunk → embed → Qdrant."""
        content = (
            "# RAG API ingestion test\n\n"
            "Project Aegis launch code: 4-8-15-16-23-42.\n"
        ).encode()
        response = client.post(
            "/upload",
            files={"file": ("rag_api_test.md", content, "text/markdown")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "rag_api_test.md"
        assert data["chunks"] >= 1
        assert data["document_id"]

        # Uploaded doc is visible in the shared document registry.
        listing = client.get("/api/documents").json()
        assert any(d["filename"] == "rag_api_test.md" for d in listing["documents"])


class TestQuery:
    def _fake_chunk(self) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            text="The launch code is 4-8-15.",
            score=0.91,
            source="rag_api_test.md",
            page=2,
            chunk_index=3,
        )

    def test_full_flow_with_mocked_boundaries(self, client: TestClient, monkeypatch) -> None:
        retrieval = MagicMock()
        retrieval.retrieve.return_value = [self._fake_chunk()]
        llm = MagicMock()
        llm.generate.return_value = "According to [1], the code is 4-8-15."
        monkeypatch.setattr(rag, "_get_retrieval", lambda: retrieval)
        monkeypatch.setattr(rag, "_get_llm", lambda: llm)

        response = client.post(
            "/query",
            json={"query": "What is the launch code?", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "4-8-15" in data["answer"]
        assert data["chunks_retrieved"] == 1
        assert data["sources"][0]["source"] == "rag_api_test.md"
        assert data["sources"][0]["score"] == pytest.approx(0.91)
        assert data["sources"][0]["page"] == 2
        assert data["sources"][0]["chunk_index"] == 3

        # Retrieval called with the query; LLM called with grounded context.
        retrieval.retrieve.assert_called_once()
        assert retrieval.retrieve.call_args.args[0] == "What is the launch code?"
        prompt = llm.generate.call_args.args[0]
        assert "4-8-15" in prompt  # retrieved context present in prompt
        assert "What is the launch code?" in prompt
        assert "page=2" in prompt

    def test_doc_id_filter_is_forwarded(self, client: TestClient, monkeypatch) -> None:
        retrieval = MagicMock()
        retrieval.retrieve.return_value = []
        monkeypatch.setattr(rag, "_get_retrieval", lambda: retrieval)

        response = client.post(
            "/query",
            json={"query": "hello", "doc_id": "doc-123"},
        )
        assert response.status_code == 200
        assert response.json()["chunks_retrieved"] == 0
        assert retrieval.retrieve.call_args.kwargs["doc_id"] == "doc-123"

    def test_empty_context_returns_graceful_answer(self, client: TestClient, monkeypatch) -> None:
        retrieval = MagicMock()
        retrieval.retrieve.return_value = []
        monkeypatch.setattr(rag, "_get_retrieval", lambda: retrieval)

        response = client.post("/query", json={"query": "anything"})
        data = response.json()
        assert data["success"] is True
        assert data["chunks_retrieved"] == 0
        assert "No relevant documents" in data["answer"]

    def test_llm_failure_returns_error_with_sources(self, client: TestClient, monkeypatch) -> None:
        retrieval = MagicMock()
        retrieval.retrieve.return_value = [self._fake_chunk()]
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("quota exhausted")
        monkeypatch.setattr(rag, "_get_retrieval", lambda: retrieval)
        monkeypatch.setattr(rag, "_get_llm", lambda: llm)

        response = client.post("/query", json={"query": "q"})
        data = response.json()
        assert data["success"] is False
        assert "LLM generation failed" in data["error"]
        assert data["chunks_retrieved"] == 1  # sources still returned
        assert len(data["sources"]) == 1

    def test_retrieval_failure_returns_error(self, client: TestClient, monkeypatch) -> None:
        retrieval = MagicMock()
        retrieval.retrieve.side_effect = RuntimeError("qdrant down")
        monkeypatch.setattr(rag, "_get_retrieval", lambda: retrieval)

        response = client.post("/query", json={"query": "q"})
        data = response.json()
        assert data["success"] is False
        assert "Retrieval failed" in data["error"]

    def test_validation_rejects_empty_query(self, client: TestClient) -> None:
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

    @pytest.mark.integration
    def test_live_end_to_end(self, client: TestClient) -> None:
        """Live RAG: embed query → Qdrant → LLM → grounded answer."""
        response = client.post(
            "/query",
            json={"query": "What is the Project Aegis launch code?", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True, data.get("error")
        assert data["chunks_retrieved"] >= 1
        assert data["answer"]
        assert data["sources"]