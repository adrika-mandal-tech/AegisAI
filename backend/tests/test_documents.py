# tests/test_documents.py
"""Tests for document upload and retrieval endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aegis.api import routes
from aegis.ingestion.pipeline import IngestionResult


@pytest.mark.integration
def test_upload_pdf(client: TestClient):
    """Verify PDF upload + processing returns success."""
    # Create a dummy PDF file
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="This is a test PDF for AegisAI.", ln=1)
    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    with open("test.pdf", "wb") as f:
        f.write(pdf_bytes)

    with open("test.pdf", "rb") as f:
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.pdf", f, "application/pdf")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "test.pdf"
    assert data["chunks"] > 0


def test_list_documents(client: TestClient):
    """Verify listing documents returns an empty list initially."""
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert isinstance(data["documents"], list)


def _find_doc_id(client: TestClient, filename: str) -> str:
    """Look up a document id from the in-memory registry by filename."""
    docs = client.get("/api/documents").json()["documents"]
    matches = [d for d in docs if d["filename"] == filename]
    assert matches, f"document {filename!r} not listed after upload"
    return matches[-1]["id"]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("filename", "mime", "content"),
    [
        ("upload_check.txt", "text/plain", "Halyard payment pipeline audit notes for Q4."),
        (
            "upload_check.md",
            "text/markdown",
            "# Findings\n\nThe Halyard payment pipeline has an open exception.",
        ),
        (
            "upload_check.csv",
            "text/csv",
            "vendor,status\nHalyard,open\nNexusGrid,closed\n",
        ),
    ],
)
def test_upload_text_formats(client: TestClient, filename: str, mime: str, content: str):
    """TXT, MD and CSV uploads ingest successfully and store extractable chunks."""
    response = client.post(
        "/api/documents/upload",
        files={"file": (filename, content.encode("utf-8"), mime)},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True, data.get("error")
    assert data["filename"] == filename
    assert data["chunks"] >= 1
    assert data["source_type"] == "file"

    doc_id = _find_doc_id(client, filename)
    chunks = client.get(f"/api/documents/{doc_id}/chunks").json()
    assert chunks["success"] is True
    assert chunks["total_chunks"] >= 1
    assert any("Halyard" in c["text"] for c in chunks["chunks"])
    assert all(c["source_type"] == "file" for c in chunks["chunks"])


def test_upload_unsupported_extension_rejected(client: TestClient):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_missing_file_rejected(client: TestClient):
    response = client.post("/api/documents/upload")
    assert response.status_code == 422  # FastAPI validation error


def test_upload_is_idempotent_for_same_filename_and_content(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ingest(**kwargs):
        result = IngestionResult()
        result.document_id = kwargs["document_id"]
        result.filename = kwargs["filename"]
        result.stored_chunks = 1
        result.status = "ready"
        return result

    monkeypatch.setattr(routes, "_document_registry", {})
    monkeypatch.setattr(routes, "_load_file_chunks", lambda *_args: ["repeatable upload content"])
    monkeypatch.setattr(routes, "ingest_document_async", ingest)

    payload = {"file": ("same.md", b"repeatable upload content", "text/markdown")}
    first = client.post("/api/documents/upload", files=payload).json()
    second = client.post("/api/documents/upload", files=payload).json()

    assert first["success"] is True
    assert second["success"] is True
    assert first["document_id"] == second["document_id"]


def test_upload_returns_failure_when_ingestion_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ingest(**kwargs):
        result = IngestionResult()
        result.document_id = kwargs["document_id"]
        result.filename = kwargs["filename"]
        result.status = "error"
        result.errors.append("embedding failed")
        return result

    monkeypatch.setattr(routes, "_document_registry", {})
    monkeypatch.setattr(routes, "_load_file_chunks", lambda *_args: ["valid content"])
    monkeypatch.setattr(routes, "ingest_document_async", ingest)

    response = client.post(
        "/api/documents/upload",
        files={"file": ("failure.md", b"valid content", "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["success"] is False
    assert response.json()["status"] == "error"
    assert "embedding failed" in response.json()["error"]


def test_upload_size_limit_is_enforced_while_staging(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes.settings, "AEGIS_DOCUMENT_MAX_BYTES", 3)

    response = client.post(
        "/api/documents/upload",
        files={"file": ("too-large.md", b"four", "text/markdown")},
    )

    assert response.status_code == 413


def test_batch_upload_reports_valid_and_invalid_files(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def ingest(**kwargs):
        result = IngestionResult()
        result.document_id = kwargs["document_id"]
        result.filename = kwargs["filename"]
        result.stored_chunks = 1
        result.status = "ready"
        return result

    monkeypatch.setattr(routes, "_document_registry", {})
    monkeypatch.setattr(routes, "_load_file_chunks", lambda *_args: ["valid batch content"])
    monkeypatch.setattr(routes, "ingest_document_async", ingest)

    response = client.post(
        "/api/documents/batch-upload",
        files=[
            ("files", ("batch.md", b"valid batch content", "text/markdown")),
            ("files", ("unsupported.exe", b"invalid", "application/octet-stream")),
        ],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data["successful"] == 1
    assert data["failed"] == 1
    results_by_filename = {item["filename"]: item for item in data["results"]}
    assert results_by_filename["batch.md"]["status"] == "ready"
    assert results_by_filename["batch.md"]["document_id"]
    assert results_by_filename["unsupported.exe"]["success"] is False


@pytest.mark.integration
def test_upload_embed_retrieve_roundtrip(client: TestClient):
    """Full pipeline: upload → chunk → embed → retrieve by unique marker."""
    marker = "zephyr-quasar-4471"
    body = (
        "# Reliability Report\n\n"
        f"The {marker} signal was detected during the overnight run. "
        "Investigators confirmed the anomaly persists across retries. "
        "The recommended action is to rotate credentials and re-baseline monitors. "
        "Historical comparisons show no prior occurrence of this signal. "
        "Therefore the finding is recorded as unique for this reporting period."
    )
    response = client.post(
        "/api/documents/upload",
        files={"file": ("reliability.md", body.encode("utf-8"), "text/markdown")},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True

    doc_id = _find_doc_id(client, "reliability.md")

    # Chunks endpoint exposes the stored text
    chunks = client.get(f"/api/documents/{doc_id}/chunks").json()
    assert any(marker in c["text"] for c in chunks["chunks"])

    # Retrieval roundtrip: the embedded marker must come back from Qdrant
    rt = client.post(
        f"/api/documents/{doc_id}/test-retrieval",
        json={"doc_id": doc_id, "query": marker, "top_k": 5, "score_threshold": 0.0},
    )
    assert rt.status_code == 200, rt.text
    rt_data = rt.json()
    assert rt_data["success"] is True, rt_data.get("error")
    assert rt_data["total_matches"] >= 1
    previews = " ".join(r["text_preview"] for r in rt_data["results"])
    assert marker in previews

    # Delete removes the document from the registry and its vectors
    deleted = client.delete(f"/api/documents/{doc_id}")
    assert deleted.status_code == 200
    assert deleted.json()["success"] is True
    after = client.get(f"/api/documents/{doc_id}/chunks").json()
    assert after.get("success") is False or after.get("total_chunks", 0) == 0
