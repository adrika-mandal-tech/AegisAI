# backend/tests/test_ingestion.py
# """Ingestion pipeline tests: URL, git repo, file sources, and stage wiring."""
from __future__ import annotations

import functools
import http.server
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from aegis.tools.multi_loader import (
    _sanitize_repo_url,
    load_from_source,
    load_git_repo,
    load_url,
)


# ── Local HTTP server fixture (deterministic, no external network) ──────────


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence request logging
        pass


@pytest.fixture(scope="module")
def local_url_server(tmp_path_factory):
    """Serve a temp directory over HTTP on an ephemeral port."""
    directory = tmp_path_factory.mktemp("urlsrv")
    (directory / "bulletin.md").write_text(
        "# Security Bulletin\n\nCRITICAL: quokka-token-93017 SQL injection. "
        "Patch the vendor gateway immediately.\n",
        encoding="utf-8",
    )
    handler = functools.partial(_QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


# ── Request validation (kept from the original suite) ───────────────────────


class TestRequestValidation:
    def test_ingest_url_missing_source(self, client: TestClient):
        response = client.post("/api/documents/ingest", json={"source_type": "url"})
        assert response.status_code in (400, 422)

    def test_ingest_url_invalid_source_type(self, client: TestClient):
        response = client.post(
            "/api/documents/ingest",
            json={"source": "https://example.com", "source_type": "invalid"},
        )
        assert response.status_code in (400, 422)

    def test_ingest_repo_missing_source(self, client: TestClient):
        response = client.post("/api/documents/ingest", json={"source_type": "repo"})
        assert response.status_code in (400, 422)


# ── load_url ─────────────────────────────────────────────────────────────────


class TestLoadUrl:
    def test_fetches_and_extracts_text(self, local_url_server: str):
        chunks = load_url(f"{local_url_server}/bulletin.md")
        assert chunks, "expected at least one chunk"
        joined = " ".join(chunks)
        assert "quokka-token-93017" in joined
        assert "Security Bulletin" in joined

    def test_unreachable_url_returns_empty(self):
        assert load_url("http://127.0.0.1:9/missing.md") == []

    def test_non_http_url_returns_empty(self):
        assert load_url("ftp://example.com/file.md") == []


# ── load_git_repo (subprocess mocked for determinism) ───────────────────────


class TestLoadGitRepo:
    def test_clone_then_load_supported_files(self, monkeypatch):
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            tmpdir = Path(cmd[-1])
            (tmpdir / "README.md").write_text(
                "# MockRepo\n\nGadgetFlip provisioning guide for staging clusters.\n",
                encoding="utf-8",
            )
            (tmpdir / "skip.js").write_text("not a supported extension", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr("subprocess.run", fake_run)
        chunks = load_git_repo("https://github.com/acme/mockrepo?utm=1")
        joined = " ".join(chunks)
        assert "GadgetFlip" in joined
        assert "not a supported extension" not in joined
        assert calls and calls[0][0] == "git"
        assert "--depth" in calls[0]
        # tracking query params stripped before cloning
        assert "utm=1" not in " ".join(calls[0])

    def test_clone_failure_returns_empty(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(128, cmd, stderr=b"repository not found")

        monkeypatch.setattr("subprocess.run", fake_run)
        assert load_git_repo("https://github.com/acme/does-not-exist") == []

    def test_repo_with_no_supported_files_returns_empty(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            Path(cmd[-1], "main.js").write_text("const x = 1;", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr("subprocess.run", fake_run)
        assert load_git_repo("https://github.com/acme/js-only") == []


class TestSanitizeRepoUrl:
    def test_strips_query_and_fragment(self):
        assert (
            _sanitize_repo_url("https://github.com/acme/repo?tab=readme-ov-file#top")
            == "https://github.com/acme/repo"
        )

    @pytest.mark.parametrize("bad", ["ftp://x/y", "not a url", "file:///etc", ""])
    def test_rejects_non_http_sources(self, bad: str):
        assert _sanitize_repo_url(bad) is None


# ── load_from_source dispatcher ─────────────────────────────────────────────


class TestLoadFromSource:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert load_from_source(str(tmp_path / "nope.txt")) == []

    def test_txt_file_dispatch(self, tmp_path: Path):
        target = tmp_path / "notes.txt"
        target.write_text("Larkspur financial ledger reconciliation notes.", encoding="utf-8")
        chunks = load_from_source(str(target))
        assert chunks and "Larkspur" in chunks[0]

    def test_url_dispatch(self, local_url_server: str):
        chunks = load_from_source(f"{local_url_server}/bulletin.md", source_type="url")
        assert chunks and "quokka-token-93017" in " ".join(chunks)

    def test_repo_dispatch_uses_git_clone(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            Path(cmd[-1], "README.md").write_text("# Repo content", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr("subprocess.run", fake_run)
        chunks = load_from_source("https://github.com/acme/mockrepo", source_type="repo")
        assert chunks and "Repo content" in chunks[0]


# ── ingest_document stage wiring ────────────────────────────────────────────


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Stub out Qdrant and the embed/store step; capture what reaches storage."""
    import aegis.ingestion.pipeline as pipeline_mod

    captured: dict = {"metadata_list": None, "texts": None}

    class FakeRetrieval:
        def store_batch(self, texts, metadata_list=None):
            captured["texts"] = list(texts)
            captured["metadata_list"] = metadata_list
            return len(texts)

    monkeypatch.setattr(pipeline_mod, "RetrievalService", FakeRetrieval)
    monkeypatch.setattr(
        "aegis.memory.qdrant.QdrantDB",
        lambda: SimpleNamespace(_client=None),
    )
    return captured


class TestIngestDocumentStages:
    def test_preloaded_text_skips_parse(self, monkeypatch, patched_pipeline):
        import aegis.ingestion.pipeline as pipeline_mod

        def boom(*_a, **_k):
            raise AssertionError("parse_document must not be called with preloaded_text")

        monkeypatch.setattr(pipeline_mod, "parse_document", boom)
        result = pipeline_mod.ingest_document(
            source="https://github.com/acme/repo",
            document_id="d-url",
            filename="https://github.com/acme/repo",
            source_type="repo",
            preloaded_text="alpha beta gamma " * 10,
        )
        assert result.status == "ready"
        assert result.total_chunks >= 1
        assert result.stored_chunks >= 1
        assert not result.errors

    def test_source_type_recorded_in_payload(self, monkeypatch, patched_pipeline):
        import aegis.ingestion.pipeline as pipeline_mod

        monkeypatch.setattr(
            pipeline_mod, "parse_document", lambda *_a, **_k: "should not be used"
        )
        pipeline_mod.ingest_document(
            source="http://example.com/page",
            document_id="d-url2",
            filename="http://example.com/page",
            source_type="url",
            preloaded_text="payload text for source typing " * 10,
        )
        meta = patched_pipeline["metadata_list"][0]
        assert meta["source_type"] == "url"
        assert meta["doc_id"] == "d-url2"

    def test_normal_file_path_uses_parse_document(self, monkeypatch, patched_pipeline):
        import aegis.ingestion.pipeline as pipeline_mod

        calls: list = []

        def fake_parse(source, timeout=120.0):
            calls.append(str(source))
            return "parsed document body " * 10

        monkeypatch.setattr(pipeline_mod, "parse_document", fake_parse)
        result = pipeline_mod.ingest_document(
            source="/tmp/x.pdf", document_id="d-file", filename="x.pdf"
        )
        assert calls == ["/tmp/x.pdf"]
        assert result.status == "ready"
        assert patched_pipeline["metadata_list"][0]["source_type"] == "file"

    def test_empty_preloaded_text_is_an_error(self, patched_pipeline):
        import aegis.ingestion.pipeline as pipeline_mod

        result = pipeline_mod.ingest_document(
            source="http://example.com/empty",
            filename="empty",
            preloaded_text="",
        )
        assert result.status == "error"
        assert any("Empty document" in e for e in result.errors)

    def test_parse_exception_is_captured(self, monkeypatch, patched_pipeline):
        import aegis.ingestion.pipeline as pipeline_mod

        def boom(*_a, **_k):
            raise RuntimeError("docling exploded")

        monkeypatch.setattr(pipeline_mod, "parse_document", boom)
        result = pipeline_mod.ingest_document(source="/tmp/bad.pdf", filename="bad.pdf")
        assert result.status == "error"
        assert any("Parse failed" in e for e in result.errors)


# ── End-to-end through the API (live embedding + Qdrant) ────────────────────


@pytest.mark.integration
class TestIngestUrlEndToEnd:
    def test_url_ingest_through_api(self, client: TestClient, local_url_server: str):
        url = f"{local_url_server}/bulletin.md"
        response = client.post(
            "/api/documents/ingest", json={"source": url, "source_type": "url"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True, data.get("error")
        assert data["chunks"] >= 1
        assert data["source_type"] == "url"

        doc_id = self._find_doc_id(client, url)
        chunks = client.get(f"/api/documents/{doc_id}/chunks").json()
        assert chunks["success"] is True
        assert chunks["chunks"], "expected stored chunks"
        assert any("quokka-token-93017" in c["text"] for c in chunks["chunks"])
        assert all(c["source_type"] == "url" for c in chunks["chunks"])

    @staticmethod
    def _find_doc_id(client: TestClient, source: str) -> str:
        docs = client.get("/api/documents").json()["documents"]
        matches = [d for d in docs if d["filename"] == source]
        assert matches, f"document {source!r} not listed after ingest"
        return matches[-1]["id"]


@pytest.mark.integration
class TestIngestRepoEndToEnd:
    def test_repo_ingest_through_api(self, client: TestClient):
        response = client.post(
            "/api/documents/ingest",
            json={"source": "https://github.com/octocat/Spoon-Knife", "source_type": "repo"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True, data.get("error")
        assert data["chunks"] >= 1

        docs = client.get("/api/documents").json()["documents"]
        matches = [
            d for d in docs
            if d["filename"] == "https://github.com/octocat/Spoon-Knife"
        ]
        assert matches
        chunks = client.get(f"/api/documents/{matches[-1]['id']}/chunks").json()
        text = " ".join(c["text"] for c in chunks["chunks"])
        assert "fork" in text.lower(), "expected README content, got page boilerplate"
        assert all(c["source_type"] == "repo" for c in chunks["chunks"])