from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client.http import models as qm

import aegis.memory.qdrant as qdrant

pytestmark = pytest.mark.unit


def test_remote_client_receives_configured_api_key(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def get_collections(self):
            return SimpleNamespace(collections=[])

    monkeypatch.setattr(qdrant, "_QdrantClient", FakeClient)
    monkeypatch.setattr(qdrant.settings, "AEGIS_QDRANT_API_KEY", "test-api-key")

    qdrant.QdrantDB(url="https://qdrant.example", collection="documents")

    assert calls == [{
        "url": "https://qdrant.example",
        "api_key": "test-api-key",
        "timeout": 5,
    }]


def test_upsert_reports_client_failure():
    db = qdrant.QdrantDB(url=":memory:", collection="documents")

    class FailingClient:
        def upsert(self, **_kwargs):
            raise RuntimeError("storage unavailable")

    db._client = FailingClient()
    db._active_collection = "documents"
    point = qm.PointStruct(id=1, vector=[0.1], payload={})

    assert db.upsert([point]) is False
