# backend/tests/test_embeddings.py
# """Unit tests for EmbeddingModel: key rotation, 429 handling, local fallback."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from aegis.memory.embeddings import EmbeddingModel

pytestmark = pytest.mark.unit


def make_api_error(code: int = 429, message: str = "RESOURCE_EXHAUSTED: Quota exceeded"):
    from google.genai import errors

    return errors.APIError(code, {"error": {"code": code, "message": message}})


def make_model(clients, local_model=None) -> EmbeddingModel:
    """Build an EmbeddingModel without running the heavy __init__."""
    m = EmbeddingModel.__new__(EmbeddingModel)
    m._model_name = "models/text-embedding-004"
    m._clients = list(clients)
    m._key_index = 0
    m._lock = threading.Lock()
    m._local_model = local_model
    return m


class FakeClient:
    """Client whose embed_content follows a per-call scripted behaviour."""

    def __init__(self, behaviours: list):
        self._behaviours = behaviours
        self.calls = 0
        self.models = self

    def embed_content(self, *, model, contents, config):
        idx = min(self.calls, len(self._behaviours) - 1)
        self.calls += 1
        behaviour = self._behaviours[idx]
        if isinstance(behaviour, Exception):
            raise behaviour
        # Real Gemini returns one embedding per content item.
        n = len(contents) if isinstance(contents, list) else 1
        vals = behaviour.embeddings[0].values
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=list(vals)) for _ in range(n)]
        )


def ok_result(values=None):
    vals = values if values is not None else [0.5] * 768
    return SimpleNamespace(embeddings=[SimpleNamespace(values=vals)])


class FakeLocalEmbedder:
    """Stands in for fastembed TextEmbedding."""

    def __init__(self, vector=None):
        self._vector = vector if vector is not None else [0.25] * 8
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        for _ in texts:
            yield self._vector


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skip the real key-rotation backoff sleeps."""
    import aegis.memory.embeddings as embed_mod

    monkeypatch.setattr(embed_mod.time, "sleep", lambda _s: None)


class TestEmbed:
    def test_success_first_client(self):
        c = FakeClient([ok_result()])
        m = make_model([c])
        vec = m.embed("hello")
        assert vec == [0.5] * 768
        assert c.calls == 1

    def test_rotates_to_second_key_on_429(self):
        c0 = FakeClient([make_api_error(429)])
        c1 = FakeClient([ok_result([0.9] * 768)])
        m = make_model([c0, c1])
        vec = m.embed("hello")
        assert vec == [0.9] * 768
        assert c0.calls == 1
        assert c1.calls == 1

    def test_all_keys_rate_limited_falls_back_to_local(self):
        local = FakeLocalEmbedder([0.75] * 8)
        c0 = FakeClient([make_api_error(429)])
        m = make_model([c0], local_model=local)
        vec = m.embed("hello")
        assert vec == [0.75] * 8
        assert local.calls == 1

    def test_no_clients_uses_local(self):
        local = FakeLocalEmbedder()
        m = make_model([], local_model=local)
        assert m.embed("hello") == [0.25] * 8

    def test_no_clients_no_local_returns_empty(self):
        m = make_model([], local_model=None)
        assert m.embed("hello") == []

    def test_unexpected_exception_falls_back_to_local(self):
        c0 = FakeClient([ValueError("boom")])
        local = FakeLocalEmbedder()
        m = make_model([c0], local_model=local)
        assert m.embed("hello") == [0.25] * 8

    def test_503_is_treated_as_retryable(self):
        c0 = FakeClient([make_api_error(503, "UNAVAILABLE: backend down")])
        c1 = FakeClient([ok_result([0.1] * 768)])
        m = make_model([c0, c1])
        assert m.embed("hello") == [0.1] * 768

    def test_non_retryable_api_error_uses_local(self):
        # 400 → no rotation, straight to fallback.
        c0 = FakeClient([make_api_error(400, "INVALID_ARGUMENT")])
        local = FakeLocalEmbedder()
        m = make_model([c0], local_model=local)
        assert m.embed("hello") == [0.25] * 8
        assert c0.calls == 1


class TestClientRotation:
    def test_next_client_round_robins(self):
        c0, c1 = FakeClient([ok_result()]), FakeClient([ok_result()])
        m = make_model([c0, c1])
        assert [m._next_client() for _ in range(4)] == [c0, c1, c0, c1]

    def test_next_client_none_when_empty(self):
        assert make_model([])._next_client() is None


class TestEmbedBatch:
    def test_empty_input_returns_empty(self):
        m = make_model([FakeClient([ok_result()])])
        assert m.embed_batch([]) == []

    def test_batch_success(self):
        c = FakeClient([ok_result([0.5] * 768), ok_result([0.5] * 768)])
        m = make_model([c])
        out = m.embed_batch(["a", "b"], pause_between_batches=0)
        assert len(out) == 2
        assert all(v == [0.5] * 768 for v in out)

    def test_batch_rate_limited_falls_back_to_local(self):
        c = FakeClient([make_api_error(429)])
        local = FakeLocalEmbedder([0.33] * 8)
        m = make_model([c], local_model=local)
        out = m.embed_batch(["a"], pause_between_batches=0)
        assert out == [[0.33] * 8]

    def test_batch_no_clients_uses_local(self):
        local = FakeLocalEmbedder([0.66] * 8)
        m = make_model([], local_model=local)
        out = m.embed_batch(["a", "b"], pause_between_batches=0)
        assert out == [[0.66] * 8, [0.66] * 8]