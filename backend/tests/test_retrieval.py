# backend/tests/test_retrieval.py
# """Unit tests for RetrievalService: embed → search → typed chunks, store paths."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from aegis.memory.retrieval import RetrievalPipeline, RetrievalService

pytestmark = pytest.mark.unit


class FakeEmbedder:
    def __init__(self, vector=None):
        self._vector = [0.5] * 8 if vector is None else vector
        self.embedded: list[str] = []
        self.batch_texts: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        return list(self._vector)

    def embed_batch(self, texts, **_kw) -> list[list[float]]:
        self.batch_texts.append(list(texts))
        return [list(self._vector) for _ in texts]


class FakeDB:
    def __init__(
        self,
        results=None,
        *,
        ensure_ok=True,
        raise_on_search=False,
        upsert_ok=True,
    ):
        self._collection = "fake_collection"
        self._client = None
        self.results = results if results is not None else []
        self.ensure_ok = ensure_ok
        self.raise_on_search = raise_on_search
        self.upsert_ok = upsert_ok
        self.search_calls: list[dict] = []
        self.upserted: list = []
        self.ensured_sizes: list[int] = []

    def ensure_collection(self, *, vector_size: int) -> bool:
        self.ensured_sizes.append(vector_size)
        return self.ensure_ok

    def search(self, *, query_vector, limit, score_threshold, filter_condition):
        self.search_calls.append({
            "query_vector": query_vector,
            "limit": limit,
            "score_threshold": score_threshold,
            "filter_condition": filter_condition,
        })
        if self.raise_on_search:
            raise RuntimeError("qdrant down")
        return [
            r for r in self.results
            if score_threshold is None or r.get("score", 0.0) >= score_threshold
        ]

    def upsert(self, points):
        self.upserted.extend(points)
        return self.upsert_ok

    def get_count(self) -> int:
        return len(self.upserted)


def raw(id: str, score: float, text: str, doc_id: str = "docA", **extra):
    payload = {
        "chunk_id": f"chunk-{id}",
        "doc_id": doc_id,
        "text": text,
        "filename": "file.md",
        "chunk_index": 0,
        "type": "document_chunk",
        "source_type": "file",
        "token_count": 3,
        "char_offset": 0,
        "content_hash": "abc",
    }
    payload.update(extra)
    return {"id": id, "score": score, "payload": payload}


def make_service(results=None, **db_kw) -> tuple[RetrievalService, FakeDB, FakeEmbedder]:
    db = FakeDB(results, **db_kw)
    embedder = FakeEmbedder()
    svc = RetrievalService(embedder=embedder, vector_db=db, llm=None)
    return svc, db, embedder


class TestRetrieve:
    def test_returns_typed_chunks_sorted_by_score(self):
        svc, db, _ = make_service([
            raw("1", 0.60, "lower"),
            raw("2", 0.95, "higher"),
        ])
        chunks = svc.retrieve("query")
        assert len(chunks) == 2
        assert chunks[0].text == "higher"
        assert chunks[0].score == pytest.approx(0.95)
        assert chunks[0].chunk_id == "chunk-2"
        assert chunks[0].document_id == "docA"
        assert chunks[0].source == "file.md"
        assert isinstance(chunks[0].metadata, dict)

    def test_metadata_strips_known_payload_keys(self):
        svc, _, _ = make_service([
            raw("1", 0.9, "text", custom_field="kept"),
        ])
        chunk = svc.retrieve("q")[0]
        assert chunk.metadata.get("custom_field") == "kept"
        assert "type" not in chunk.metadata
        assert "token_count" not in chunk.metadata
        assert "content_hash" not in chunk.metadata

    def test_empty_embedding_returns_no_results(self):
        class NoEmbed(FakeEmbedder):
            def embed(self, text):
                return []

        db = FakeDB([raw("1", 0.9, "x")])
        svc = RetrievalService(embedder=NoEmbed(), vector_db=db, llm=None)
        with pytest.raises(RuntimeError, match="no query vectors"):
            svc.retrieve("q")
        assert db.search_calls == []  # never reaches Qdrant

    def test_ensure_collection_failure_returns_empty(self):
        svc, db, _ = make_service([raw("1", 0.9, "x")], ensure_ok=False)
        with pytest.raises(RuntimeError, match="Qdrant collection is unavailable"):
            svc.retrieve("q")
        assert db.search_calls == []

    def test_search_exception_is_propagated(self):
        svc, _, _ = make_service([raw("1", 0.9, "x")], raise_on_search=True)
        with pytest.raises(RuntimeError, match="Qdrant search failed"):
            svc.retrieve("q")

    def test_no_results_returns_empty(self):
        svc, _, _ = make_service([])
        assert svc.retrieve("q") == []

    def test_score_threshold_filters_low_scores(self):
        svc, db, _ = make_service([
            raw("1", 0.9, "good"),
            raw("2", 0.1, "weak"),
        ])
        chunks = svc.retrieve("q", score_threshold=0.5)
        assert [c.text for c in chunks] == ["good"]
        assert db.search_calls[0]["score_threshold"] == 0.5

    def test_doc_id_filter_passed_to_qdrant(self):
        svc, db, _ = make_service([])
        svc.retrieve("q", doc_id="doc-42")
        flt = db.search_calls[0]["filter_condition"]
        assert flt is not None
        assert flt.must[0].key == "doc_id"
        assert flt.must[0].match.value == "doc-42"

    def test_no_doc_id_means_no_filter(self):
        svc, db, _ = make_service([])
        svc.retrieve("q")
        assert db.search_calls[0]["filter_condition"] is None

    def test_dedup_merges_identical_text(self):
        svc, _, _ = make_service([
            raw("1", 0.9, "same text"),
            raw("2", 0.8, "same text"),
        ])
        chunks = svc.retrieve("q")
        assert len(chunks) == 1
        assert chunks[0].score == pytest.approx(0.9)  # highest kept

    def test_dedup_disabled_keeps_duplicates_up_to_top_k(self):
        svc, _, _ = make_service([
            raw("1", 0.9, "same text"),
            raw("2", 0.8, "same text"),
        ])
        chunks = svc.retrieve("q", enable_dedup=False)
        assert len(chunks) == 2

    def test_multi_query_without_llm_uses_original_query(self):
        svc, db, embedder = make_service([])
        svc.retrieve("my question", enable_multi_query=True)
        assert embedder.embedded == ["my question"]
        assert len(db.search_calls) == 1

    def test_hyde_disabled_by_default(self):
        svc, _, embedder = make_service([])
        svc.retrieve("q")
        assert len(embedder.embedded) == 1

    def test_merge_deduplicates_same_point_across_vectors(self):
        svc, _, _ = make_service([raw("1", 0.9, "once")])
        chunks = svc.retrieve("q")
        assert len(chunks) == 1

    def test_top_k_limits_results(self):
        results = [raw(str(i), 0.5 + i * 0.01, f"t{i}") for i in range(10)]
        svc, _, _ = make_service(results)
        chunks = svc.retrieve("q", top_k=3)
        assert len(chunks) == 3


class TestStore:
    def test_store_embeds_and_upserts(self):
        svc, db, embedder = make_service()
        svc.store("hello", metadata={"doc_id": "d1"})
        assert embedder.embedded == ["hello"]
        assert len(db.upserted) == 1
        assert db.upserted[0].payload == {"doc_id": "d1"}
        assert len(db.upserted[0].vector) == 8

    def test_store_empty_embedding_fails_explicitly(self):
        class NoEmbed(FakeEmbedder):
            def embed(self, text):
                return []

        db = FakeDB()
        svc = RetrievalService(embedder=NoEmbed(), vector_db=db, llm=None)
        with pytest.raises(RuntimeError, match="empty vector"):
            svc.store("hello")
        assert db.upserted == []

    def test_store_batch_empty_texts_returns_zero(self):
        svc, db, _ = make_service()
        assert svc.store_batch([]) == 0
        assert db.upserted == []

    def test_store_batch_success_counts_points(self):
        svc, db, embedder = make_service()
        stored = svc.store_batch(
            ["a", "b"],
            metadata_list=[{"i": 0}, {"i": 1}],
        )
        assert stored == 2
        assert len(db.upserted) == 2
        assert embedder.batch_texts == [["a", "b"]]
        assert db.upserted[0].payload == {"i": 0}

    def test_store_batch_rejects_empty_vectors(self):
        class PartialEmbed(FakeEmbedder):
            def embed_batch(self, texts, **_kw):
                return [[0.5] * 8 if t != "bad" else [] for t in texts]

        db = FakeDB()
        svc = RetrievalService(embedder=PartialEmbed(), vector_db=db, llm=None)
        with pytest.raises(ValueError, match="dimensions"):
            svc.store_batch(["ok", "bad"])
        assert db.upserted == []

    def test_store_batch_ensure_failure_is_explicit(self):
        svc, db, _ = make_service(ensure_ok=False)
        with pytest.raises(RuntimeError, match="Qdrant collection is unavailable"):
            svc.store_batch(["a"])
        assert db.upserted == []

    def test_store_batch_rejects_non_finite_embeddings(self):
        svc, db, embedder = make_service()
        embedder._vector[0] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            svc.store_batch(["a"])
        assert db.upserted == []

    def test_store_batch_rejects_metadata_count_mismatch(self):
        svc, db, _ = make_service()
        with pytest.raises(ValueError, match="metadata records"):
            svc.store_batch(["a", "b"], metadata_list=[{"chunk_id": "only-one"}])
        assert db.upserted == []

    def test_store_batch_uses_stable_point_ids(self):
        svc, db, _ = make_service()
        metadata = [{"doc_id": "doc-1", "chunk_id": "chunk-1"}]
        svc.store_batch(["repeatable text"], metadata_list=metadata)
        first_id = db.upserted[-1].id
        svc.store_batch(["repeatable text"], metadata_list=metadata)
        assert db.upserted[-1].id == first_id

    def test_store_batch_upsert_failure_is_explicit(self):
        svc, db, _ = make_service(upsert_ok=False)
        with pytest.raises(RuntimeError, match="failed to upsert"):
            svc.store_batch(["a"])


class TestStats:
    def test_stats_error_path_when_client_missing(self):
        svc, _, _ = make_service()
        stats = svc.get_collection_stats()
        assert stats["status"] == "error"
        assert stats["collection"] == "fake_collection"
        assert "error" in stats

    def test_stats_success(self):
        svc, db, _ = make_service()
        info = SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=8)))
        )
        db._client = SimpleNamespace(get_collection=lambda name: info)
        stats = svc.get_collection_stats()
        assert stats["status"] == "ok"
        assert stats["vector_size"] == 8
        assert stats["points_count"] == 0


def test_retrieval_pipeline_alias_is_subclass():
    assert issubclass(RetrievalPipeline, RetrievalService)