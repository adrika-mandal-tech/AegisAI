# backend/tests/test_chunker.py
"""Unit tests for Section 4 Advanced Chunking Engine (900-word default, 10% dynamic overlap)."""

from __future__ import annotations

import hashlib

import pytest

from aegis.ingestion.chunker import (
    DocumentChunk,
    _resolve_chunk_size,
    _resolve_overlap,
    chunk_normalized_document,
    chunk_text,
    chunk_text_adaptive,
    count_words,
    verify_chunk_overlap,
)
from aegis.ingestion.parser_router import NormalizedDocument


@pytest.mark.unit
class TestChunkText:
    """chunk_text: word-based splitting with mandatory 900-word default & 10% (90w) overlap."""

    def test_short_text_returns_single_chunk(self):
        assert chunk_text("hello world") == ["hello world"]

    def test_empty_and_whitespace_return_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\t  ") == []

    def test_mandatory_default_900_word_chunks_and_90_word_ten_percent_overlap(self):
        assert _resolve_chunk_size(None) == 900
        assert _resolve_overlap(900, None) == 90

        words = [f"w{i}" for i in range(2500)]
        text = " ".join(words)
        chunks = chunk_text(text)
        assert len(chunks) == 3
        assert count_words(chunks[0]) == 900
        assert count_words(chunks[1]) == 900
        assert count_words(chunks[2]) == 880

        diag = verify_chunk_overlap(chunks)
        assert diag["overlaps_words"] == [90, 90]
        assert diag["all_pairs_overlap"] is True

    def test_dynamic_ten_percent_overlap_for_custom_chunk_size(self):
        assert _resolve_overlap(500, None) == 50
        assert _resolve_overlap(1200, None) == 120
        words = [f"w{i}" for i in range(1200)]
        chunks = chunk_text(" ".join(words), chunk_size=500, overlap=None)
        diag = verify_chunk_overlap(chunks)
        assert diag["overlaps_words"][0] == 50

    def test_all_words_preserved_across_chunks(self):
        words = [f"w{i}" for i in range(2500)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=900)
        seen: set[str] = set()
        for c in chunks:
            seen.update(c.split())
        assert seen == set(words)

    def test_consecutive_chunks_overlap(self):
        words = [f"w{i}" for i in range(1200)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=500, overlap=50, min_chunk_size=10)
        first, second = chunks[0].split(), chunks[1].split()
        shared = set(first) & set(second)
        assert len(shared) == 50

    def test_custom_size_and_zero_overlap(self):
        words = [f"w{i}" for i in range(120)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=50, overlap=0, min_chunk_size=5)
        assert count_words(chunks[0]) <= 50
        assert len(chunks) == 3

    def test_min_chunk_size_drops_undersized_non_final_chunks(self):
        words = [f"w{i}" for i in range(100)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=30, overlap=5, min_chunk_size=50)
        assert all(count_words(c) <= 30 for c in chunks)
        assert chunks[-1].split()[-1] == words[-1]

    def test_exact_boundary_length_single_chunk(self):
        words = [f"w{i}" for i in range(900)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=900)
        assert len(chunks) == 1
        assert count_words(chunks[0]) == 900

    def test_sentence_and_paragraph_boundary_preservation(self):
        sentence = "Autonomous multi-agent reasoning validates every empirical claim carefully."
        paragraph = " ".join([sentence] * 8)  # 72 words per paragraph
        text = "\n\n".join([paragraph] * 25)  # 1800 words
        chunks = chunk_text(text, chunk_size=900)
        assert len(chunks) >= 2
        assert chunks[0].rstrip().endswith(".")


@pytest.mark.unit
class TestChunkTextAdaptiveAndStructural:
    """Adaptive & structural NormalizedDocument chunking modes."""

    @pytest.mark.parametrize("file_size", [0, 6 * 1024 * 1024, 11 * 1024 * 1024, 21 * 1024 * 1024])
    def test_window_scales_with_file_size(self, file_size: int):
        text = " ".join(f"w{i}" for i in range(2500))
        chunks = chunk_text_adaptive(text, file_size_bytes=file_size)
        baseline = chunk_text_adaptive(text, file_size_bytes=0)
        assert len(chunks) <= len(baseline)

    def test_empty_text_returns_nothing(self):
        assert chunk_text_adaptive("", file_size_bytes=100) == []

    @pytest.mark.parametrize(
        "strategy",
        ["word", "adaptive", "document_aware", "section_aware", "heading_aware", "table_aware", "page_aware"],
    )
    def test_all_seven_chunking_strategies_work(self, strategy: str):
        sec1 = " ".join(f"alpha{i}" for i in range(600))
        sec2 = " ".join(f"beta{i}" for i in range(600))
        norm = NormalizedDocument(
            document_id="doc-struct-1",
            filename="report.pdf",
            file_type="pdf",
            page_count=2,
            extracted_text=f"# Executive Overview\n{sec1}\n\n# Financial Results\n{sec2}",
            pages=[
                {"page_number": 1, "text": sec1, "headings": ["Executive Overview"]},
                {"page_number": 2, "text": sec2, "headings": ["Financial Results"]},
            ],
            sections=[
                {"heading": "Executive Overview", "level": 1, "text": sec1, "page_number": 1},
                {"heading": "Financial Results", "level": 1, "text": sec2, "page_number": 2},
            ],
            tables=[
                {
                    "table_index": 0,
                    "page_number": 2,
                    "headers": ["Quarter", "Revenue"],
                    "rows": [["Q1", "120"], ["Q2", "145"]],
                    "markdown": "| Quarter | Revenue |\n| --- | --- |\n| Q1 | 120 |",
                }
            ],
            parser_used="pymupdf",
            extraction_status="success",
        )
        chunks = chunk_normalized_document(norm, strategy=strategy, embedding_model="BAAI/bge-m3")
        assert len(chunks) >= 1
        payload = chunks[0].to_payload()
        assert payload["document_id"] == "doc-struct-1"
        assert payload["doc_id"] == "doc-struct-1"
        assert payload["filename"] == "report.pdf"
        assert payload["extraction_method"] == "pymupdf"
        assert payload["embedding_model"] == "BAAI/bge-m3"
        assert payload["creation_timestamp"]


@pytest.mark.unit
class TestDocumentChunk:
    """DocumentChunk: stable IDs and Qdrant payload provenance contract."""

    def _chunk(self, **kwargs) -> DocumentChunk:
        defaults = dict(
            text="alpha beta gamma",
            document_id="doc-1",
            chunk_index=0,
            source="sample.md",
            section_heading="Intro",
            extraction_method="markdown",
            embedding_model="BAAI/bge-m3",
        )
        defaults.update(kwargs)
        return DocumentChunk.from_text(**defaults)

    def test_chunk_id_is_deterministic(self):
        a = self._chunk()
        b = self._chunk()
        assert a.chunk_id == b.chunk_id
        assert len(a.chunk_id) == 16

    def test_chunk_id_varies_by_index_and_text(self):
        base = self._chunk()
        assert base.chunk_id != self._chunk(chunk_index=1).chunk_id
        assert base.chunk_id != self._chunk(text="different content").chunk_id

    def test_payload_contract(self):
        payload = self._chunk(chunk_index=2, page=4).to_payload()
        assert payload["doc_id"] == "doc-1"
        assert payload["document_id"] == "doc-1"
        assert payload["text"] == "alpha beta gamma"
        assert payload["filename"] == "sample.md"
        assert payload["chunk_index"] == 2
        assert payload["page"] == 4
        assert payload["page_number"] == 4
        assert payload["section_heading"] == "Intro"
        assert payload["extraction_method"] == "markdown"
        assert payload["embedding_model"] == "BAAI/bge-m3"
        assert payload["creation_timestamp"]
        assert payload["word_count"] == 3
        assert payload["type"] == "document_chunk"
        assert payload["source_type"] == "file"
        assert payload["token_count"] >= 1
        assert len(payload["content_hash"]) == 16

    def test_token_count_at_least_one_for_tiny_text(self):
        payload = DocumentChunk.from_text(text="hi", document_id="d", chunk_index=0).to_payload()
        assert payload["token_count"] == 1

    def test_metadata_source_type_overrides_default(self):
        payload = DocumentChunk.from_text(
            text="some text",
            document_id="d",
            chunk_index=0,
            metadata={"source_type": "url", "char_offset": 42},
        ).to_payload()
        assert payload["source_type"] == "url"
        assert payload["char_offset"] == 42

    def test_content_hash_matches_text_hash(self):
        c = self._chunk()
        expected = hashlib.sha256("alpha beta gamma".encode()).hexdigest()[:16]
        assert c.to_payload()["content_hash"] == expected
