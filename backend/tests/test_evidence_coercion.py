"""Tests for loose-dict → Evidence coercion and pipeline resilience."""

from __future__ import annotations

from aegis.schemas.agent import Evidence, coerce_evidence


class TestCoerceEvidence:
    def test_canonical_dict_passes_through(self) -> None:
        item = {"claim": "c", "source": "s", "supporting_passage": "p"}
        evidence = coerce_evidence(item)
        assert isinstance(evidence, Evidence)
        assert evidence.claim == "c"
        assert evidence.source == "s"
        assert evidence.supporting_passage == "p"

    def test_llm_finding_shape_is_coerced(self) -> None:
        # Exact shape observed from research-LLM output that used to crash
        # reasoner_node with a pydantic ValidationError.
        item = {
            "finding": "Critical SQL injection in billing service",
            "recommendation": "Patch billing service by 2026-10-15",
            "confidence": 0.7,
            "relevance": 9,
        }
        evidence = coerce_evidence(item)
        assert evidence is not None
        assert evidence.claim.startswith("Critical SQL injection")
        assert evidence.supporting_passage
        assert 0.0 <= evidence.confidence <= 1.0
        assert 0.0 <= evidence.relevance <= 1.0

    def test_statement_alias_maps_to_claim(self) -> None:
        evidence = coerce_evidence({"statement": "s1", "text": "passage"})
        assert evidence is not None
        assert evidence.claim == "s1"
        assert evidence.supporting_passage == "passage"

    def test_evidence_instance_passthrough(self) -> None:
        original = Evidence(claim="c", source="s", supporting_passage="p")
        assert coerce_evidence(original) is original

    def test_hopeless_items_return_none(self) -> None:
        assert coerce_evidence({"foo": "bar"}) is None
        assert coerce_evidence("not a dict") is None
        assert coerce_evidence(None) is None
        assert coerce_evidence(42) is None


class TestPipelineEvidenceResilience:
    def test_to_evidence_objects_coerces_and_skips(self) -> None:
        from aegis.graph.nodes import _to_evidence_objects

        items = [
            {"claim": "c", "source": "s", "supporting_passage": "p"},
            {"finding": "f1"},
            {"useless": "x"},
        ]
        objects = _to_evidence_objects(items)
        assert len(objects) == 2
        assert all(isinstance(o, Evidence) for o in objects)

    def test_reasoner_node_survives_loose_evidence(self, monkeypatch) -> None:
        """reasoner_node must not crash on LLM-shaped evidence dicts."""
        from aegis.graph import nodes

        class _FakeReasoner:
            def reason(self, *_args, **_kwargs):
                class _R:
                    summary = "ok"
                    details = "ok"

                return _R()

        monkeypatch.setattr(nodes, "_reasoner", _FakeReasoner())
        state = {
            "task": "t",
            "retrieved_context": "ctx",
            "research_results": "res",
            "validation_issues": [],
            "evidence_items": [{"finding": "some finding"}],
        }
        result = nodes.reasoner_node(state)
        assert result["agent_steps"][0]["status"] == "ok"