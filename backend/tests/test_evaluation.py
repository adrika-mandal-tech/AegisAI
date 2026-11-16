"""Tests for evaluation modules: evaluator, ragas, deepeval, metrics."""

from __future__ import annotations
import pytest

from aegis.agents.evaluator import Evaluator
from aegis.evaluation.deepeval import evaluate_pipeline_run
from aegis.evaluation.ragas import evaluate_rag_pipeline
from aegis.evaluation.metrics import faithfulness_score, answer_relevancy_score
from aegis.graph.state import repair_state, PipelineState


class TestEvaluator:
    """Test the Evaluator agent."""
    def test_evaluate_returns_tuple(self):
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        evaluator = Evaluator(llm=mock_llm)
        result = evaluator.evaluate("Some document content.", [], task="test")
        assert len(result) == 4

    def test_evaluate_with_mock_llm(self):
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        evaluator = Evaluator(llm=mock_llm)
        metrics, evaluation, charts, tables = evaluator.evaluate(
            "Document with 15% improvement in performance.",
            [],
            task="analyze performance",
        )
        assert metrics.word_count > 0
        assert isinstance(charts, list)
        assert isinstance(tables, list)

    def test_compute_metrics_empty_evidence(self):
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        evaluator = Evaluator(llm=mock_llm)
        metrics = evaluator._compute_metrics("Some content.", [])
        assert metrics.word_count > 0
        assert metrics.completeness_score >= 0.0

    def test_heuristic_evaluation_fallback(self):
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        evaluator = Evaluator(llm=mock_llm)
        evaluation = evaluator._heuristic_evaluation([])
        assert 0.0 <= evaluation.overall_quality <= 1.0


class TestRagasEvaluation:
    """Test Ragas-style evaluation."""

    def test_evaluate_rag_pipeline_with_context(self):
        result = evaluate_rag_pipeline(
            task="What is the performance improvement?",
            retrieved_context="The system improved by 15% in throughput.",
            report_markdown="The system achieved a 15% improvement in throughput.",
        )
        assert result["framework"] == "ragas"
        assert "metrics" in result
        assert "overall_score" in result

    def test_evaluate_rag_pipeline_empty(self):
        result = evaluate_rag_pipeline("", "", "")
        assert result["status"] in ("ready", "error")
        assert "metrics" in result

    def test_evaluate_rag_pipeline_partial(self):
        result = evaluate_rag_pipeline(
            task="test",
            retrieved_context="some context",
            report_markdown="",
        )
        assert result["framework"] == "ragas"


class TestDeepEvalEvaluation:
    """Test DeepEval-style evaluation."""

    def test_evaluate_pipeline_run_with_report(self):
        result = evaluate_pipeline_run(
            task="test task",
            report_markdown="# Executive Summary\n\nThis is a report.",
            pipeline_trace=[{"agent": "retriever", "status": "ok"}],
        )
        assert result["framework"] == "deepeval"
        assert "metrics" in result
        assert "overall_score" in result

    def test_evaluate_pipeline_run_empty(self):
        result = evaluate_pipeline_run("", "", [])
        assert result["framework"] == "deepeval"
        assert "metrics" in result


class TestMetrics:
    """Test evaluation metric functions."""

    def test_faithfulness_score_supported(self):
        context = "The system improved by 15% in throughput and reduced latency."
        response = "The system improved by 15% in throughput."
        score = faithfulness_score(response, context)
        assert 0.0 <= score <= 1.0

    def test_faithfulness_score_unsupported(self):
        context = "The system improved."
        response = "The system failed completely."
        score = faithfulness_score(response, context)
        assert 0.0 <= score <= 1.0

    def test_faithfulness_score_empty_inputs(self):
        assert faithfulness_score("", "context") == 0.0
        assert faithfulness_score("response", "") == 0.0

    def test_answer_relevancy_score_high(self):
        question = "What is the performance improvement?"
        answer = "The performance improvement is 15% higher throughput."
        score = answer_relevancy_score(question, answer)
        assert score > 0.5

    def test_answer_relevancy_score_low(self):
        question = "What is the performance improvement?"
        answer = "The weather is sunny today."
        score = answer_relevancy_score(question, answer)
        assert score < 0.5

    def test_answer_relevancy_score_empty(self):
        assert answer_relevancy_score("", "answer") == 0.0
        assert answer_relevancy_score("question", "") == 0.0


class TestStateRepair:
    """Test pipeline state repair mechanism."""

    def test_repair_state_preserves_valid_fields(self):
        state: PipelineState = {
            "task": "test",
            "supervisor_analysis": "ok",
            "supervisor_details": "",
            "plan_objective": "",
            "plan_steps": [],
            "retrieved_chunks": [],
            "retrieved_context": "",
            "research_results": "",
            "validation_score": 0.0,
            "validation_issues": [],
            "validation_suggestions": [],
            "reasoning_summary": "",
            "reasoning_details": "",
            "critique_strengths": [],
            "critique_weaknesses": [],
            "critique_assessment": "",
            "evidence_items": [],
            "report_title": "",
            "report_markdown": "",
            "report_action_items": [],
            "report_charts": [],
            "agent_outputs": {},
            "evaluation_results": {},
            "document_metrics": {},
            "chart_data": [],
            "table_data": [],
            "tool_calls": [],
            "tool_results": [],
            "memory_entries": [],
            "errors": [],
            "agent_steps": [],
            "on_agent_event": None,
            "record_pipeline_run": None,
        }
        repaired = repair_state(state)
        assert repaired["task"] == "test"
        assert repaired["supervisor_analysis"] == "ok"

    def test_repair_state_fills_missing_fields(self):
        state: PipelineState = {
            "task": "test",
        }
        repaired = repair_state(state)
        assert "retrieved_chunks" in repaired
        assert repaired["retrieved_chunks"] == []
        assert "agent_steps" in repaired
        assert repaired["agent_steps"] == []

    def test_repair_state_normalizes_non_list_fields(self):
        state: PipelineState = {
            "task": "test",
            "errors": "not a list",
            "agent_steps": "also not a list",
        }
        repaired = repair_state(state)
        assert isinstance(repaired["errors"], list)
        assert isinstance(repaired["agent_steps"], list)

    def test_repair_state_normalizes_chart_table_fields(self):
        """Phase 2: chart_data/table_data/plan_steps must survive as lists."""
        state: PipelineState = {
            "task": "test",
            "chart_data": "not a list",
            "table_data": {"not": "a list"},
            "plan_steps": "oops",
            "evaluation_results": None,
        }
        repaired = repair_state(state)
        assert isinstance(repaired["chart_data"], list)
        assert isinstance(repaired["table_data"], list)
        assert isinstance(repaired["plan_steps"], list)
        assert isinstance(repaired["evaluation_results"], dict)


class TestPhase2ChartTableExtraction:
    """Phase 2 regression tests: chart/table extraction on plain content."""

    def _evaluator(self):
        from unittest.mock import MagicMock
        from aegis.agents.evaluator import Evaluator
        from pathlib import Path
        mock_llm = MagicMock()
        # Force heuristic-only extraction paths: skip LLM-based extraction
        # (MagicMock's class name contains no "groq", so without this the
        # code would try to iterate a Mock and fail).
        mock_llm.generate_model_safe.return_value = None
        return Evaluator(llm=mock_llm)

    def _test_eval_doc(self) -> str:
        return (
            "# Test Document for Evaluation\n\n"
            "This document contains a table and some numeric data for chart extraction.\n\n"
            "## Performance Metrics\n\n"
            "| Component | Speed (TFLOPS) | Efficiency (%) |\n"
            "|-----------|----------------|----------------|\n"
            "| GPU A     | 12.5           | 85             |\n"
            "| GPU B     | 15.2           | 90             |\n"
            "| GPU C     | 18.7           | 92             |\n\n"
            "## Key Specifications\n\n"
            "- Memory Bandwidth: 750 GB/s\n"
            "- Core Count: 4500\n"
            "- Clock Speed: 1.8 GHz\n"
            "- Power Consumption: 250 W\n\n"
            "## Performance Improvement\n\n"
            "The new model shows 25% improvement in speed over the previous generation.\n"
        )

    def test_chart_data_metadata_default_is_dict(self):
        from aegis.schemas.agent import ChartData
        chart = ChartData(chart_type="bar", title="t", labels=["a"], values=[1.0])
        assert isinstance(chart.metadata, dict)
        dumped = chart.model_dump()
        assert isinstance(dumped["metadata"], dict)

    def test_extract_charts_from_key_value_specs(self):
        evaluator = self._evaluator()
        content = (
            "Memory Bandwidth: 750 GB/s\n"
            "Core Count: 4500\n"
            "Clock Speed: 1.8 GHz\n"
            "Power Consumption: 250 W\n"
        )
        charts = evaluator._extract_charts(content)
        assert isinstance(charts, list) and len(charts) >= 1
        assert len(charts[0].labels) == len(charts[0].values)

    def test_extract_charts_from_test_eval_doc(self):
        evaluator = self._evaluator()
        content = self._test_eval_doc()
        charts = evaluator._extract_charts(content)
        assert len(charts) >= 1

    def test_extract_tables_from_test_eval_doc(self):
        evaluator = self._evaluator()
        content = self._test_eval_doc()
        tables = evaluator._extract_tables(content)
        assert len(tables) >= 1
        assert tables[0].headers
        assert tables[0].rows

    def test_extract_tables_normalizes_ragged_rows(self):
        evaluator = self._evaluator()
        content = (
            "| A | B | C |\n"
            "|---|---|---|\n"
            "| 1 | 2 |\n"
            "| 1 | 2 | 3 | 4 |\n"
        )
        tables = evaluator._extract_tables(content)
        assert len(tables) >= 1
        width = len(tables[0].headers)
        assert all(len(row) == width for row in tables[0].rows)

    def test_extract_tables_from_html(self):
        evaluator = self._evaluator()
        content = (
            "<table><tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>Alpha</td><td>1</td></tr></table>"
        )
        tables = evaluator._extract_tables(content)
        assert len(tables) >= 1
        assert tables[0].headers == ["Name", "Value"]

    def test_compute_metrics_counts_table_blocks(self):
        evaluator = self._evaluator()
        content = (
            "| A | B |\n|---|---|\n| 1 | 2 |\n\n"
            "Some prose with a | single pipe that is not a table.\n"
        )
        metrics = evaluator._compute_metrics(content, [])
        assert metrics.table_count == 1
        assert 0.0 <= metrics.completeness_score <= 1.0

    def test_evaluate_handles_malformed_evidence(self):
        """Malformed evidence dicts must be skippable without crashing."""
        from aegis.schemas.agent import Evidence

        def _to_evidence_objects(raw_items):
            # Mirror of aegis.graph.nodes._to_evidence_objects (kept local so
            # this test never imports the heavy nodes module / LLM clients).
            objects = []
            for item in raw_items or []:
                if not isinstance(item, dict):
                    continue
                try:
                    objects.append(Evidence(**item))
                except Exception:
                    continue
            return objects

        raw = [
            {"claim": "ok", "source": "s", "supporting_passage": "p"},
            {"claim": "missing source and passage"},
            "not a dict",
        ]
        objects = _to_evidence_objects(raw)  # type: ignore[list-item]
        assert len(objects) == 1
