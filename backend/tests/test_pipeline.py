"""Tests for the LangGraph RAG pipeline.

Uses mocked agents to verify state flow through all nodes without
requiring live API calls.
"""

from __future__ import annotations

from importlib import import_module
from unittest.mock import patch

import pytest

from aegis.graph.nodes import (
    critic_node,
    memory_node,
    planner_node,
    reasoner_node,
    report_node,
    researcher_node,
    retriever_node,
    supervisor_node,
    validator_node,
)
from aegis.graph.pipeline import _repair_wrapper
from aegis.graph.state import PipelineState
from aegis.schemas.agent import (
    AgentOutput,
    AgentPlan,
    CritiqueResult,
    PlanStep,
    ReportOutput,
    ResearchResult,
    ValidationResult,
)


@pytest.fixture
def base_state() -> PipelineState:
    """A minimal PipelineState with a task."""
    return {
        "task": "test task",
        "doc_id": "doc-42",
        "supervisor_analysis": "",
        "supervisor_details": "",
        "plan_objective": "",
        "plan_steps": [],
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
        "report_title": "",
        "report_markdown": "",
        "report_action_items": [],
        "errors": [],
        "agent_steps": [],
    }


# ── Supervisor ──────────────────────────────────────────────────────────────


@patch("aegis.graph.nodes._supervisor")
def test_supervisor_node_updates_state(mock_supervisor, base_state: PipelineState):
    mock_supervisor.supervise.return_value = AgentOutput(
        summary="analysis", details="details"
    )
    result = supervisor_node(base_state)
    assert result["supervisor_analysis"] == "analysis"
    assert result["supervisor_details"] == "details"
    assert result["agent_steps"] == [{"agent": "supervisor", "status": "ok"}]


# ── Planner ─────────────────────────────────────────────────────────────────


@patch("aegis.graph.nodes._planner")
def test_planner_node_updates_state(mock_planner, base_state: PipelineState):
    mock_planner.plan.return_value = AgentPlan(
        objective="plan objective",
        steps=[PlanStep(step=1, agent="retriever", description="search")],
    )
    result = planner_node(base_state)
    assert result["plan_objective"] == "plan objective"
    assert len(result["plan_steps"]) == 1


# ── Retriever ───────────────────────────────────────────────────────────────


@patch("aegis.graph.nodes._retriever")
def test_retriever_node_updates_state(mock_retriever, base_state: PipelineState):
    mock_retriever.retrieve.return_value = ResearchResult(
        query="test task", content="retrieved data"
    )
    result = retriever_node(base_state)
    assert result["retrieved_context"] == "retrieved data"
    mock_retriever.retrieve.assert_called_once_with("test task", doc_id="doc-42")


def test_repair_wrapper_returns_reducer_deltas_only():
    previous_step = {"agent": "previous", "status": "ok"}
    new_step = {"agent": "current", "status": "ok"}

    def node(_state):
        return {"agent_steps": [new_step], "errors": ["new error"]}

    wrapped = _repair_wrapper(node)
    result = wrapped({
        "task": "test",
        "agent_steps": [previous_step],
        "errors": ["old error"],
    })
    assert result["agent_steps"] == [new_step]
    assert result["errors"] == ["new error"]


def test_compiled_pipeline_carries_retrieved_context_and_reducer_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_module = import_module("aegis.graph.pipeline")

    def node_for(agent: str):
        def run(state):
            if agent == "reasoner":
                assert state["retrieved_context"] == "retrieved evidence"
            updates = {"agent_steps": [{"agent": agent, "status": "ok"}]}
            if agent == "retriever":
                updates["retrieved_context"] = "retrieved evidence"
            return updates

        return run

    node_names = (
        "supervisor_node",
        "planner_node",
        "retriever_node",
        "researcher_node",
        "validator_node",
        "reasoner_node",
        "critic_node",
        "report_node",
        "evaluation_node",
        "tool_node",
        "memory_node",
    )
    agent_names = (
        "supervisor",
        "planner",
        "retriever",
        "researcher",
        "validator",
        "reasoner",
        "critic",
        "report_generator",
        "evaluation",
        "tool_agent",
        "memory",
    )
    for node_name, agent_name in zip(node_names, agent_names, strict=True):
        monkeypatch.setattr(pipeline_module, node_name, node_for(agent_name))

    result = pipeline_module.build_pipeline().invoke({
        "task": "test grounded workflow",
        "doc_id": "doc-42",
        "errors": [],
        "agent_steps": [],
    })

    assert result["retrieved_context"] == "retrieved evidence"
    expected_order = (
        "supervisor",
        "planner",
        "retriever",
        "researcher",
        "tool_agent",
        "validator",
        "reasoner",
        "critic",
        "report_generator",
        "evaluation",
        "memory",
    )
    assert [step["agent"] for step in result["agent_steps"]] == list(expected_order)


# ── Researcher ──────────────────────────────────────────────────────────────


@patch("aegis.graph.nodes._researcher")
def test_researcher_node_updates_state(mock_researcher, base_state: PipelineState):
    mock_researcher.research.return_value = ResearchResult(
        query="test task", content="research data"
    )
    result = researcher_node(base_state)
    assert result["research_results"] == "research data"


# ── Validator ───────────────────────────────────────────────────────────────


@patch("aegis.graph.nodes._validator")
def test_validator_node_updates_state(mock_validator, base_state: PipelineState):
    mock_validator.validate.return_value = ValidationResult(
        score=0.85, issues=["minor issue"]
    )
    result = validator_node(base_state)
    assert result["validation_score"] == 0.85
    assert "minor issue" in result["validation_issues"]


# ── Reasoner ────────────────────────────────────────────────────────────────


@patch("aegis.graph.nodes._reasoner")
def test_reasoner_node_updates_state(mock_reasoner, base_state: PipelineState):
    mock_reasoner.reason.return_value = AgentOutput(
        summary="reasoning summary", details="detailed reasoning"
    )
    result = reasoner_node(base_state)
    assert result["reasoning_summary"] == "reasoning summary"


# ── Critic ──────────────────────────────────────────────────────────────────


@patch("aegis.graph.nodes._critic")
def test_critic_node_updates_state(mock_critic, base_state: PipelineState):
    mock_critic.critique.return_value = CritiqueResult(
        strengths=["clear"], weaknesses=["brief"], overall_assessment="ok"
    )
    base_state["reasoning_details"] = "some reasoning"
    result = critic_node(base_state)
    assert result["critique_assessment"] == "ok"


# ── Report Generator ────────────────────────────────────────────────────────


@patch("aegis.graph.nodes._report_gen")
def test_report_node_updates_state(mock_report, base_state: PipelineState):
    mock_report.generate.return_value = ReportOutput(
        title="Test Report",
        markdown="# Report",
        action_items=["action 1"],
    )
    result = report_node(base_state)
    assert result["report_title"] == "Test Report"
    assert "action 1" in result["report_action_items"]


# ── Memory ──────────────────────────────────────────────────────────────────


@patch("aegis.graph.nodes._memory_mgr")
def test_memory_node_stores(mock_memory, base_state: PipelineState):
    base_state["report_markdown"] = "# Final Report"
    result = memory_node(base_state)
    assert mock_memory.store.called
    assert result["agent_steps"] == [{"agent": "memory", "status": "ok"}]


# ── Error classification (warnings vs hard errors) ──────────────────────────


class TestClassifyRunErrors:
    """Transient provider failures must not fail an otherwise-complete run."""

    def test_no_errors(self):
        from aegis.api.routes import _classify_run_errors

        hard, warnings = _classify_run_errors([])
        assert hard == []
        assert warnings == []

    def test_quota_and_rate_limit_become_warnings(self):
        from aegis.api.routes import _classify_run_errors

        hard, warnings = _classify_run_errors([
            "report_generator: 429 RESOURCE_EXHAUSTED: Quota exceeded",
            "researcher: upstream 503 service unavailable",
            "validator: request timeout after 30s",
        ])
        assert hard == []
        assert len(warnings) == 3

    def test_genuine_errors_stay_hard(self):
        from aegis.api.routes import _classify_run_errors

        hard, warnings = _classify_run_errors([
            "reasoner: KeyError: 'evidence'",
            "critic: malformed response after parsing",
        ])
        assert len(hard) == 2
        assert warnings == []

    def test_mixed_errors_split_correctly(self):
        from aegis.api.routes import _classify_run_errors

        hard, warnings = _classify_run_errors([
            "report_generator: 429 rate limit exceeded",
            "memory: connection refused",
        ])
        assert hard == ["memory: connection refused"]
        assert len(warnings) == 1


def test_pipeline_response_carries_warnings():
    from aegis.api.schemas import PipelineResponse

    resp = PipelineResponse(success=True, warnings=["agent: 429 quota exceeded"])
    assert resp.success is True
    assert resp.warnings == ["agent: 429 quota exceeded"]
    # Backward-compatible default
    assert PipelineResponse().warnings == []
