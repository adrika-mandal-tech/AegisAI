# backend/aegis/graph/state.py
"""LangGraph state definition for the AegisAI multi-agent execution platform.

Holds the complete typed execution context as it flows through dynamic routing,
parallel retrieval/research branches, bounded Critic revision loops, data
analysis, visualization, and deliverable generation.
"""

from __future__ import annotations

import operator
import uuid
from collections.abc import Callable
from typing import Annotated, Any

from typing_extensions import TypedDict

from aegis.core.logging import get_logger
from aegis.schemas.agent import RetrievedChunk

log = get_logger(__name__)


# ── Parallel-Safe State Reducers ───────────────────────────────────────────


def merge_dicts(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge dictionary updates safely across parallel or sequential nodes."""
    merged = dict(left or {})
    if right:
        merged.update(right)
    return merged


def merge_agent_outputs(
    left: dict[str, list[str]] | None, right: dict[str, list[str]] | None
) -> dict[str, list[str]]:
    """Merge per-agent output lists without overwriting earlier agents."""
    merged: dict[str, list[str]] = {k: list(v) for k, v in (left or {}).items()}
    for agent, items in (right or {}).items():
        if not isinstance(items, list):
            items = [str(items)]
        existing = merged.get(agent, [])
        for item in items:
            if item and item not in existing:
                existing.append(item)
        merged[agent] = existing
    return merged


def merge_evidence_lists(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Merge and deduplicate evidence items across parallel Retriever & Researcher."""
    combined: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in list(left or []) + list(right or []):
        if not isinstance(item, dict):
            continue
        key = (
            str(item.get("source", ""))[:120],
            str(item.get("claim", ""))[:160],
            str(item.get("supporting_passage", ""))[:160],
        )
        if key not in seen:
            seen.add(key)
            combined.append(item)
    return combined


def merge_illustrations(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Merge per-agent work illustrations, updating an agent's entry on revision."""
    by_agent: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in list(left or []) + list(right or []):
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent", ""))
        if not agent:
            continue
        if agent not in by_agent:
            order.append(agent)
        by_agent[agent] = item
    return [by_agent[a] for a in order]


def merge_agent_steps(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Append agent step traces while keeping parallel retriever/researcher order canonical."""
    existing = list(left or [])
    incoming = list(right or [])
    if (
        len(existing) >= 2
        and len(incoming) == 1
        and incoming[0].get("agent") == "retriever"
        and existing[-1].get("agent") == "researcher"
        and existing[-2].get("agent") == "planner"
    ):
        return existing[:-1] + incoming + [existing[-1]]
    return existing + incoming


class PipelineState(TypedDict, total=False):
    """Shared typed state that travels through the LangGraph StateGraph.

    Includes both the Section 12 canonical orchestration fields and all
    existing AegisAI fields so all existing callers and unit tests remain
    100% backward-compatible.
    """

    # ── Execution Identity & Session ──────────────────────────────────────
    execution_id: str
    session_id: str
    user_query: str
    task: str  # Original user request (alias kept in sync with user_query)
    doc_id: str | None
    document_ids: list[str]
    requested_formats: list[str]
    enable_parallel: bool

    # ── Supervisor & Classification ───────────────────────────────────────
    task_type: str
    task_complexity: str
    agent_assignments: list[str]
    requires_retrieval: bool
    requires_research: bool
    requires_data_analysis: bool
    requires_visualizations: bool
    supervisor_analysis: str
    supervisor_details: str

    # ── Planner & Subtasks ────────────────────────────────────────────────
    execution_plan: dict[str, Any]
    subtasks: list[dict[str, Any]]
    plan_objective: str
    plan_steps: list[dict[str, Any]]

    # ── Retrieval ─────────────────────────────────────────────────────────
    retrieved_documents: list[dict[str, Any]]
    retrieved_chunks: list[RetrievedChunk]
    retrieved_context: str

    # ── Researcher ────────────────────────────────────────────────────────
    research_findings: list[dict[str, Any]]
    research_results: str

    # ── Evidence (Parallel-Safe Reducers) ─────────────────────────────────
    evidence: Annotated[list[dict[str, Any]], merge_evidence_lists]
    evidence_items: Annotated[list[dict[str, Any]], merge_evidence_lists]

    # ── Validator ─────────────────────────────────────────────────────────
    validation_results: dict[str, Any]
    validation_score: float
    validation_issues: list[str]
    validation_suggestions: list[str]
    citation_validation_score: float | None

    # ── Reasoner & Data Analyst ───────────────────────────────────────────
    analysis_results: dict[str, Any]
    reasoning_summary: str
    reasoning_details: str

    # ── Critic & Bounded Revision Loop ────────────────────────────────────
    critic_feedback: dict[str, Any]
    critic_approved: bool
    retry_count: int
    max_retries: int
    critique_strengths: list[str]
    critique_weaknesses: list[str]
    critique_assessment: str

    # ── Visualizations, Deliverables & Final Answer ───────────────────────
    generated_visualizations: list[dict[str, Any]]
    generated_artifacts: list[dict[str, Any]]
    final_answer: str
    report_title: str
    report_markdown: str
    report_action_items: list[str]
    report_charts: list[Any]
    command_type: str

    # ── Evaluation & Live Sidebar Metrics ─────────────────────────────────
    evaluation_results: dict[str, Any]
    response_metrics: dict[str, Any]
    document_metrics: dict[str, Any]
    chart_data: list[dict[str, Any]]
    table_data: list[dict[str, Any]]

    # ── Controlled Tool Layer ─────────────────────────────────────────────
    tool_settings: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    # ── Memory ────────────────────────────────────────────────────────────
    memory_entries: list[dict[str, Any]]

    # ── Traceability, Agent Status & Error Reducers ───────────────────────
    agent_outputs: Annotated[dict[str, list[str]], merge_agent_outputs]
    agent_status: Annotated[dict[str, str], merge_dicts]
    agent_work_illustrations: Annotated[list[dict[str, Any]], merge_illustrations]
    execution_history: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    agent_steps: Annotated[list[dict[str, Any]], merge_agent_steps]

    # ── Streaming & Task Lifecycle Callbacks ──────────────────────────────
    on_agent_event: Callable[..., None] | None
    record_pipeline_run: Callable[..., None] | None
    cancel_check: Callable[[], bool] | None


# ── SSE Event helpers ─────────────────────────────────────────────────────

SSE_EVENT_AGENT_START = "agent_start"
SSE_EVENT_AGENT_OUTPUT = "agent_output"
SSE_EVENT_AGENT_COMPLETE = "agent_complete"
SSE_EVENT_PIPELINE_COMPLETE = "pipeline_complete"
SSE_EVENT_TASK_STATE = "task_state"
SSE_EVENT_ERROR = "error"


def make_sse_data(
    agent: str,
    event_type: str,
    output: str = "",
    status: str = "",
) -> str:
    """Build an SSE ``data:`` line from structured fields."""
    import json

    payload = {
        "agent": agent,
        "type": event_type,
        "output": output,
        "status": status,
    }
    return f"data: {json.dumps(payload)}\n\n"


def serialize_charts(charts: list[Any]) -> list[str]:
    """Normalize report charts to plain base64 image strings."""
    serialized: list[str] = []
    for chart in charts or []:
        if isinstance(chart, str):
            if chart:
                serialized.append(chart)
            continue
        image_data = ""
        if isinstance(chart, dict):
            image_data = str(chart.get("image_data") or "")
        else:
            image_data = str(getattr(chart, "image_data", "") or "")
        if image_data:
            serialized.append(image_data)
    return serialized


# ── Initial State Factory & State Repair ──────────────────────────────────

REDUCER_FIELDS: frozenset[str] = frozenset(
    {
        "errors",
        "agent_steps",
        "execution_history",
        "agent_outputs",
        "agent_status",
        "evidence",
        "evidence_items",
        "agent_work_illustrations",
    }
)

_REPAIR_DEFAULTS: dict[str, Any] = {
    "execution_id": "",
    "session_id": "default",
    "user_query": "",
    "task": "",
    "doc_id": None,
    "document_ids": [],
    "requested_formats": [],
    "enable_parallel": True,
    "task_type": "hybrid",
    "task_complexity": "medium",
    "agent_assignments": [],
    "requires_retrieval": True,
    "requires_research": True,
    "requires_data_analysis": False,
    "requires_visualizations": False,
    "supervisor_analysis": "",
    "supervisor_details": "",
    "execution_plan": {},
    "subtasks": [],
    "plan_objective": "",
    "plan_steps": [],
    "retrieved_documents": [],
    "retrieved_chunks": [],
    "retrieved_context": "",
    "research_findings": [],
    "research_results": "",
    "evidence": [],
    "evidence_items": [],
    "validation_results": {},
    "validation_score": 0.0,
    "validation_issues": [],
    "validation_suggestions": [],
    "citation_validation_score": None,
    "analysis_results": {},
    "reasoning_summary": "",
    "reasoning_details": "",
    "critic_feedback": {},
    "critic_approved": True,
    "retry_count": 0,
    "max_retries": 2,
    "critique_strengths": [],
    "critique_weaknesses": [],
    "critique_assessment": "",
    "generated_visualizations": [],
    "generated_artifacts": [],
    "final_answer": "",
    "report_title": "",
    "report_markdown": "",
    "report_action_items": [],
    "report_charts": [],
    "command_type": "general",
    "agent_outputs": {},
    "agent_status": {},
    "agent_work_illustrations": [],
    "evaluation_results": {},
    "response_metrics": {},
    "document_metrics": {},
    "chart_data": [],
    "table_data": [],
    "tool_settings": {},
    "tool_calls": [],
    "tool_results": [],
    "memory_entries": [],
    "execution_history": [],
    "errors": [],
    "agent_steps": [],
    "on_agent_event": None,
    "record_pipeline_run": None,
    "cancel_check": None,
}

_NULLABLE_FIELDS: frozenset[str] = frozenset(
    {
        "doc_id",
        "citation_validation_score",
        "on_agent_event",
        "record_pipeline_run",
        "cancel_check",
    }
)


def make_initial_state(
    task: str,
    *,
    doc_id: str | None = None,
    document_ids: list[str] | None = None,
    execution_id: str | None = None,
    session_id: str | None = None,
    requested_formats: list[str] | None = None,
    tool_settings: dict[str, Any] | None = None,
    on_agent_event: Callable[..., None] | None = None,
    record_pipeline_run: Callable[..., None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    max_retries: int = 2,
) -> PipelineState:
    """Create a complete, cleanly initialized PipelineState."""
    exec_id = execution_id or f"exec-{uuid.uuid4().hex[:12]}"
    sess_id = session_id or "default"
    doc_list = list(document_ids or ([] if not doc_id else [doc_id]))
    base = {
        k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
        for k, v in _REPAIR_DEFAULTS.items()
    }
    base.update(
        {
            "execution_id": exec_id,
            "session_id": sess_id,
            "user_query": task,
            "task": task,
            "doc_id": doc_id,
            "document_ids": doc_list,
            "requested_formats": list(requested_formats or []),
            "tool_settings": dict(tool_settings or {}),
            "on_agent_event": on_agent_event,
            "record_pipeline_run": record_pipeline_run,
            "cancel_check": cancel_check,
            "max_retries": max_retries,
        }
    )
    return base  # type: ignore[return-value]


def repair_state(state: PipelineState) -> PipelineState:
    """Repair missing or invalid fields in pipeline state."""
    repaired = dict(state)
    repaired_fields: list[str] = []

    for key, default in _REPAIR_DEFAULTS.items():
        if key not in repaired:
            repaired[key] = (
                list(default)
                if isinstance(default, list)
                else dict(default)
                if isinstance(default, dict)
                else default
            )
            repaired_fields.append(key)
        elif repaired[key] is None and key not in _NULLABLE_FIELDS:
            repaired[key] = (
                list(default)
                if isinstance(default, list)
                else dict(default)
                if isinstance(default, dict)
                else default
            )
            repaired_fields.append(key)
        elif isinstance(default, list) and not isinstance(repaired[key], list):
            value = repaired[key]
            if isinstance(value, str):
                repaired[key] = [value]
            elif isinstance(value, (tuple, set)):
                repaired[key] = list(value)
            else:
                repaired[key] = []
            repaired_fields.append(key)
        elif isinstance(default, dict) and not isinstance(repaired[key], dict):
            repaired[key] = {}
            repaired_fields.append(key)
        elif isinstance(default, str) and not isinstance(repaired[key], str):
            repaired[key] = str(repaired[key]) if repaired[key] is not None else ""
            repaired_fields.append(key)

    # Keep user_query and task synchronized
    if repaired.get("task") and not repaired.get("user_query"):
        repaired["user_query"] = repaired["task"]
    elif repaired.get("user_query") and not repaired.get("task"):
        repaired["task"] = repaired["user_query"]

    if not repaired.get("execution_id"):
        repaired["execution_id"] = f"exec-{uuid.uuid4().hex[:12]}"

    if repaired_fields:
        log.debug("state_repair", repaired_count=len(repaired_fields))
    return repaired  # type: ignore[return-value]


# Alias
create_initial_state = make_initial_state
