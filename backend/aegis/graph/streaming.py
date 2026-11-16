# backend/aegis/graph/streaming.py
"""Streaming pipeline runner — runs the LangGraph pipeline and yields SSE events.

Emits real-time agent lifecycle events, execution_id, per-agent work
illustrations, downloadable deliverables, and live per-response evaluation
metrics for the Workspace Sidebar.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aegis.core.logging import get_logger
from aegis.graph.pipeline import build_pipeline
from aegis.graph.state import (
    SSE_EVENT_AGENT_COMPLETE,
    SSE_EVENT_AGENT_OUTPUT,
    SSE_EVENT_AGENT_START,
    SSE_EVENT_ERROR,
    SSE_EVENT_PIPELINE_COMPLETE,
    PipelineState,
    make_initial_state,
    serialize_charts,
)

log = get_logger(__name__)


def _sse_format(event: str, data: dict[str, Any]) -> str:
    """Format a single SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def stream_pipeline(
    task: str,
    on_event: callable,
    on_run_complete: callable | None = None,
    doc_id: str | None = None,
    *,
    document_ids: list[str] | None = None,
    execution_id: str | None = None,
    session_id: str | None = None,
    requested_formats: list[str] | None = None,
    tool_settings: dict[str, Any] | None = None,
    record_pipeline_run: callable | None = None,
    cancel_check: callable | None = None,
) -> dict[str, Any]:
    """Run the full multi-agent pipeline and stream SSE events via *on_event*."""
    log.info("stream_pipeline: start", task=task[:80], doc_id=doc_id)

    compiled = build_pipeline()
    run_cb = record_pipeline_run or on_run_complete

    def _record_run_cb(
        task_str: str,
        report_title: str = "",
        agent_steps: list[dict[str, Any]] | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        if callable(run_cb):
            try:
                run_cb(task_str, report_title, agent_steps, success, error)
            except Exception:
                pass

    def _agent_event_cb(agent: str, event_type: str, output: str = "", status: str = "") -> None:
        if event_type == SSE_EVENT_AGENT_START:
            on_event(
                _sse_format(
                    SSE_EVENT_AGENT_START,
                    {
                        "agent": agent,
                        "status": status or "running",
                    },
                )
            )
        elif event_type == SSE_EVENT_AGENT_OUTPUT:
            on_event(
                _sse_format(
                    SSE_EVENT_AGENT_OUTPUT,
                    {
                        "agent": agent,
                        "output": output,
                    },
                )
            )
        elif event_type == SSE_EVENT_AGENT_COMPLETE:
            on_event(
                _sse_format(
                    SSE_EVENT_AGENT_COMPLETE,
                    {
                        "agent": agent,
                        "status": status or "ok",
                    },
                )
            )

    initial_state = make_initial_state(
        task,
        doc_id=doc_id,
        document_ids=document_ids,
        execution_id=execution_id,
        session_id=session_id,
        requested_formats=requested_formats,
        tool_settings=tool_settings,
        on_agent_event=_agent_event_cb,
        record_pipeline_run=_record_run_cb,
        cancel_check=cancel_check,
    )

    try:
        # Run synchronous LangGraph invoke in worker thread so FastAPI event loop flushes SSE live!
        result: PipelineState = await asyncio.to_thread(compiled.invoke, initial_state)
        on_event(
            _sse_format(
                SSE_EVENT_PIPELINE_COMPLETE,
                {
                    "status": "ok",
                    "execution_id": result.get("execution_id", ""),
                    "session_id": result.get("session_id", "default"),
                    "task_type": result.get("task_type", "hybrid"),
                    "task_complexity": result.get("task_complexity", "medium"),
                    "report_title": result.get("report_title", ""),
                    "report_markdown": result.get("report_markdown", ""),
                    "action_items": result.get("report_action_items", []),
                    "charts": serialize_charts(result.get("report_charts", [])),
                    "chart_data": result.get("generated_visualizations") or result.get("chart_data", []),
                    "table_data": result.get("table_data", []),
                    "analysis_results": result.get("analysis_results", {}),
                    "execution_plan": result.get("execution_plan", {}),
                    "subtasks": result.get("subtasks", []),
                    "artifacts": result.get("generated_artifacts", []),
                    "agent_work_illustrations": result.get("agent_work_illustrations", []),
                    "response_metrics": result.get("response_metrics", {}),
                    "evaluation": result.get("evaluation_results", {}),
                },
            )
        )
        log.info("stream_pipeline: completed", execution_id=result.get("execution_id"))
        return dict(result)
    except Exception as exc:
        log.exception("stream_pipeline: failed")
        on_event(
            _sse_format(
                SSE_EVENT_ERROR,
                {
                    "agent": "error",
                    "error": str(exc),
                    "output": str(exc),
                    "status": "error",
                },
            )
        )
        return {"success": False, "error": str(exc)}
