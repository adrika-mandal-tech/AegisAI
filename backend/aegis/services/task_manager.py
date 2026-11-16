# backend/aegis/services/task_manager.py
"""Long-Running Task Execution Manager for AegisAI AgentSwarm (Section 13).

Provides:
- Non-blocking background task execution with explicit lifecycle states:
  PENDING -> PLANNING -> RUNNING -> WAITING_FOR_TOOL -> VALIDATING -> REVISING
  -> GENERATING_ARTIFACT -> COMPLETED | FAILED | CANCELLED
- Thread-safe cancellation tokens checked between and inside graph nodes
- LangGraph MemorySaver checkpointing + JSON disk persistence per execution_id
- Session isolation and resumability for interrupted or multi-step executions
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from aegis.core.config import settings
from aegis.core.logging import get_logger
from aegis.graph.pipeline import build_pipeline
from aegis.graph.state import (
    SSE_EVENT_AGENT_COMPLETE,
    SSE_EVENT_AGENT_OUTPUT,
    SSE_EVENT_AGENT_START,
    make_initial_state,
    serialize_charts,
)

log = get_logger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskState(str, Enum):
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING_FOR_TOOL = "WAITING_FOR_TOOL"
    VALIDATING = "VALIDATING"
    REVISING = "REVISING"
    GENERATING_ARTIFACT = "GENERATING_ARTIFACT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}
)


class TaskRecord(BaseModel):
    """Serializable execution record for a long-running multi-agent task."""

    model_config = {"extra": "ignore"}

    execution_id: str
    session_id: str = "default"
    user_query: str
    doc_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    requested_formats: list[str] = Field(default_factory=list)
    state: TaskState = TaskState.PENDING
    current_agent: str = ""
    progress_percent: int = 0
    retry_count: int = 0
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    completed_at: str | None = None
    agent_status: dict[str, str] = Field(default_factory=dict)
    agent_events: list[dict[str, Any]] = Field(default_factory=list)
    execution_plan: dict[str, Any] = Field(default_factory=dict)
    subtasks: list[dict[str, Any]] = Field(default_factory=list)
    agent_work_illustrations: list[dict[str, Any]] = Field(default_factory=list)
    response_metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[str] = Field(default_factory=list)
    chart_data: list[dict[str, Any]] = Field(default_factory=list)
    table_data: list[dict[str, Any]] = Field(default_factory=list)
    analysis_results: dict[str, Any] = Field(default_factory=dict)
    report_title: str = ""
    report_markdown: str = ""
    action_items: list[str] = Field(default_factory=list)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses (`TaskStatusResponse` / `TaskListResponse`)."""
        dumped = self.model_dump()
        dumped["state"] = self.state.value if isinstance(self.state, TaskState) else str(self.state)
        dumped["progress_pct"] = int(self.progress_percent)
        dumped["agent_steps"] = [
            {"agent": ag, "status": st} for ag, st in (self.agent_status or {}).items()
        ]
        dumped["events_count"] = len(self.agent_events or [])
        return dumped


_AGENT_TO_STAGE_STATE: dict[str, TaskState] = {
    "supervisor": TaskState.PLANNING,
    "planner": TaskState.PLANNING,
    "retriever": TaskState.RUNNING,
    "researcher": TaskState.RUNNING,
    "tool_agent": TaskState.WAITING_FOR_TOOL,
    "validator": TaskState.VALIDATING,
    "reasoner": TaskState.RUNNING,
    "critic": TaskState.VALIDATING,
    "report_generator": TaskState.GENERATING_ARTIFACT,
    "evaluation": TaskState.RUNNING,
    "memory": TaskState.RUNNING,
}

_AGENT_PROGRESS: dict[str, int] = {
    "supervisor": 10,
    "planner": 20,
    "retriever": 35,
    "researcher": 45,
    "tool_agent": 55,
    "validator": 65,
    "reasoner": 75,
    "critic": 85,
    "report_generator": 92,
    "evaluation": 97,
    "memory": 99,
}


class TaskExecutionManager:
    """Manages background multi-agent executions, checkpoints, and cancellation."""

    def __init__(self, storage_dir: str | Path | None = None, max_workers: int = 4) -> None:
        base_dir = (
            Path(storage_dir)
            if storage_dir is not None
            else Path(settings.AEGIS_DOCUMENT_ROOT).parent / "tasks"
        )
        self._storage_dir = base_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: dict[str, TaskRecord] = {}
        self._cancel_flags: dict[str, threading.Event] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aegis-task")
        self._checkpointer = MemorySaver()
        self._load_persisted_tasks()

    def _load_persisted_tasks(self) -> None:
        try:
            for file_path in sorted(self._storage_dir.glob("exec-*.json")):
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                    record = TaskRecord(**data)
                    # Any non-terminal task from a previous process was interrupted
                    if record.state not in TERMINAL_STATES:
                        record.state = TaskState.FAILED
                        record.error = "Interrupted by server restart (resumable via /api/tasks/{id}/resume)"
                    self._records[record.execution_id] = record
                except Exception as exc:
                    log.warning("Failed to load task file", file=str(file_path), error=str(exc))
        except Exception as exc:
            log.warning("Task directory scan failed", error=str(exc))

    def _persist(self, record: TaskRecord) -> None:
        try:
            target = self._storage_dir / f"{record.execution_id}.json"
            target.write_text(
                json.dumps(record.model_dump(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Failed to persist task record", execution_id=record.execution_id, error=str(exc))

    def submit_task(
        self,
        user_query: str,
        *,
        session_id: str = "default",
        doc_id: str | None = None,
        document_ids: list[str] | None = None,
        requested_formats: list[str] | None = None,
        tool_settings: dict[str, Any] | None = None,
        execution_id: str | None = None,
    ) -> TaskRecord:
        """Enqueue a new background execution and return its initial PENDING record."""
        exec_id = execution_id or f"exec-{uuid.uuid4().hex[:12]}"
        cancel_event = threading.Event()
        record = TaskRecord(
            execution_id=exec_id,
            session_id=session_id or "default",
            user_query=user_query,
            doc_id=doc_id,
            document_ids=list(document_ids or ([doc_id] if doc_id else [])),
            requested_formats=list(requested_formats or []),
            state=TaskState.PENDING,
        )

        with self._lock:
            self._records[exec_id] = record
            self._cancel_flags[exec_id] = cancel_event
            self._persist(record)

        future = self._executor.submit(
            self._execute_task,
            exec_id,
            tool_settings or {},
        )
        with self._lock:
            self._futures[exec_id] = future

        return record

    def _execute_task(self, execution_id: str, tool_settings: dict[str, Any]) -> None:
        with self._lock:
            record = self._records.get(execution_id)
            cancel_event = self._cancel_flags.get(execution_id)
        if record is None or cancel_event is None:
            return

        if cancel_event.is_set():
            self._mark_cancelled(execution_id)
            return

        with self._lock:
            record.state = TaskState.PLANNING
            record.progress_percent = 5
            record.updated_at = _utc_now()
            self._persist(record)

        def _is_cancelled() -> bool:
            return cancel_event.is_set()

        def _on_agent_event(agent: str, event_type: str, output: str = "") -> None:
            with self._lock:
                rec = self._records.get(execution_id)
                if rec is None or rec.state == TaskState.CANCELLED:
                    return
                rec.updated_at = _utc_now()
                if event_type == SSE_EVENT_AGENT_START:
                    rec.current_agent = agent
                    rec.agent_status[agent] = "running"
                    if agent == "reasoner" and rec.retry_count > 0:
                        rec.state = TaskState.REVISING
                    else:
                        rec.state = _AGENT_TO_STAGE_STATE.get(agent, TaskState.RUNNING)
                    rec.progress_percent = max(rec.progress_percent, _AGENT_PROGRESS.get(agent, rec.progress_percent))
                elif event_type == SSE_EVENT_AGENT_OUTPUT:
                    rec.agent_events.append(
                        {
                            "agent": agent,
                            "type": event_type,
                            "output": output[:600],
                            "timestamp": _utc_now(),
                        }
                    )
                elif event_type == SSE_EVENT_AGENT_COMPLETE:
                    rec.agent_status[agent] = "completed"
                    if agent == "critic":
                        # Check if revision will happen next
                        rec.retry_count = rec.retry_count
                    rec.progress_percent = max(rec.progress_percent, _AGENT_PROGRESS.get(agent, rec.progress_percent))
                self._persist(rec)

        initial_state = make_initial_state(
            record.user_query,
            doc_id=record.doc_id,
            document_ids=record.document_ids,
            execution_id=execution_id,
            session_id=record.session_id,
            requested_formats=record.requested_formats,
            tool_settings=tool_settings,
            on_agent_event=_on_agent_event,
            cancel_check=_is_cancelled,
        )

        try:
            compiled = build_pipeline()
            result = compiled.invoke(initial_state)

            if cancel_event.is_set():
                self._mark_cancelled(execution_id)
                return

            with self._lock:
                rec = self._records[execution_id]
                rec.state = TaskState.COMPLETED
                rec.current_agent = ""
                rec.progress_percent = 100
                rec.retry_count = int(result.get("retry_count", 0) or 0)
                rec.updated_at = _utc_now()
                rec.completed_at = rec.updated_at
                rec.execution_plan = result.get("execution_plan", {})
                rec.subtasks = result.get("subtasks", [])
                rec.agent_status.update(result.get("agent_status", {}))
                rec.agent_work_illustrations = result.get("agent_work_illustrations", [])
                rec.response_metrics = result.get("response_metrics", {})
                rec.artifacts = result.get("generated_artifacts", [])
                rec.charts = serialize_charts(result.get("report_charts", []))
                rec.chart_data = result.get("generated_visualizations") or result.get("chart_data", [])
                rec.table_data = result.get("table_data", [])
                rec.analysis_results = result.get("analysis_results", {})
                rec.report_title = result.get("report_title", "")
                rec.report_markdown = result.get("report_markdown", "")
                rec.action_items = result.get("report_action_items", [])
                rec.evaluation = result.get("evaluation_results", {})
                rec.errors = list(result.get("errors", []))
                self._persist(rec)
        except Exception as exc:
            if cancel_event.is_set():
                self._mark_cancelled(execution_id)
                return
            log.exception("Background task failed", execution_id=execution_id)
            with self._lock:
                rec = self._records[execution_id]
                rec.state = TaskState.FAILED
                rec.error = str(exc)
                rec.errors.append(str(exc))
                rec.updated_at = _utc_now()
                rec.completed_at = rec.updated_at
                self._persist(rec)

    def _mark_cancelled(self, execution_id: str) -> None:
        with self._lock:
            rec = self._records.get(execution_id)
            if rec is None:
                return
            rec.state = TaskState.CANCELLED
            rec.current_agent = ""
            rec.error = "Execution cancelled by user"
            rec.updated_at = _utc_now()
            rec.completed_at = rec.updated_at
            self._persist(rec)

    def cancel_task(self, execution_id: str) -> TaskRecord | None:
        """Signal cancellation for an active execution."""
        with self._lock:
            rec = self._records.get(execution_id)
            if rec is None:
                return None
            if rec.state in TERMINAL_STATES:
                return rec
            flag = self._cancel_flags.get(execution_id)
            if flag is not None:
                flag.set()
            future = self._futures.get(execution_id)
            if future is not None:
                future.cancel()
            rec.state = TaskState.CANCELLED
            rec.error = "Execution cancelled by user"
            rec.updated_at = _utc_now()
            rec.completed_at = rec.updated_at
            self._persist(rec)
            return rec

    def resume_task(self, execution_id: str) -> TaskRecord | None:
        """Resume a failed or cancelled execution using its saved parameters."""
        with self._lock:
            existing = self._records.get(execution_id)
            if existing is None:
                return None
            if existing.state not in (TaskState.FAILED, TaskState.CANCELLED):
                return existing
            query = existing.user_query
            session_id = existing.session_id
            doc_id = existing.doc_id
            document_ids = list(existing.document_ids)
            requested_formats = list(existing.requested_formats)

        return self.submit_task(
            query,
            session_id=session_id,
            doc_id=doc_id,
            document_ids=document_ids,
            requested_formats=requested_formats,
            execution_id=execution_id,
        )

    def get_task(self, execution_id: str) -> TaskRecord | None:
        with self._lock:
            return self._records.get(execution_id)

    def list_tasks(
        self,
        *,
        session_id: str | None = None,
        limit: int = 30,
    ) -> list[TaskRecord]:
        with self._lock:
            items = list(self._records.values())
        if session_id:
            items = [t for t in items if t.session_id == session_id]
        items.sort(key=lambda r: r.updated_at, reverse=True)
        return items[:limit]


_task_manager: TaskExecutionManager | None = None
_task_mgr_lock = threading.Lock()


def get_task_manager() -> TaskExecutionManager:
    global _task_manager
    with _task_mgr_lock:
        if _task_manager is None:
            _task_manager = TaskExecutionManager()
        return _task_manager
