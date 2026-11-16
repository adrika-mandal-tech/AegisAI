# backend/aegis/graph/nodes.py
"""LangGraph node functions for AegisAI's eight-agent autonomous swarm.

Each node:
1. Checks cancellation state before executing.
2. Emits real-time SSE lifecycle events (agent_start, agent_output, agent_complete).
3. Executes its specialized agent with typed inputs and outputs.
4. Produces parallel-safe delta updates for shared reducers (evidence, agent_outputs,
   agent_status, agent_work_illustrations, execution_history, agent_steps, errors).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable

from aegis.agents import (
    Critic,
    MemoryManager,
    Planner,
    Reasoner,
    ReportGenerator,
    Researcher,
    Retriever,
    Supervisor,
    ToolAgent,
    Validator,
)
from aegis.core.logging import get_logger
from aegis.graph.state import (
    SSE_EVENT_AGENT_COMPLETE,
    SSE_EVENT_AGENT_OUTPUT,
    SSE_EVENT_AGENT_START,
    PipelineState,
)
from aegis.llm.factory import get_default_llm
from aegis.schemas.agent import AgentWorkIllustration, RetrievedChunk

log = get_logger(__name__)

# ── Module-level agent instances (shared across pipeline runs) ──────────
_llm = get_default_llm()
_supervisor = Supervisor(llm=_llm)
_planner = Planner(llm=_llm)
_researcher = Researcher(llm=_llm)
_retriever = Retriever(llm=_llm)
_validator = Validator(llm=_llm)
_reasoner = Reasoner(llm=_llm)
_critic = Critic(llm=_llm)
_report_gen = ReportGenerator(llm=_llm)
_memory_mgr = MemoryManager()
_tool_agent = ToolAgent(llm=_llm)


class TaskCancelledException(RuntimeError):
    """Raised inside a node when the user cancels a running task."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_cancelled(state: PipelineState, agent_name: str) -> None:
    cancel_fn = state.get("cancel_check")
    if callable(cancel_fn) and cancel_fn():
        log.info("Node cancelled by user", agent=agent_name, execution_id=state.get("execution_id"))
        raise TaskCancelledException(f"Execution cancelled before {agent_name}")


def _to_evidence_objects(evidence_items: list[dict[str, Any]]) -> list[Any]:
    """Convert raw evidence dicts to ``Evidence`` models, skipping invalid items."""
    from aegis.schemas.agent import coerce_evidence

    objects: list[Any] = []
    for item in evidence_items or []:
        evidence = coerce_evidence(item)
        if evidence is None:
            continue
        objects.append(evidence)
    return objects


def _emit(
    state: PipelineState,
    agent: str,
    event_type: str,
    output: str = "",
    status: str = "",
) -> None:
    """Send an SSE event if a callback is registered on the state."""
    cb = state.get("on_agent_event")
    if cb:
        try:
            cb(agent, event_type, output)
        except TypeError:
            cb(agent, event_type, output, status)


def _make_illustration(
    *,
    agent: str,
    label: str,
    role: str,
    status: str,
    summary: str,
    key_outputs: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    duration_ms: float = 0.0,
) -> dict[str, Any]:
    return AgentWorkIllustration(
        agent=agent,
        label=label,
        role=role,
        status=status,
        summary=summary[:500],
        key_outputs=[str(x)[:240] for x in (key_outputs or []) if x][:6],
        metrics=metrics or {},
        duration_ms=round(duration_ms, 1),
    ).model_dump()


def with_retry(max_retries: int = 2, base_delay: float = 1.0):
    """Decorator adding bounded retry logic for transient LLM/network errors."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(state: PipelineState) -> dict[str, Any]:
            agent_name = func.__name__.replace("_node", "")
            if agent_name == "report":
                agent_name = "report_generator"
            elif agent_name == "tool":
                agent_name = "tool_agent"

            last_exception: Exception | None = None

            for attempt in range(max_retries):
                try:
                    _check_cancelled(state, agent_name)
                    return func(state)
                except TaskCancelledException as cancel_exc:
                    _emit(state, agent_name, "agent_error", output="Task cancelled", status="cancelled")
                    return {
                        "agent_status": {agent_name: "cancelled"},
                        "execution_history": [
                            {
                                "agent": agent_name,
                                "event": "cancelled",
                                "timestamp": _utc_now(),
                            }
                        ],
                        "errors": [f"{agent_name}: {cancel_exc}"],
                        "agent_steps": [{"agent": agent_name, "status": "cancelled"}],
                    }
                except Exception as exc:
                    last_exception = exc
                    error_str = str(exc).lower()

                    is_retryable = any(
                        token in error_str
                        for token in (
                            "429",
                            "resource_exhausted",
                            "quota",
                            "rate limit",
                            "503",
                            "service unavailable",
                            "unavailable",
                            "timeout",
                            "deadline exceeded",
                            "connection",
                        )
                    )

                    if is_retryable and attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        log.warning(
                            f"Node {agent_name} failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s",
                            error=str(exc)[:200],
                        )
                        _emit(
                            state,
                            agent_name,
                            "agent_error",
                            output=f"Retrying... (attempt {attempt + 1})",
                            status="retrying",
                        )
                        time.sleep(delay)
                        continue

                    log.exception(f"Node {agent_name} failed permanently")
                    _emit(state, agent_name, "agent_error", output=str(exc)[:500], status="error")
                    return {
                        "agent_status": {agent_name: "error"},
                        "execution_history": [
                            {
                                "agent": agent_name,
                                "event": "error",
                                "error": str(exc)[:300],
                                "timestamp": _utc_now(),
                            }
                        ],
                        "errors": [f"{agent_name}: {exc}"],
                        "agent_steps": [{"agent": agent_name, "status": "error", "error": str(exc)}],
                    }

            return {
                "agent_status": {agent_name: "error"},
                "errors": [f"{agent_name}: Failed after {max_retries} attempts - {last_exception}"],
                "agent_steps": [{"agent": agent_name, "status": "error", "error": str(last_exception)}],
            }

        return wrapper

    return decorator


# ── Node 1: Supervisor ──────────────────────────────────────────────────


@with_retry(max_retries=2, base_delay=1.0)
def supervisor_node(state: PipelineState) -> dict[str, Any]:
    """Classify user intent, task complexity, deliverables, and agent routing."""
    t0 = time.perf_counter()
    task_text = state.get("task") or state.get("user_query") or ""
    log.info("pipeline_node: supervisor", task=task_text[:80])
    _emit(state, "supervisor", SSE_EVENT_AGENT_START, status="running")

    result = _supervisor.supervise(task_text)
    duration_ms = (time.perf_counter() - t0) * 1000

    output = result.summary if result.summary else "Task analysis completed."
    task_type = getattr(result, "task_type", "") or result.metadata.get("task_type", "hybrid")
    task_complexity = getattr(result, "task_complexity", "") or result.metadata.get("task_complexity", "medium")
    selected_agents = list(getattr(result, "selected_agents", []) or result.metadata.get("selected_agents", []))
    requires_retrieval = getattr(result, "requires_retrieval", True)
    requires_research = getattr(result, "requires_research", True)
    requires_data_analysis = getattr(result, "requires_data_analysis", False)
    requires_visualizations = getattr(result, "requires_visualizations", False)
    requested_deliverables = list(getattr(result, "requested_deliverables", []) or [])

    existing_formats = list(state.get("requested_formats") or [])
    merged_formats = list(dict.fromkeys(existing_formats + requested_deliverables))

    _emit(state, "supervisor", SSE_EVENT_AGENT_OUTPUT, output=output, status="running")
    _emit(state, "supervisor", SSE_EVENT_AGENT_COMPLETE, status="ok")

    updates: dict[str, Any] = {
        "supervisor_analysis": result.summary,
        "supervisor_details": result.details,
        "task_type": task_type,
        "task_complexity": task_complexity,
        "requires_retrieval": requires_retrieval,
        "requires_research": requires_research,
        "requires_data_analysis": requires_data_analysis,
        "requires_visualizations": requires_visualizations,
        "agent_status": {"supervisor": "completed"},
        "agent_steps": [{"agent": "supervisor", "status": "ok"}],
        "agent_outputs": {"supervisor": [output]},
        "agent_work_illustrations": [
            _make_illustration(
                agent="supervisor",
                label="Supervisor",
                role="Intent Classification & Dynamic Swarm Routing",
                status="completed",
                summary=output,
                key_outputs=[
                    f"Task Type: {task_type}",
                    f"Complexity: {task_complexity}",
                    f"Assigned Agents: {', '.join(selected_agents) if selected_agents else 'full swarm'}",
                ],
                metrics={"complexity": task_complexity, "agent_count": len(selected_agents) or 8},
                duration_ms=duration_ms,
            )
        ],
        "execution_history": [
            {
                "agent": "supervisor",
                "event": "completed",
                "task_type": task_type,
                "task_complexity": task_complexity,
                "duration_ms": round(duration_ms, 1),
                "timestamp": _utc_now(),
            }
        ],
    }
    if selected_agents:
        updates["agent_assignments"] = selected_agents
    if merged_formats:
        updates["requested_formats"] = merged_formats
    return updates


# ── Node 2: Planner ─────────────────────────────────────────────────────


@with_retry(max_retries=2, base_delay=1.0)
def planner_node(state: PipelineState) -> dict[str, Any]:
    """Decompose user objective into ordered subtasks with dependencies and validation criteria."""
    t0 = time.perf_counter()
    task_text = state.get("task") or state.get("user_query") or ""
    log.info("pipeline_node: planner", task=task_text[:80])
    _emit(state, "planner", SSE_EVENT_AGENT_START, status="running")

    plan = _planner.plan(task_text, task_type=str(state.get("task_type", "")))
    duration_ms = (time.perf_counter() - t0) * 1000

    plan_steps_dumped = [s.model_dump() for s in plan.steps]
    subtasks_dumped = (
        [st.model_dump() for st in plan.subtasks]
        if getattr(plan, "subtasks", None)
        else [s.to_subtask().model_dump() for s in plan.steps]
    )

    plan_lines = [f"Objective: {plan.objective}", "Steps:"]
    for step in plan.steps:
        plan_lines.append(f"- Step {step.step}: [{step.agent}] {step.description}")
    plan_text = "\n".join(plan_lines)

    execution_plan = {
        "objective": plan.objective,
        "estimated_complexity": getattr(plan, "estimated_complexity", state.get("task_complexity", "medium")),
        "steps": plan_steps_dumped,
        "subtasks": subtasks_dumped,
    }

    _emit(state, "planner", SSE_EVENT_AGENT_OUTPUT, output=plan_text, status="running")
    _emit(state, "planner", SSE_EVENT_AGENT_COMPLETE, status="ok")

    return {
        "plan_objective": plan.objective,
        "plan_steps": plan_steps_dumped,
        "execution_plan": execution_plan,
        "subtasks": subtasks_dumped,
        "agent_status": {"planner": "completed"},
        "agent_steps": [{"agent": "planner", "status": "ok"}],
        "agent_outputs": {"planner": [plan_text]},
        "agent_work_illustrations": [
            _make_illustration(
                agent="planner",
                label="Planner",
                role="Structured Task Decomposition & Dependency Graph",
                status="completed",
                summary=plan.objective or f"Generated {len(plan.steps)}-step execution plan.",
                key_outputs=[f"[{s.agent}] {s.description}" for s in plan.steps[:5]],
                metrics={"subtask_count": len(subtasks_dumped)},
                duration_ms=duration_ms,
            )
        ],
        "execution_history": [
            {
                "agent": "planner",
                "event": "completed",
                "subtasks": len(subtasks_dumped),
                "duration_ms": round(duration_ms, 1),
                "timestamp": _utc_now(),
            }
        ],
    }


# ── Node 3: Retriever ───────────────────────────────────────────────────


@with_retry(max_retries=2, base_delay=1.0)
def retriever_node(state: PipelineState) -> dict[str, Any]:
    """Retrieve semantic + lexical chunks from Qdrant with provenance."""
    t0 = time.perf_counter()
    task_text = state.get("task") or state.get("user_query") or ""
    log.info("pipeline_node: retriever", task=task_text[:80])
    _emit(state, "retriever", SSE_EVENT_AGENT_START, status="running")

    doc_ids = state.get("document_ids") or []
    if len(doc_ids) > 1:
        result = _retriever.retrieve(task_text, doc_id=state.get("doc_id"), document_ids=doc_ids)
    else:
        result = _retriever.retrieve(task_text, doc_id=state.get("doc_id"))

    duration_ms = (time.perf_counter() - t0) * 1000

    new_evidence: list[dict[str, Any]] = []
    retrieved_chunks: list[RetrievedChunk] = []
    retrieved_docs_summary: list[dict[str, Any]] = []

    for ev in result.evidence_items or []:
        new_evidence.append(ev)
        src_meta = ev.get("source_metadata", {}) if isinstance(ev, dict) else {}
        chunk = RetrievedChunk(
            chunk_id=str(src_meta.get("chunk_id", ev.get("id", ""))),
            document_id=str(src_meta.get("document_id", "")),
            text=str(ev.get("supporting_passage", "")),
            score=float(src_meta.get("score", ev.get("relevance", 0.0)) or 0.0),
            source=str(ev.get("source", "unknown")),
            page=src_meta.get("page"),
            section_heading=str(src_meta.get("section_heading", src_meta.get("section_path", ""))),
            chunk_index=int(src_meta.get("chunk_index", 0) or 0),
            extraction_method=str(src_meta.get("extraction_method", "")),
            embedding_model=str(src_meta.get("embedding_model", "")),
            metadata=src_meta,
        )
        retrieved_chunks.append(chunk)
        retrieved_docs_summary.append(chunk.model_dump())

    if result.content:
        retrieved_context = result.content
    elif retrieved_chunks:
        retrieved_context = "\n\n".join(
            f"[source={c.source} page={c.page} score={c.score:.3f}] {c.text}"
            for c in retrieved_chunks
        )
    else:
        retrieved_context = ""

    output = retrieved_context if retrieved_context else "No relevant context found in the knowledge base."
    _emit(state, "retriever", SSE_EVENT_AGENT_OUTPUT, output=output[:500], status="running")
    _emit(state, "retriever", SSE_EVENT_AGENT_COMPLETE, status="ok")

    unique_sources = sorted({c.source for c in retrieved_chunks if c.source})
    avg_score = (
        round(sum(c.score for c in retrieved_chunks) / len(retrieved_chunks), 3)
        if retrieved_chunks
        else 0.0
    )

    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieved_documents": retrieved_docs_summary,
        "retrieved_context": retrieved_context,
        "evidence": new_evidence,
        "evidence_items": new_evidence,
        "agent_status": {"retriever": "completed"},
        "agent_steps": [{"agent": "retriever", "status": "ok"}],
        "agent_outputs": {"retriever": [output]},
        "agent_work_illustrations": [
            _make_illustration(
                agent="retriever",
                label="Retriever",
                role="Qdrant Semantic & Hybrid Document Retrieval",
                status="completed",
                summary=(
                    f"Retrieved {len(retrieved_chunks)} grounded passage(s) across "
                    f"{len(unique_sources)} document(s) (avg relevance {avg_score:.2f})."
                    if retrieved_chunks
                    else "Queried Qdrant vector store; no matching indexed chunks above threshold."
                ),
                key_outputs=[f"{s}" for s in unique_sources[:4]] or [output[:180]],
                metrics={
                    "chunks_retrieved": len(retrieved_chunks),
                    "unique_documents": len(unique_sources),
                    "avg_similarity": avg_score,
                },
                duration_ms=duration_ms,
            )
        ],
        "execution_history": [
            {
                "agent": "retriever",
                "event": "completed",
                "chunks_retrieved": len(retrieved_chunks),
                "duration_ms": round(duration_ms, 1),
                "timestamp": _utc_now(),
            }
        ],
    }


# ── Node 4: Researcher ──────────────────────────────────────────────────


@with_retry(max_retries=2, base_delay=1.0)
def researcher_node(state: PipelineState) -> dict[str, Any]:
    """Conduct multi-source external research without fabricating citations."""
    t0 = time.perf_counter()
    task_text = state.get("task") or state.get("user_query") or ""
    log.info("pipeline_node: researcher", task=task_text[:80])
    _emit(state, "researcher", SSE_EVENT_AGENT_START, status="running")

    tool_settings = state.get("tool_settings") or {}
    if tool_settings.get("enableWebSearch") is False:
        output = "External web search disabled by workspace tool settings."
        _emit(state, "researcher", SSE_EVENT_AGENT_OUTPUT, output=output, status="running")
        _emit(state, "researcher", SSE_EVENT_AGENT_COMPLETE, status="ok")
        return {
            "research_results": "",
            "research_findings": [],
            "agent_status": {"researcher": "skipped"},
            "agent_steps": [{"agent": "researcher", "status": "ok"}],
            "agent_outputs": {"researcher": [output]},
            "agent_work_illustrations": [
                _make_illustration(
                    agent="researcher",
                    label="Researcher",
                    role="Multi-Source External Research & Synthesis",
                    status="skipped",
                    summary=output,
                    duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
            ],
        }

    result = _researcher.research(
        task_text,
        retrieved_context=state.get("retrieved_context", ""),
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    output = result.content if result.content else "No additional research findings available."
    _emit(state, "researcher", SSE_EVENT_AGENT_OUTPUT, output=output[:500], status="running")
    _emit(state, "researcher", SSE_EVENT_AGENT_COMPLETE, status="ok")

    new_evidence: list[dict[str, Any]] = []
    if result.evidence_items:
        from aegis.schemas.agent import coerce_evidence

        for ev in result.evidence_items:
            coerced = coerce_evidence(ev)
            if coerced is not None:
                new_evidence.append(coerced.model_dump())

    return {
        "research_results": result.content,
        "research_findings": new_evidence,
        "evidence": new_evidence,
        "evidence_items": new_evidence,
        "agent_status": {"researcher": "completed"},
        "agent_steps": [{"agent": "researcher", "status": "ok"}],
        "agent_outputs": {"researcher": [output]},
        "agent_work_illustrations": [
            _make_illustration(
                agent="researcher",
                label="Researcher",
                role="Multi-Source External Research & Synthesis",
                status="completed",
                summary=output[:280],
                key_outputs=[ev.get("source", "") for ev in new_evidence[:4] if ev.get("source")],
                metrics={"findings_count": len(new_evidence)},
                duration_ms=duration_ms,
            )
        ],
        "execution_history": [
            {
                "agent": "researcher",
                "event": "completed",
                "findings_count": len(new_evidence),
                "duration_ms": round(duration_ms, 1),
                "timestamp": _utc_now(),
            }
        ],
    }


# ── Controlled Tool & Dataset Profiling Node ────────────────────────────


def tool_node(state: PipelineState) -> dict[str, Any]:
    """Execute allowlisted tools (dataset profiling, document inspection, MCP, web)."""
    t0 = time.perf_counter()
    log.info("pipeline_node: tool_agent")
    _emit(state, "tool_agent", SSE_EVENT_AGENT_START, status="running")

    tool_calls = list(state.get("tool_calls") or [])
    outputs: list[str] = []
    results: list[dict[str, Any]] = []
    analysis_results: dict[str, Any] = dict(state.get("analysis_results") or {})
    generated_visualizations: list[dict[str, Any]] = list(state.get("generated_visualizations") or [])
    table_data: list[dict[str, Any]] = list(state.get("table_data") or [])

    # Auto-trigger tabular dataset profiling if a CSV/XLSX/XLS/JSON document is targeted
    try:
        from aegis.services.data_analysis import analyze_registered_documents_for_state

        auto_analysis = analyze_registered_documents_for_state(state)
        if auto_analysis:
            analysis_results.update(auto_analysis.get("analysis_results", {}))
            generated_visualizations.extend(auto_analysis.get("visualizations", []))
            table_data.extend(auto_analysis.get("tables", []))
            summary_msg = auto_analysis.get("summary", "")
            if summary_msg:
                outputs.append(summary_msg)
                results.append(
                    {
                        "tool": "dataset_profiler",
                        "payload": {"doc_id": state.get("doc_id")},
                        "success": True,
                        "summary": summary_msg,
                        "details": summary_msg,
                        "errors": [],
                    }
                )
    except ImportError:
        pass
    except Exception as exc:
        log.warning("tool_node: auto dataset analysis skipped", error=str(exc)[:200])

    for call in tool_calls:
        tool = call.get("tool", "")
        payload = call.get("payload", {})
        result = _tool_agent.run(tool, payload)
        outputs.append(result.summary or f"{tool}: {result.success}")
        results.append(
            {
                "tool": tool,
                "payload": payload,
                "success": result.success,
                "summary": result.summary,
                "details": result.details,
                "errors": result.errors,
            }
        )

    duration_ms = (time.perf_counter() - t0) * 1000
    output_text = "\n".join(outputs) if outputs else "No external tool calls required."
    _emit(state, "tool_agent", SSE_EVENT_AGENT_OUTPUT, output=output_text[:500], status="running")
    _emit(state, "tool_agent", SSE_EVENT_AGENT_COMPLETE, status="ok")

    updates: dict[str, Any] = {
        "tool_results": results,
        "agent_status": {"tool_agent": "completed"},
        "agent_steps": [{"agent": "tool_agent", "status": "ok"}],
        "execution_history": [
            {
                "agent": "tool_agent",
                "event": "completed",
                "tool_calls": len(results),
                "duration_ms": round(duration_ms, 1),
                "timestamp": _utc_now(),
            }
        ],
    }
    if analysis_results:
        updates["analysis_results"] = analysis_results
    if generated_visualizations:
        updates["generated_visualizations"] = generated_visualizations
        updates["chart_data"] = generated_visualizations
    if table_data:
        updates["table_data"] = table_data
    return updates


# ── Node 5: Validator ───────────────────────────────────────────────────


@with_retry(max_retries=2, base_delay=1.0)
def validator_node(state: PipelineState) -> dict[str, Any]:
    """Verify factual grounding, detect contradictions, and check numerical consistency."""
    t0 = time.perf_counter()
    log.info("pipeline_node: validator")
    _emit(state, "validator", SSE_EVENT_AGENT_START, status="running")

    combined = (
        f"Retrieved: {state.get('retrieved_context', '')}\n\n"
        f"Research: {state.get('research_results', '')}\n\n"
        f"Data Analysis: {state.get('analysis_results', {})}"
    )
    evidence_items = state.get("evidence_items") or state.get("evidence") or []
    evidence_objects = _to_evidence_objects(evidence_items) or None
    result = _validator.validate(
        combined,
        context=state.get("task", ""),
        evidence_items=evidence_objects,
    )

    citation_validation = None
    report_markdown = state.get("report_markdown", "")
    if report_markdown and evidence_objects:
        try:
            citation_validation = _validator.validate_citations(report_markdown, evidence_objects)
        except Exception as exc:
            log.warning("validator: citation validation failed", error=str(exc)[:200])

    duration_ms = (time.perf_counter() - t0) * 1000
    validation_details = (
        f"Validation Score: {result.score:.2f}\n"
        f"Issues: {result.issues}\n"
        f"Suggestions: {result.suggestions}\n"
    )
    if citation_validation:
        validation_details += (
            f"\nCitation Validation Score: {citation_validation.score:.2f}\n"
            f"Citation Issues: {citation_validation.issues}\n"
        )

    _emit(state, "validator", SSE_EVENT_AGENT_OUTPUT, output=validation_details, status="running")
    _emit(state, "validator", SSE_EVENT_AGENT_COMPLETE, status="ok")

    validation_results_dict = result.model_dump()
    if citation_validation:
        validation_results_dict["citation_validation_score"] = citation_validation.score

    return {
        "validation_results": validation_results_dict,
        "validation_score": result.score,
        "validation_issues": result.issues,
        "validation_suggestions": result.suggestions,
        "citation_validation_score": citation_validation.score if citation_validation else None,
        "agent_status": {"validator": "completed"},
        "agent_steps": [{"agent": "validator", "status": "ok"}],
        "agent_outputs": {"validator": [validation_details]},
        "agent_work_illustrations": [
            _make_illustration(
                agent="validator",
                label="Validator",
                role="Evidence Verification & Contradiction Detection",
                status="completed",
                summary=f"Validated {len(evidence_items)} evidence item(s) with confidence {result.score:.2f}.",
                key_outputs=(result.issues[:3] or ["All retrieved claims passed consistency checks."]),
                metrics={
                    "validation_score": round(result.score, 3),
                    "issues_found": len(result.issues),
                    "is_hallucination": result.is_hallucination,
                },
                duration_ms=duration_ms,
            )
        ],
        "execution_history": [
            {
                "agent": "validator",
                "event": "completed",
                "score": round(result.score, 3),
                "duration_ms": round(duration_ms, 1),
                "timestamp": _utc_now(),
            }
        ],
    }


# ── Node 6: Reasoner & Analyst ──────────────────────────────────────────


@with_retry(max_retries=2, base_delay=1.0)
def reasoner_node(state: PipelineState) -> dict[str, Any]:
    """Perform multi-step logical synthesis and real Python numerical/statistical analysis."""
    t0 = time.perf_counter()
    is_revision = not state.get("critic_approved", True)
    next_retry_count = int(state.get("retry_count", 0) or 0) + (1 if is_revision else 0)

    log.info("pipeline_node: reasoner", is_revision=is_revision, retry_count=next_retry_count)
    _emit(state, "reasoner", SSE_EVENT_AGENT_START, status="revising" if is_revision else "running")

    critic_note = ""
    if is_revision:
        fb = state.get("critic_feedback") or {}
        critic_note = (
            f"\n\nCRITIC REVISION MANDATE (Pass {next_retry_count}):\n"
            f"Weaknesses to fix: {state.get('critique_weaknesses', [])}\n"
            f"Instructions: {fb.get('revision_instructions', state.get('critique_assessment', ''))}\n"
        )

    analysis_context = ""
    if state.get("analysis_results"):
        analysis_context = f"\nComputed Dataset Statistics (Python): {state['analysis_results']}\n"

    evidence = (
        f"Task: {state.get('task', '')}\n\n"
        f"Retrieved context: {state.get('retrieved_context', '')}\n"
        f"Research: {state.get('research_results', '')}\n"
        f"Validation issues: {state.get('validation_issues', '')}"
        f"{analysis_context}"
        f"{critic_note}"
    )
    evidence_items = state.get("evidence_items") or state.get("evidence") or []
    evidence_objects = _to_evidence_objects(evidence_items) or None

    result = _reasoner.reason(
        evidence,
        evidence_items=evidence_objects,
        analysis_results=state.get("analysis_results") or None,
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    output = result.details if result.details else (result.summary if result.summary else "Reasoning complete.")
    _emit(state, "reasoner", SSE_EVENT_AGENT_OUTPUT, output=output[:500], status="running")
    _emit(state, "reasoner", SSE_EVENT_AGENT_COMPLETE, status="ok")

    updates: dict[str, Any] = {
        "reasoning_summary": result.summary,
        "reasoning_details": result.details,
        "retry_count": next_retry_count,
        "agent_status": {"reasoner": "revised" if is_revision else "completed"},
        "agent_steps": [{"agent": "reasoner", "status": "ok"}],
        "agent_outputs": {"reasoner": [result.summary, result.details]},
        "agent_work_illustrations": [
            _make_illustration(
                agent="reasoner",
                label="Reasoner & Analyst",
                role="Multi-Step Synthesis & Quantitative Computation",
                status="revised" if is_revision else "completed",
                summary=result.summary or output[:260],
                key_outputs=[line.strip() for line in output.splitlines() if line.strip()][:4],
                metrics={
                    "revision_pass": next_retry_count,
                    "computed_metrics": len(result.metadata.get("computations", {}))
                    if isinstance(getattr(result, "metadata", None), dict)
                    else 0,
                },
                duration_ms=duration_ms,
            )
        ],
        "execution_history": [
            {
                "agent": "reasoner",
                "event": "revised" if is_revision else "completed",
                "retry_count": next_retry_count,
                "duration_ms": round(duration_ms, 1),
                "timestamp": _utc_now(),
            }
        ],
    }

    if isinstance(getattr(result, "metadata", None), dict) and result.metadata.get("analysis_results"):
        merged_analysis = dict(state.get("analysis_results") or {})
        merged_analysis.update(result.metadata["analysis_results"])
        updates["analysis_results"] = merged_analysis

    return updates


# ── Node 7: Critic ──────────────────────────────────────────────────────


@with_retry(max_retries=2, base_delay=1.0)
def critic_node(state: PipelineState) -> dict[str, Any]:
    """Adversarially review reasoning for hallucinations, math errors, and gaps."""
    t0 = time.perf_counter()
    log.info("pipeline_node: critic")
    _emit(state, "critic", SSE_EVENT_AGENT_START, status="running")

    content = state.get("reasoning_details", "") or state.get("reasoning_summary", "")
    # Pass grounding context when supported by Critic.critique
    try:
        result = _critic.critique(
            content,
            task=state.get("task", ""),
            evidence_context=state.get("retrieved_context", ""),
            retry_count=int(state.get("retry_count", 0) or 0),
        )
    except TypeError:
        result = _critic.critique(content)

    duration_ms = (time.perf_counter() - t0) * 1000
    approved = getattr(result, "approved", True)
    quality_score = float(getattr(result, "quality_score", 0.8) or 0.8)

    critique_text = (
        f"Approved: {approved} (Quality: {quality_score:.2f})\n"
        f"Overall Assessment: {result.overall_assessment}\n"
        f"Strengths: {result.strengths}\n"
        f"Weaknesses: {result.weaknesses}"
    )
    _emit(state, "critic", SSE_EVENT_AGENT_OUTPUT, output=critique_text[:500], status="running")
    _emit(state, "critic", SSE_EVENT_AGENT_COMPLETE, status="ok")

    critic_feedback = result.model_dump()
    return {
        "critique_strengths": result.strengths,
        "critique_weaknesses": result.weaknesses,
        "critique_assessment": result.overall_assessment,
        "critic_approved": approved,
        "critic_feedback": critic_feedback,
        "agent_status": {"critic": "approved" if approved else "rejected_for_revision"},
        "agent_steps": [{"agent": "critic", "status": "ok"}],
        "agent_outputs": {"critic": [critique_text]},
        "agent_work_illustrations": [
            _make_illustration(
                agent="critic",
                label="Critic",
                role="Adversarial Quality, Logic & Hallucination Audit",
                status="completed" if approved else "revised",
                summary=result.overall_assessment or critique_text[:240],
                key_outputs=(
                    result.improvement_suggestions[:3]
                    or result.weaknesses[:3]
                    or result.strengths[:3]
                ),
                metrics={"approved": approved, "quality_score": round(quality_score, 3)},
                duration_ms=duration_ms,
            )
        ],
        "execution_history": [
            {
                "agent": "critic",
                "event": "approved" if approved else "rejected",
                "quality_score": round(quality_score, 3),
                "duration_ms": round(duration_ms, 1),
                "timestamp": _utc_now(),
            }
        ],
    }


# ── Node 8: Report Generator & Deliverable Engine ───────────────────────


@with_retry(max_retries=2, base_delay=1.0)
def report_node(state: PipelineState) -> dict[str, Any]:
    """Synthesize the 12-section report and generate downloadable deliverables."""
    t0 = time.perf_counter()
    log.info("pipeline_node: report_generator")
    _emit(state, "report_generator", SSE_EVENT_AGENT_START, status="running")

    evidence_items = state.get("evidence_items") or state.get("evidence") or []
    data = {
        "execution_id": state.get("execution_id", ""),
        "session_id": state.get("session_id", "default"),
        "task": state.get("task", ""),
        "task_type": state.get("task_type", "hybrid"),
        "plan": state.get("plan_objective", ""),
        "plan_steps": state.get("plan_steps", []),
        "subtasks": state.get("subtasks", []),
        "retrieved_context": state.get("retrieved_context", ""),
        "research_results": state.get("research_results", ""),
        "validation_score": state.get("validation_score", 0.0),
        "validation_issues": state.get("validation_issues", []),
        "validation_suggestions": state.get("validation_suggestions", []),
        "analysis_results": state.get("analysis_results", {}),
        "reasoning_output": state.get("reasoning_details", "") or state.get("reasoning_summary", ""),
        "reasoning_details": state.get("reasoning_details", ""),
        "reasoning_summary": state.get("reasoning_summary", ""),
        "critique_assessment": state.get("critique_assessment", ""),
        "critique_strengths": state.get("critique_strengths", []),
        "critique_weaknesses": state.get("critique_weaknesses", []),
        "supervisor_analysis": state.get("supervisor_analysis", ""),
        "supervisor_details": state.get("supervisor_details", ""),
        "agent_outputs": state.get("agent_outputs", {}),
        "evidence_items": evidence_items,
        "force_charts": True,
        "force_metrics": True,
        "force_evals": True,
        "command_type": state.get("command_type", "general"),
        "document_metrics": state.get("document_metrics", {}),
        "chart_data": state.get("generated_visualizations") or state.get("chart_data", []),
        "table_data": state.get("table_data", []),
        "evaluation_results": state.get("evaluation_results", {}),
        "requested_formats": state.get("requested_formats", []),
    }

    result = _report_gen.generate(data)

    # Generate real downloadable artifact files when ArtifactEngine is available
    artifacts_dumped: list[dict[str, Any]] = [
        a.model_dump() if hasattr(a, "model_dump") else dict(a)
        for a in (getattr(result, "artifacts", None) or [])
    ]
    if not artifacts_dumped and state.get("execution_id"):
        try:
            from aegis.services.artifact_generator import generate_execution_artifacts

            artifacts_dumped = generate_execution_artifacts(
                execution_id=state["execution_id"],
                title=result.title,
                markdown=result.markdown,
                sections=result.sections,
                action_items=result.action_items,
                charts=result.charts,
                tables=state.get("table_data", []),
                evidence_items=evidence_items,
                analysis_results=state.get("analysis_results", {}),
                requested_formats=state.get("requested_formats") or None,
            )
        except ImportError:
            pass
        except Exception as exc:
            log.warning("report_node: artifact file generation skipped", error=str(exc)[:200])

    duration_ms = (time.perf_counter() - t0) * 1000
    output = result.title if result.title else "Report generated."
    _emit(state, "report_generator", SSE_EVENT_AGENT_OUTPUT, output=output, status="running")
    _emit(state, "report_generator", SSE_EVENT_AGENT_COMPLETE, status="ok")

    charts_dumped = [
        c.model_dump() if hasattr(c, "model_dump") else c
        for c in (result.charts or [])
    ]

    return {
        "report_title": result.title,
        "report_markdown": result.markdown,
        "final_answer": result.markdown,
        "report_action_items": result.action_items,
        "report_charts": result.charts,
        "generated_visualizations": charts_dumped or state.get("generated_visualizations", []),
        "generated_artifacts": artifacts_dumped,
        "agent_status": {"report_generator": "completed"},
        "agent_steps": [{"agent": "report_generator", "status": "ok"}],
        "agent_outputs": {"report_generator": [output]},
        "agent_work_illustrations": [
            _make_illustration(
                agent="report_generator",
                label="Report Generator",
                role="Structured Report Synthesis & Deliverable Export",
                status="completed",
                summary=f"Generated '{result.title}' with {len(result.charts or [])} chart(s) and {len(artifacts_dumped)} downloadable file(s).",
                key_outputs=[a.get("filename", "") for a in artifacts_dumped[:5]] or result.action_items[:3],
                metrics={
                    "charts_count": len(result.charts or []),
                    "artifacts_count": len(artifacts_dumped),
                    "word_count": len((result.markdown or "").split()),
                },
                duration_ms=duration_ms,
            )
        ],
        "execution_history": [
            {
                "agent": "report_generator",
                "event": "completed",
                "artifacts": len(artifacts_dumped),
                "duration_ms": round(duration_ms, 1),
                "timestamp": _utc_now(),
            }
        ],
    }


# ── Post-Report Live Evaluation Node ────────────────────────────────────


@with_retry(max_retries=2, base_delay=1.0)
def evaluation_node(state: PipelineState) -> dict[str, Any]:
    """Compute DeepEval, Ragas, Evidence Gap, and the 7 live Sidebar metrics."""
    t0 = time.perf_counter()
    log.info("pipeline_node: evaluation")
    _emit(state, "evaluation", SSE_EVENT_AGENT_START, status="running")

    combined_content = "\n\n".join(
        [
            state.get("retrieved_context", ""),
            state.get("research_results", ""),
            state.get("reasoning_details", ""),
            state.get("report_markdown", ""),
        ]
    )

    evidence_items = state.get("evidence_items") or state.get("evidence") or []
    evidence_objects = _to_evidence_objects(evidence_items)

    from aegis.agents.evaluator import Evaluator

    evaluator = Evaluator()
    metrics, evaluation, charts, tables = evaluator.evaluate(
        combined_content,
        evidence_objects,
        task=state.get("task", ""),
    )

    response_metrics: dict[str, Any] = {}
    try:
        from aegis.agents.evidence_gap_analyzer import analyze_evidence_gaps
        from aegis.evaluation.deepeval import evaluate_pipeline_run
        from aegis.evaluation.ragas import evaluate_rag_pipeline

        deepeval_result = evaluate_pipeline_run(
            task=state.get("task", ""),
            report_markdown=state.get("report_markdown", ""),
            pipeline_trace=state.get("agent_steps", []),
        )
        ragas_result = evaluate_rag_pipeline(
            task=state.get("task", ""),
            retrieved_context=state.get("retrieved_context", ""),
            report_markdown=state.get("report_markdown", ""),
        )
        gap_analysis = analyze_evidence_gaps(
            response=state.get("report_markdown", ""),
            evidence_items=evidence_items,
            task=state.get("task", ""),
        )

        # Compute the 7 live Sidebar response metrics grounded in real pipeline data
        ragas_m = ragas_result.get("metrics", {})
        deepeval_m = deepeval_result.get("metrics", {})

        faithfulness = float(ragas_m.get("faithfulness", evaluation.factual_accuracy) or 0.0)
        hallucination_rate = round(
            max(0.0, min(1.0, float(ragas_m.get("report_hallucination_score", 1.0 - faithfulness)))),
            4,
        )
        context_precision = round(
            max(0.0, min(1.0, float(ragas_m.get("context_precision", ragas_m.get("context_utilization", evaluation.citation_quality))))),
            4,
        )
        context_recall = round(
            max(0.0, min(1.0, float(ragas_m.get("context_recall", 1.0 - gap_analysis.get("gap_score", 0.3))))),
            4,
        )

        steps = state.get("agent_steps") or []
        ok_steps = sum(1 for s in steps if s.get("status") == "ok")
        task_success_rate = round(ok_steps / len(steps), 4) if steps else 1.0

        tool_results = state.get("tool_results") or []
        if tool_results:
            ok_tools = sum(1 for tr in tool_results if tr.get("success", True))
            tool_call_accuracy = round(ok_tools / len(tool_results), 4)
        else:
            tool_call_accuracy = 1.0

        accuracy = round(
            max(
                0.0,
                min(
                    1.0,
                    0.5 * float(evaluation.factual_accuracy)
                    + 0.3 * faithfulness
                    + 0.2 * float(state.get("validation_score", evaluation.factual_accuracy) or evaluation.factual_accuracy),
                ),
            ),
            4,
        )
        correctness = round(
            max(
                0.0,
                min(
                    1.0,
                    0.4 * accuracy
                    + 0.3 * float(deepeval_m.get("answer_relevancy", deepeval_result.get("overall_score", evaluation.overall_quality)) or 0.75)
                    + 0.3 * float(evaluation.coherence_score or 0.75),
                ),
            ),
            4,
        )

        response_metrics = {
            "hallucination_rate": hallucination_rate,
            "accuracy": accuracy,
            "correctness": correctness,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "task_success_rate": task_success_rate,
            "tool_call_accuracy": tool_call_accuracy,
        }

        evaluation_results = {
            "deepeval": deepeval_result,
            "ragas": ragas_result,
            "evidence_gap_analysis": gap_analysis,
            "hallucination_rate": hallucination_rate,
            "response_metrics": response_metrics,
            "custom_metrics": metrics.model_dump(),
            "custom_evaluation": evaluation.model_dump(),
            "charts": [c.model_dump() for c in charts],
            "tables": [t.model_dump() for t in tables],
        }
    except Exception as exc:
        log.warning("External evaluation fallback used", error=str(exc))
        response_metrics = {
            "hallucination_rate": round(float(evaluation.hallucination_score), 4),
            "accuracy": round(float(evaluation.factual_accuracy), 4),
            "correctness": round(float(evaluation.overall_quality), 4),
            "context_precision": round(float(evaluation.citation_quality), 4),
            "context_recall": round(float(evaluation.coverage_score), 4),
            "task_success_rate": 1.0,
            "tool_call_accuracy": 1.0,
        }
        evaluation_results = {
            "response_metrics": response_metrics,
            "hallucination_rate": response_metrics["hallucination_rate"],
            "custom_metrics": metrics.model_dump(),
            "custom_evaluation": evaluation.model_dump(),
            "charts": [c.model_dump() for c in charts],
            "tables": [t.model_dump() for t in tables],
            "error": str(exc),
        }

    duration_ms = (time.perf_counter() - t0) * 1000
    _emit(state, "evaluation", SSE_EVENT_AGENT_OUTPUT, output="Evaluation complete.", status="running")
    _emit(state, "evaluation", SSE_EVENT_AGENT_COMPLETE, status="ok")

    existing_charts = list(state.get("generated_visualizations") or state.get("chart_data") or [])
    eval_charts = [c.model_dump() for c in charts]
    existing_tables = list(state.get("table_data") or [])
    eval_tables = [t.model_dump() for t in tables]

    return {
        "evaluation_results": evaluation_results,
        "response_metrics": response_metrics,
        "document_metrics": metrics.model_dump(),
        "chart_data": existing_charts + eval_charts,
        "table_data": existing_tables + eval_tables,
        "agent_status": {"evaluation": "completed"},
        "agent_steps": [{"agent": "evaluation", "status": "ok"}],
        "agent_outputs": {"evaluation": ["Evaluation complete."]},
        "execution_history": [
            {
                "agent": "evaluation",
                "event": "completed",
                "response_metrics": response_metrics,
                "duration_ms": round(duration_ms, 1),
                "timestamp": _utc_now(),
            }
        ],
    }


# ── Session Memory Node ─────────────────────────────────────────────────


def memory_node(state: PipelineState) -> dict[str, Any]:
    """Persist completed execution summary without polluting RAG document retrieval."""
    log.info("pipeline_node: memory")
    _emit(state, "memory", SSE_EVENT_AGENT_START, status="running")
    _memory_mgr.store(
        state.get("task", ""),
        {
            "execution_id": state.get("execution_id", ""),
            "session_id": state.get("session_id", "default"),
            "report": state.get("report_markdown", ""),
            "reasoning": state.get("reasoning_summary", ""),
            "response_metrics": state.get("response_metrics", {}),
        },
    )
    record_run = state.get("record_pipeline_run")
    if callable(record_run):
        try:
            record_run(
                state.get("task", ""),
                state.get("report_title", ""),
                state.get("agent_steps", []),
                True,
                None,
            )
        except Exception:
            pass
    _emit(state, "memory", SSE_EVENT_AGENT_COMPLETE, status="ok")
    return {"agent_steps": [{"agent": "memory", "status": "ok"}]}
