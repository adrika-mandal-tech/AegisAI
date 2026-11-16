# backend/aegis/graph/edges.py
"""Dynamic edge-routing logic for the AegisAI LangGraph StateGraph.

Supports:
1. Cancellation short-circuiting at every transition.
2. Supervisor-driven dynamic agent selection (skipping unnecessary stages for simple tasks).
3. Parallel fan-out from Planner into Retriever + Researcher when both internal documents
   and external research are applicable, or single-branch routing when only one is needed.
4. Bounded Critic rejection & revision loop back to Reasoner or Retriever (capped by max_retries).
"""

from __future__ import annotations

from aegis.core.logging import get_logger
from aegis.graph.state import PipelineState

log = get_logger(__name__)

# ── Node name constants (must match node names in pipeline.py) ──────────

SUPERVISOR = "supervisor"
PLANNER = "planner"
RETRIEVER = "retriever"
RESEARCHER = "researcher"
TOOL_AGENT = "tool_agent"
VALIDATOR = "validator"
REASONER = "reasoner"
CRITIC = "critic"
REPORTER = "report_generator"
EVALUATION = "evaluation"
MEMORY = "memory"
END = "__end__"

DEFAULT_MAX_RETRIES = 2


def _is_cancelled(state: PipelineState) -> bool:
    cancel_fn = state.get("cancel_check")
    if callable(cancel_fn):
        try:
            return bool(cancel_fn())
        except Exception:
            return False
    return False


def route_after_supervisor(state: PipelineState) -> str:
    """Route from Supervisor to Planner, or fast-path to Reasoner ONLY on trivial non-document tasks."""
    if _is_cancelled(state):
        log.info("route_after_supervisor: task cancelled -> END")
        return END

    has_docs = bool(state.get("doc_id")) or bool(state.get("document_ids"))
    has_deliverables = bool(state.get("requested_formats"))
    assignments = state.get("agent_assignments") or []
    complexity = state.get("task_complexity", "medium")
    if (
        not has_docs
        and not has_deliverables
        and not state.get("requires_retrieval", True)
        and assignments
        and complexity == "low"
        and PLANNER not in assignments
        and REASONER in assignments
    ):
        log.info("route_after_supervisor: low-complexity fast path -> reasoner")
        return REASONER

    return PLANNER


def route_after_planner(state: PipelineState) -> list[str] | str:
    """Dynamically fan out to parallel [retriever, researcher] or selected single branch."""
    if _is_cancelled(state):
        log.info("route_after_planner: task cancelled -> END")
        return END

    assignments = state.get("agent_assignments") or []
    enable_parallel = state.get("enable_parallel", True)

    # Default / full swarm execution: parallel retrieval + external research
    if not assignments:
        if enable_parallel:
            return [RETRIEVER, RESEARCHER]
        return RETRIEVER

    want_retriever = RETRIEVER in assignments or bool(state.get("doc_id")) or bool(state.get("document_ids"))
    want_researcher = RESEARCHER in assignments and state.get("requires_research", True)

    branches: list[str] = []
    if want_retriever:
        branches.append(RETRIEVER)
    if want_researcher:
        branches.append(RESEARCHER)

    if len(branches) == 2 and enable_parallel:
        log.info("route_after_planner: parallel fan-out -> retriever + researcher")
        return branches
    if len(branches) >= 1:
        log.info("route_after_planner: single branch", target=branches[0])
        return branches[0]
    if TOOL_AGENT in assignments or state.get("requires_data_analysis"):
        return TOOL_AGENT
    return REASONER


def route_after_tool_agent(state: PipelineState) -> str:
    """Proceed to Validator unless cancelled or fast-pathed on non-document tasks."""
    if _is_cancelled(state):
        return END
    has_docs = bool(state.get("doc_id")) or bool(state.get("document_ids")) or bool(state.get("evidence"))
    assignments = state.get("agent_assignments") or []
    if not has_docs and assignments and VALIDATOR not in assignments:
        return REASONER
    return VALIDATOR


def route_after_validation(state: PipelineState) -> str:
    """Route after Validator: flag low-confidence evidence and enter Reasoner."""
    if _is_cancelled(state):
        return END
    score = state.get("validation_score", 0.0)
    if score < 0.3:
        log.warning("Low validation score — Reasoner will apply strict grounding", score=score)
    return REASONER


def route_after_reasoner(state: PipelineState) -> str:
    """Route from Reasoner to Critic (or directly to Reporter if Critic skipped on trivial tasks)."""
    if _is_cancelled(state):
        return END
    has_docs = bool(state.get("doc_id")) or bool(state.get("document_ids"))
    assignments = state.get("agent_assignments") or []
    if not has_docs and assignments and CRITIC not in assignments:
        return REPORTER
    return CRITIC


def route_after_critic(state: PipelineState) -> str:
    """Bounded Critic review loop: reject & route back for revision, or approve -> Reporter."""
    if _is_cancelled(state):
        return END

    approved = state.get("critic_approved", True)
    retry_count = int(state.get("retry_count", 0) or 0)
    max_retries = int(state.get("max_retries", DEFAULT_MAX_RETRIES) or DEFAULT_MAX_RETRIES)

    if not approved and retry_count < max_retries:
        feedback = state.get("critic_feedback") or {}
        target = str(feedback.get("target_revision_agent", REASONER))
        if target not in (REASONER, RETRIEVER):
            target = REASONER
        log.info(
            "Critic rejected output — routing back for revision",
            retry_count=retry_count + 1,
            max_retries=max_retries,
            target=target,
        )
        return target

    return REPORTER


def route_after_report(state: PipelineState) -> str:
    """After report & deliverable generation, proceed to live evaluation."""
    if _is_cancelled(state):
        return END
    return EVALUATION


def route_after_evaluation(state: PipelineState) -> str:
    """After evaluation, persist run summary to memory."""
    if _is_cancelled(state):
        return END
    return MEMORY


def should_continue(state: PipelineState) -> str:
    """Determine if the pipeline should stop based on cancellation or error budget."""
    if _is_cancelled(state):
        return END
    errors = state.get("errors", [])
    if len(errors) > 4:
        log.error("Too many pipeline errors — aborting", count=len(errors))
        return END
    return "continue"
