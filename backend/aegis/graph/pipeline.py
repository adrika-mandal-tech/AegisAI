# backend/aegis/graph/pipeline.py
"""LangGraph StateGraph orchestration for AegisAI's eight-agent swarm.

Supports:
- Dynamic Supervisor routing (fast-path vs full swarm)
- Parallel execution of independent Retriever + Researcher branches
- Controlled Tool & Dataset Analysis execution
- Bounded Critic review & revision loop (Critic -> Reasoner/Retriever)
- Checkpointing via LangGraph MemorySaver when a thread_id is configured
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from aegis.graph.edges import (
    route_after_critic,
    route_after_evaluation,
    route_after_planner,
    route_after_reasoner,
    route_after_report,
    route_after_supervisor,
    route_after_tool_agent,
    route_after_validation,
)
from aegis.graph.nodes import (
    critic_node,
    evaluation_node,
    memory_node,
    planner_node,
    reasoner_node,
    report_node,
    researcher_node,
    retriever_node,
    supervisor_node,
    tool_node,
    validator_node,
)
from aegis.graph.state import REDUCER_FIELDS, PipelineState, repair_state


def _repair_wrapper(node_func):
    """Wrap a graph node so non-reducer state fields are validated while
    LangGraph reducer fields emit pure node deltas without duplication.
    """

    def wrapped(state: PipelineState) -> dict[str, Any]:
        result = node_func(state)
        if not isinstance(result, dict):
            return {}
        repaired = repair_state({**state, **result})  # type: ignore[arg-type]
        output: dict[str, Any] = {}
        for key, raw_val in result.items():
            if key in REDUCER_FIELDS:
                output[key] = raw_val
            else:
                output[key] = repaired.get(key, raw_val)
        return output

    return wrapped


def build_pipeline(*, checkpointer: Any = None):
    """Construct and compile the AegisAI LangGraph StateGraph."""
    workflow = StateGraph(PipelineState)

    # ── Add nodes with state-repair wrappers ───────────────────────────────
    workflow.add_node("supervisor", _repair_wrapper(supervisor_node))
    workflow.add_node("planner", _repair_wrapper(planner_node))
    workflow.add_node("retriever", _repair_wrapper(retriever_node))
    workflow.add_node("researcher", _repair_wrapper(researcher_node))
    workflow.add_node("tool_agent", _repair_wrapper(tool_node))
    workflow.add_node("validator", _repair_wrapper(validator_node))
    workflow.add_node("reasoner", _repair_wrapper(reasoner_node))
    workflow.add_node("critic", _repair_wrapper(critic_node))
    workflow.add_node("report_generator", _repair_wrapper(report_node))
    workflow.add_node("evaluation", _repair_wrapper(evaluation_node))
    workflow.add_node("memory", _repair_wrapper(memory_node))

    # ── Entry & dynamic routing ───────────────────────────────────────────
    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "planner": "planner",
            "reasoner": "reasoner",
            "__end__": END,
        },
    )

    # Planner fans out dynamically (parallel [retriever, researcher] or single branch)
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        ["retriever", "researcher", "tool_agent", "reasoner", END],
    )

    # Parallel branches converge at tool_agent
    workflow.add_edge("retriever", "tool_agent")
    workflow.add_edge("researcher", "tool_agent")

    workflow.add_conditional_edges(
        "tool_agent",
        route_after_tool_agent,
        {
            "validator": "validator",
            "reasoner": "reasoner",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "reasoner": "reasoner",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "reasoner",
        route_after_reasoner,
        {
            "critic": "critic",
            "report_generator": "report_generator",
            "__end__": END,
        },
    )

    # Bounded Critic revision loop: can route back to reasoner/retriever or approve -> report_generator
    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "reasoner": "reasoner",
            "retriever": "retriever",
            "report_generator": "report_generator",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "report_generator",
        route_after_report,
        {
            "evaluation": "evaluation",
            "__end__": END,
        },
    )

    workflow.add_conditional_edges(
        "evaluation",
        route_after_evaluation,
        {
            "memory": "memory",
            "__end__": END,
        },
    )

    workflow.add_edge("memory", END)

    if checkpointer is not None:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


# ── Singleton ───────────────────────────────────────────────────────────────

pipeline = build_pipeline()
