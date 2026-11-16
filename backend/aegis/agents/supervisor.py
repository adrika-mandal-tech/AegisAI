# backend/aegis/agents/supervisor.py
"""Supervisor agent — top-level intent classifier and dynamic swarm router (Section 7.1)."""

from __future__ import annotations

from typing import cast

from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider
from aegis.llm.factory import get_default_llm
from aegis.schemas.agent import AgentOutput, SupervisorDecision

log = get_logger(__name__)

_SUPERVISOR_SYSTEM_PROMPT: str = """\
You are the Supervisor agent — central orchestrator of the AegisAI 8-agent swarm:
[supervisor, planner, retriever, researcher, validator, reasoner, critic, report_generator].

Given a user objective:
1. Classify the user's intent, complexity (low | medium | high), and whether it requires
   document retrieval, external research, dataset statistical computation, visualizations,
   or downloadable deliverables (pdf, docx, pptx, xlsx, csv, html, md).
2. Summarize the routing strategy clearly for downstream agents in `summary` and `details`.

Return ONLY valid JSON matching the AgentOutput schema.
"""


class Supervisor:
    """Classifies user intent and dynamically assigns specialized agents."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_default_llm()

    @staticmethod
    def build_routing_decision(task: str, has_documents: bool = False) -> SupervisorDecision:
        """Deterministic fast routing heuristic enriched by task cues."""
        low = (task or "").lower()
        words = low.split()

        # Simple conversational greeting shortcut
        greetings = {"hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening"}
        if low.strip("!.? ") in greetings:
            return SupervisorDecision(
                intent_summary="Conversational greeting — fast direct response path.",
                task_type="conversational",
                task_complexity="low",
                selected_agents=["supervisor", "report_generator"],
                requires_retrieval=False,
                requires_research=False,
                requires_data_analysis=False,
                requires_visualizations=False,
                requested_deliverables=[],
            )

        wants_data = any(
            k in low
            for k in (
                "csv", "excel", "xlsx", "xls", "dataset", "spreadsheet", "correlation",
                "outlier", "regression", "mean", "median", "variance", "std", "statistic",
            )
        )
        wants_vis = wants_data or any(
            k in low
            for k in ("chart", "plot", "graph", "visual", "histogram", "heatmap", "boxplot", "scatter", "trend")
        )
        wants_research = any(
            k in low
            for k in ("latest", "web", "internet", "news", "market", "benchmark", "external", "research")
        ) or not has_documents

        deliverables: list[str] = []
        for fmt, triggers in (
            ("pdf", ("pdf",)),
            ("docx", ("docx", "word doc")),
            ("pptx", ("pptx", "presentation", "slide", "deck", "powerpoint")),
            ("xlsx", ("xlsx", "excel", "workbook", "spreadsheet")),
            ("csv", ("export csv", "csv file")),
            ("html", ("html",)),
            ("md", ("markdown",)),
        ):
            if any(t in low for t in triggers):
                deliverables.append(fmt)

        complexity = "high" if (len(words) > 35 or (wants_data and wants_research) or len(deliverables) >= 2) else (
            "low" if len(words) < 10 and not wants_data else "medium"
        )

        selected = ["supervisor", "planner", "retriever"]
        if wants_research:
            selected.append("researcher")
        selected.extend(["validator", "reasoner", "critic", "report_generator"])

        task_type = (
            "data_analysis" if wants_data else
            "document_qa" if has_documents and not wants_research else
            "research" if wants_research and not has_documents else
            "hybrid"
        )
        return SupervisorDecision(
            intent_summary=f"Route as {task_type} ({complexity} complexity) across {len(selected)} agents.",
            task_type=task_type,
            task_complexity=complexity,
            selected_agents=selected,
            requires_retrieval=True,
            requires_research=wants_research,
            requires_data_analysis=wants_data,
            requires_visualizations=wants_vis,
            requested_deliverables=deliverables,
        )

    def supervise(self, task: str, *, has_documents: bool = False) -> AgentOutput:
        """Analyse *task* at sub-millisecond speed and return structured routing guidance."""
        log.info("Supervisor.supervise", task_preview=(task or "")[:80])
        decision = self.build_routing_decision(task, has_documents=has_documents)

        deliverable_note = (
            f" | Deliverables: {', '.join(d.upper() for d in decision.requested_deliverables)}"
            if decision.requested_deliverables
            else ""
        )
        summary = (
            f"Classified objective as '{decision.task_type}' ({decision.task_complexity} complexity); "
            f"dispatching {len(decision.selected_agents)} specialist agents ({', '.join(decision.selected_agents)})"
            f"{deliverable_note}."
        )
        details = (
            f"Objective: {(task or 'General inquiry')[:240]} | "
            f"Retrieval={decision.requires_retrieval}, Research={decision.requires_research}, "
            f"DataAnalysis={decision.requires_data_analysis}, Visualizations={decision.requires_visualizations}"
        )
        return AgentOutput(
            success=True,
            summary=summary,
            details=details,
            metadata={"decision": decision.model_dump()},
        )
