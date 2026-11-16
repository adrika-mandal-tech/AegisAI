# backend/aegis/agents/planner.py
"""Planner agent — decomposes objectives into structured subtasks with dependencies (Section 7.2)."""

from __future__ import annotations

from typing import Any, cast

from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider
from aegis.llm.factory import get_default_llm
from aegis.schemas.agent import AgentPlan, PlanStep, SubtaskItem

log = get_logger(__name__)

_PLANNER_SYSTEM_PROMPT: str = """\
You are the Planner agent in the AegisAI 8-agent platform.
Break the user's objective into 4–7 concrete, ordered steps with explicit
dependencies, priority (high | medium | low), expected_output, and validation_criteria.

Valid assigned agents:
- retriever (search uploaded documents in Qdrant with provenance)
- researcher (gather external web / multi-source findings)
- validator (verify claims, citations, and numerical consistency)
- reasoner (perform multi-step logical & quantitative Python analysis)
- critic (audit quality, logic, and hallucination risk)
- report_generator (compile 12-section report, visualizations, and downloadable artifacts)

Return ONLY valid JSON matching the AgentPlan schema.
"""

_DEFAULT_CRITERIA: dict[str, tuple[str, str]] = {
    "retriever": ("Grounded document chunks with source, page, and similarity score", "At least 1 relevant passage retrieved or absence explicitly logged"),
    "researcher": ("Synthesized multi-source research notes with source attribution", "Zero fabricated URLs or unverified claims"),
    "validator": ("Claim-evidence verification matrix and validation score", "Contradictions and unsupported numbers flagged"),
    "reasoner": ("Step-by-step analytical synthesis and verified Python calculations", "Every conclusion linked to supporting evidence IDs"),
    "critic": ("Adversarial quality & hallucination audit with pass/revise verdict", "Quality score >= 0.65 and zero unresolved hallucination flags"),
    "report_generator": ("12-section structured report, charts, and downloadable deliverables", "All sections populated with grounded citations"),
}


class Planner:
    """Produces a dependency-aware execution plan and SubtaskItem list."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_default_llm()

    def plan(self, task: str, *, task_type: str = "") -> AgentPlan:
        """Break *task* into structured steps and subtasks with zero-crash fallback."""
        log.info("Planner.plan", task_preview=(task or "")[:80], task_type=task_type)
        cleaned_task = (task or "").strip()
        greetings = {"hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening"}

        result: AgentPlan | None = None
        if task_type != "conversational" and cleaned_task.lower().strip("!.? ") not in greetings and cleaned_task:
            type_hint = f" (Task Type: {task_type})" if task_type else ""
            prompt = (
                f"Objective{type_hint}: {cleaned_task[:900]}\n\n"
                "Create a concise 4-6 step multi-agent execution plan with dependencies, "
                "expected outputs, and validation criteria."
            )
            try:
                result = cast(
                    AgentPlan,
                    self._llm.generate_model_safe(
                        prompt,
                        AgentPlan,
                        system_instruction=_PLANNER_SYSTEM_PROMPT,
                    ),
                )
            except Exception as exc:
                log.warning("Planner LLM fallback to deterministic dependency plan", error=str(exc)[:180])
                result = None

        raw_steps: list[Any] = list(result.steps) if result and result.steps else []
        if not raw_steps:
            raw_steps = [
                {"step": 1, "agent": "retriever", "description": "Retrieve relevant chunks and tables from Qdrant vector store", "dependencies": [], "priority": "high"},
                {"step": 2, "agent": "researcher", "description": "Synthesize supplementary multi-source context", "dependencies": [], "priority": "medium"},
                {"step": 3, "agent": "validator", "description": "Verify factual grounding, citations, and numerical consistency", "dependencies": [1, 2], "priority": "high"},
                {"step": 4, "agent": "reasoner", "description": "Execute multi-step reasoning and statistical computations", "dependencies": [3], "priority": "high"},
                {"step": 5, "agent": "critic", "description": "Audit reasoning for hallucinations, gaps, and logical rigor", "dependencies": [4], "priority": "high"},
                {"step": 6, "agent": "report_generator", "description": "Compile 12-section report, visualizations, and deliverables", "dependencies": [5], "priority": "medium"},
            ]

        normalized_steps: list[PlanStep] = []
        subtasks: list[SubtaskItem] = []

        for idx, s in enumerate(raw_steps, start=1):
            if isinstance(s, dict):
                step_num = int(s.get("step") or s.get("step_number") or idx)
                agent = str(s.get("agent") or s.get("assigned_agent") or "reasoner").lower()
                desc = str(s.get("description") or f"Execute {agent} analysis")
                deps = [int(d) for d in (s.get("dependencies") or s.get("depends_on") or []) if str(d).isdigit()]
                prio = str(s.get("priority") or ("high" if idx <= 4 else "medium"))
                exp_out = str(s.get("expected_output") or "")
                val_crit = str(s.get("validation_criteria") or "")
            else:
                step_num = int(getattr(s, "step", None) or getattr(s, "step_number", None) or idx)
                agent = str(getattr(s, "agent", "reasoner")).lower()
                desc = str(getattr(s, "description", "") or f"Execute {agent} analysis")
                raw_deps = getattr(s, "dependencies", None) or getattr(s, "depends_on", None) or []
                deps = [int(d) for d in raw_deps if str(d).isdigit()]
                prio = str(getattr(s, "priority", "") or ("high" if idx <= 4 else "medium"))
                exp_out = str(getattr(s, "expected_output", "") or "")
                val_crit = str(getattr(s, "validation_criteria", "") or "")

            def_exp, def_crit = _DEFAULT_CRITERIA.get(
                agent,
                ("Structured agent output", "Output complete and grounded"),
            )
            ps = PlanStep(
                step=step_num,
                task_id=f"T{step_num}",
                agent=agent,
                description=desc,
                dependencies=deps,
                priority=prio,
                expected_output=exp_out or def_exp,
                status="pending",
                validation_criteria=val_crit or def_crit,
            )
            normalized_steps.append(ps)
            subtasks.append(ps.to_subtask())

        return AgentPlan(
            objective=(result.objective if result and result.objective else task) or "Execution Plan",
            steps=normalized_steps,
            subtasks=subtasks,
            estimated_complexity=(result.estimated_complexity if result and result.estimated_complexity else "medium"),
        )
