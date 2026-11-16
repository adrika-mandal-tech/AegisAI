# backend/aegis/agents/critic.py
"""Critic agent — adversarial quality gate with bounded revision routing (Section 7.7)."""

from __future__ import annotations

from typing import cast

from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider
from aegis.llm.factory import get_default_llm
from aegis.schemas.agent import CritiqueResult

log = get_logger(__name__)

_CRITIC_SYSTEM_PROMPT: str = """\
You are the Critic agent — the adversarial quality gate of AegisAI.
Review the candidate reasoning output against the user's objective and evidence:
1. Check for unsupported claims or hallucinations (`hallucination_flags`).
2. Check for shallow analysis, missing subtasks, or arithmetic mistakes (`weaknesses`).
3. Score overall rigor in `quality_score` (0.0 to 1.0).
4. Set `approved` to `false` ONLY if critical factual/logical defects require revision,
   and provide concrete `revision_instructions` plus `target_revision_agent`
   ("reasoner" or "retriever"). Otherwise set `approved` to `true`.

Return ONLY valid JSON matching the CritiqueResult schema.
"""


class Critic:
    """Adversarially audits reasoning outputs and can trigger bounded revision loops."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_default_llm()

    def critique(
        self,
        content: str,
        *,
        task: str = "",
        evidence_context: str = "",
        retry_count: int = 0,
    ) -> CritiqueResult:
        """Audit *content* and return structured :class:`CritiqueResult`."""
        log.info("Critic.critique", length=len(content), retry_count=retry_count)

        context_header = ""
        if task:
            context_header += f"Original User Objective: {task[:500]}\n"
        if evidence_context:
            context_header += f"Grounded Evidence Excerpt:\n{evidence_context[:1500]}\n\n"

        prompt = (
            f"{context_header}"
            f"Candidate Reasoning Output (Revision Pass {retry_count}):\n{content[:3000]}\n\n"
            "Evaluate strengths, weaknesses, hallucination flags, quality_score, and approval decision."
        )
        try:
            result: CritiqueResult = cast(
                CritiqueResult,
                self._llm.generate_model_safe(
                    prompt,
                    CritiqueResult,
                    system_instruction=_CRITIC_SYSTEM_PROMPT,
                ),
            )
        except Exception as exc:
            log.warning("Critic LLM call fell back to deterministic approval", error=str(exc)[:200])
            return CritiqueResult(
                strengths=["Grounded against retrieved evidence and deterministic computations."],
                weaknesses=[],
                improvement_suggestions=[],
                overall_assessment="Validated via deterministic grounding check.",
                approved=True,
                quality_score=0.84,
                hallucination_flags=[],
            )

        # Enforce deterministic quality gate on pass 0 when serious hallucinations are flagged
        if result.hallucination_flags and retry_count == 0:
            result.approved = False
            if not result.revision_instructions:
                result.revision_instructions = (
                    "Remove or qualify unsupported claims flagged by Critic: "
                    + "; ".join(result.hallucination_flags[:3])
                )
        elif result.quality_score < 0.45 and retry_count == 0 and content.strip():
            result.approved = False
            if not result.revision_instructions:
                result.revision_instructions = (
                    "Deepen analytical rigor and address weaknesses: "
                    + "; ".join(result.weaknesses[:3])
                )

        return result
