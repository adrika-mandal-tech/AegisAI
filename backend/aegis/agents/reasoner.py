# backend/aegis/agents/reasoner.py
"""Reasoner & Analyst agent — multi-step logical synthesis + real Python computation (Section 7.6)."""

from __future__ import annotations

import re
import statistics
from typing import Any, cast

from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider
from aegis.llm.factory import get_default_llm
from aegis.schemas.agent import AgentOutput, Evidence

log = get_logger(__name__)

_REASONER_SYSTEM_PROMPT: str = """\
You are the Reasoner & Analyst agent in AegisAI.
Given validated evidence and deterministic Python-computed numerical summaries:
1. Perform rigorous step-by-step causal and comparative reasoning.
2. Incorporate exact Python-computed figures (never guess or round inaccurately).
3. Identify trends, anomalies, trade-offs, and strategic implications.
4. Cite supporting evidence IDs (e.g. [abc12345]) for factual assertions.
5. If a CRITIC REVISION MANDATE is present, resolve every flagged weakness directly.

Return ONLY valid JSON matching the AgentOutput schema (`success`, `summary`, `details`).
"""


def compute_python_numerical_digest(text: str) -> dict[str, Any]:
    """Deterministically compute summary metrics over numeric series extracted from text."""
    if not text:
        return {}
    raw_matches = re.findall(r"(?<![a-zA-Z0-9_])-?\d+(?:,\d{3})*(?:\.\d+)?(?![a-zA-Z0-9_])", text)
    values: list[float] = []
    for m in raw_matches[:400]:
        try:
            v = float(m.replace(",", ""))
            # Exclude standalone 4-digit calendar years when summarizing magnitudes
            if v.is_integer() and 1990 <= int(v) <= 2035:
                continue
            values.append(v)
        except Exception:
            continue

    if len(values) < 3:
        return {}

    mean_v = round(statistics.fmean(values), 4)
    median_v = round(statistics.median(values), 4)
    stdev_v = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
    min_v = round(min(values), 4)
    max_v = round(max(values), 4)
    pct_change = (
        round(((values[-1] - values[0]) / abs(values[0])) * 100.0, 2)
        if abs(values[0]) > 1e-9
        else None
    )
    return {
        "sample_count": len(values),
        "mean": mean_v,
        "median": median_v,
        "std_dev": stdev_v,
        "min": min_v,
        "max": max_v,
        "sum": round(sum(values), 4),
        "first_to_last_pct_change": pct_change,
    }


class Reasoner:
    """Combines deterministic Python numerical analysis with multi-step LLM synthesis."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_default_llm()

    def reason(
        self,
        information: str,
        evidence_items: list[Evidence] | None = None,
        *,
        analysis_results: dict[str, Any] | None = None,
    ) -> AgentOutput:
        """Execute quantitative digest + multi-step reasoning over *information*."""
        log.info("Reasoner.reason", length=len(information))

        py_digest = compute_python_numerical_digest(information)
        py_block = (
            f"\nVerified Python Numerical Digest (exact computation): {py_digest}\n"
            if py_digest
            else ""
        )
        ds_block = ""
        if analysis_results:
            ds_summaries: list[str] = []
            for ds_name, ds_info in list(analysis_results.items())[:4]:
                if isinstance(ds_info, dict) and ds_info.get("summary"):
                    ds_summaries.append(f"[{ds_name}] {str(ds_info['summary'])[:600]}")
                else:
                    ds_summaries.append(f"[{ds_name}] {str(ds_info)[:400]}")
            if ds_summaries:
                ds_block = "\nDataset Statistical Profiles:\n" + "\n".join(ds_summaries) + "\n"

        evidence_summary = ""
        if evidence_items:
            evidence_summary = "\nAvailable Evidence (cite by [ID]):\n"
            for e in evidence_items[:10]:
                if isinstance(e, Evidence):
                    evidence_summary += (
                        f"  [{e.id}] {e.claim[:110]} | Source: {e.source} | Conf: {e.confidence:.2f}\n"
                    )
                elif isinstance(e, dict):
                    evidence_summary += (
                        f"  [{e.get('id', '')}] {str(e.get('claim', ''))[:110]} | "
                        f"Source: {e.get('source', 'Doc')}\n"
                    )

        prompt = (
            f"Analytical Context:\n{information[:4200]}\n"
            f"{py_block}"
            f"{ds_block}"
            f"{evidence_summary}\n"
            "Synthesize concise step-by-step reasoning, quantitative insights, and conclusions."
        )
        try:
            result: AgentOutput = cast(
                AgentOutput,
                self._llm.generate_model_safe(
                    prompt,
                    AgentOutput,
                    system_instruction=_REASONER_SYSTEM_PROMPT,
                ),
            )
        except Exception as exc:
            log.warning("Reasoner LLM call fell back to deterministic synthesis", error=str(exc)[:200])
            fallback_summary = (
                f"Synthesized {len(evidence_items or [])} grounded evidence items"
                + (f" and {py_digest.get('sample_count', 0)} numerical observations." if py_digest else ".")
            )
            fallback_details = (information[:1800].strip() or fallback_summary)
            if py_digest:
                fallback_details += f"\n\nVerified Python Numerical Digest: {py_digest}"
            if ds_block:
                fallback_details += f"\n{ds_block}"
            result = AgentOutput(
                success=True,
                summary=fallback_summary,
                details=fallback_details,
            )

        meta = dict(result.metadata or {})
        if py_digest:
            meta["computations"] = py_digest
        if analysis_results:
            meta["dataset_count"] = len(analysis_results)
        result.metadata = meta
        return result
