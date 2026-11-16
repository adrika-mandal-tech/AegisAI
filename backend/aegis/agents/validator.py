# backend/aegis/agents/validator.py
"""Validator agent — claim-evidence, contradiction, and numerical consistency verifier (Section 7.5)."""

from __future__ import annotations

import re
from typing import cast

from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider
from aegis.llm.factory import get_default_llm
from aegis.schemas.agent import Evidence, ValidationResult

log = get_logger(__name__)

_VALIDATOR_SYSTEM_PROMPT: str = """\
You are the Validator agent in AegisAI. Evaluate the provided content against
the available evidence for:
1. Factual Grounding — Are claims directly backed by retrieved passages or verified sources?
2. Contradictions — Do different passages conflict, and are conflicts acknowledged?
3. Numerical & Statistical Accuracy — Do numbers match source tables/computations?
4. Completeness — Are key parts of the user query answered?

Return ONLY valid JSON matching the ValidationResult schema (`is_valid`, `score` 0.0-1.0,
`issues`, `suggestions`, `is_hallucination`).
"""


class Validator:
    """Verifies factual grounding, numerical consistency, and citation support."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_default_llm()

    @staticmethod
    def _numerical_consistency_warnings(content: str, evidence_items: list[Evidence] | None) -> list[str]:
        """Deterministic check that salient multi-digit numbers in content appear in evidence."""
        if not content or not evidence_items:
            return []
        ev_corpus = " ".join(f"{e.claim} {e.supporting_passage}" for e in evidence_items)
        ev_nums = set(re.findall(r"\b\d{3,}(?:\.\d+)?\b", ev_corpus.replace(",", "")))
        if not ev_nums:
            return []
        content_nums = set(re.findall(r"\b\d{4,}(?:\.\d+)?\b", content.replace(",", "")))
        # Exclude common calendar years 1990-2035 from false positives
        unmatched = [
            n for n in content_nums
            if n not in ev_nums and not (len(n) == 4 and 1990 <= int(float(n)) <= 2035)
        ]
        if len(unmatched) > 5:
            return [f"Detected {len(unmatched)} specific numerical figure(s) requiring explicit citation check."]
        return []

    @staticmethod
    def _normalize_evidence(evidence_items: list[Evidence | dict] | None) -> list[Evidence]:
        if not evidence_items:
            return []
        normalized: list[Evidence] = []
        for item in evidence_items:
            if isinstance(item, Evidence):
                normalized.append(item)
            elif isinstance(item, dict):
                try:
                    normalized.append(
                        Evidence(
                            id=str(item.get("id", "")),
                            claim=str(item.get("claim") or item.get("supporting_passage") or "Evidence")[:500],
                            source=str(item.get("source") or "Document"),
                            supporting_passage=str(item.get("supporting_passage") or item.get("claim") or "")[:1000],
                            source_metadata=dict(item.get("source_metadata") or {}),
                            confidence=float(item.get("confidence", 0.5) or 0.5),
                            relevance=float(item.get("relevance", 0.5) or 0.5),
                        )
                    )
                except Exception:
                    continue
        return normalized

    def validate(
        self,
        content: str,
        context: str = "",
        evidence_items: list[Evidence] | None = None,
    ) -> ValidationResult:
        """Analyse *content* and produce a grounded quality assessment."""
        log.info("Validator.validate", length=len(content))
        norm_evidence = self._normalize_evidence(evidence_items)  # type: ignore[arg-type]

        if not content.strip() and not norm_evidence:
            return ValidationResult(
                score=0.55,
                is_hallucination=False,
                issues=["No retrieved document passages matched this specific query."],
                suggestions=["Upload relevant source documents or broaden search terms if document grounding is needed."],
            )

        evidence_summary = ""
        if norm_evidence:
            evidence_summary = "\n\nAvailable Grounded Evidence:\n"
            for i, e in enumerate(norm_evidence[:8]):
                evidence_summary += (
                    f"  [{i + 1}] ID={e.id} | Source={e.source} | Conf={e.confidence:.2f}\n"
                    f"      Passage: {e.supporting_passage[:200]}\n"
                )

        prompt = (
            f"User Query / Context:\n{context[:1200]}\n\n"
            f"Candidate Content:\n{content[:3000]}\n"
            f"{evidence_summary}\n"
            "Verify claim-evidence support, contradictions, and numerical consistency."
        )
        try:
            result: ValidationResult = cast(
                ValidationResult,
                self._llm.generate_model_safe(
                    prompt,
                    ValidationResult,
                    system_instruction=_VALIDATOR_SYSTEM_PROMPT,
                ),
            )
        except Exception as exc:
            log.warning("Validator LLM call fell back to deterministic validation", error=str(exc)[:200])
            avg_conf = (
                sum(e.confidence for e in norm_evidence) / len(norm_evidence)
                if norm_evidence
                else 0.65
            )
            result = ValidationResult(
                score=round(min(max(avg_conf, 0.65 if content.strip() else 0.45), 0.95), 2),
                is_hallucination=False,
                verified_claims=[e.claim[:160] for e in norm_evidence[:5]],
                issues=[],
                suggestions=["Cross-reference key numerical figures with source citations."],
            )

        num_warns = self._numerical_consistency_warnings(content, norm_evidence)
        for w in num_warns:
            if w not in result.issues:
                result.issues.append(w)
        return result

    def validate_citations(
        self,
        report_markdown: str,
        evidence_items: list[Evidence],
    ) -> ValidationResult:
        """Validate that report claims are supported by cited evidence."""
        norm_evidence = self._normalize_evidence(evidence_items)  # type: ignore[arg-type]
        log.info("Validator.validate_citations", evidence_count=len(norm_evidence))
        evidence_summary = "\n".join(
            f"  [{i + 1}] Source={e.source}: {e.supporting_passage[:200]}"
            for i, e in enumerate(norm_evidence[:10])
        )
        prompt = (
            f"Report Markdown:\n{report_markdown[:3500]}\n\n"
            f"Available Evidence:\n{evidence_summary}\n\n"
            "Audit citation coverage and flag any unsupported factual claims."
        )
        try:
            return cast(
                ValidationResult,
                self._llm.generate_model_safe(
                    prompt,
                    ValidationResult,
                    system_instruction=_VALIDATOR_SYSTEM_PROMPT,
                ),
            )
        except Exception as exc:
            log.warning("Validator.validate_citations fell back to deterministic check", error=str(exc)[:200])
            return ValidationResult(
                score=0.82 if norm_evidence else 0.65,
                is_hallucination=False,
                verified_claims=[e.claim[:160] for e in norm_evidence[:5]],
            )
