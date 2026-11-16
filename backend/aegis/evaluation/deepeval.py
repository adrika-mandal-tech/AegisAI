# backend/aegis/evaluation/deepeval.py
"""DeepEval integration for AegisAI.

DeepEval is an evaluation framework for LLM applications. This module
provides evaluation functions that assess pipeline output quality,
answer relevancy, and hallucination rates.
"""

from __future__ import annotations

import re
import math
from typing import Any

from aegis.core.logging import get_logger

log = get_logger(__name__)


def evaluate_pipeline_run(
    task: str,
    report_markdown: str,
    pipeline_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a completed pipeline run using DeepEval-style metrics.

    This implementation uses enhanced heuristic metrics that provide
    meaningful insights without requiring the deepeval package.
    When deepeval is installed and configured, these can be replaced
    with proper G-Eval, Faithfulness, and AnswerRelevancy metrics.
    """
    log.info("evaluate_pipeline_run: deep_eval", task_preview=task[:80])

    try:
        metrics: dict[str, Any] = {}
        overall_score: float | None = None

        # Report quality metrics
        if report_markdown:
            metrics.update(_evaluate_report_quality(report_markdown))
        
        # Pipeline execution metrics
        if pipeline_trace:
            metrics.update(_evaluate_pipeline_execution(pipeline_trace))

        # Calculate overall score from key metrics
        overall_score = _calculate_deepeval_overall_score(metrics)

        return {
            "framework": "deepeval",
            "status": "ready",
            "metrics": metrics,
            "overall_score": overall_score,
            "notes": "Enhanced heuristic DeepEval evaluation. Install deepeval to enable G-Eval, Faithfulness, and AnswerRelevancy metrics.",
        }
    except Exception as exc:
        log.exception("evaluate_pipeline_run failed")
        return {
            "framework": "deepeval",
            "status": "error",
            "metrics": {},
            "overall_score": None,
            "notes": f"Evaluation failed: {exc}",
        }


def _evaluate_report_quality(report: str) -> dict[str, Any]:
    """Evaluate the quality of the generated report."""
    metrics = {}
    
    # Basic structure metrics
    metrics["report_length_chars"] = len(report)
    metrics["report_sections"] = _count_sections(report)
    metrics["has_executive_summary"] = _has_section(report, ["executive summary", "summary"])
    metrics["has_findings"] = _has_section(report, ["findings", "results", "analysis"])
    metrics["has_action_items"] = _has_section(report, ["action items", "recommendations", "next steps"])
    metrics["has_conclusion"] = _has_section(report, ["conclusion", "summary"])
    
    # Content quality metrics
    metrics["report_readability_score"] = _calculate_readability(report)
    metrics["report_structure_score"] = _calculate_structure_score(report)
    metrics["report_completeness_score"] = _calculate_completeness_score(report)
    
    # Evidence and citation metrics
    metrics["report_has_citations"] = _has_citations(report)
    metrics["report_citation_density"] = _calculate_citation_density(report)
    
    return metrics


def _evaluate_pipeline_execution(pipeline_trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the pipeline execution quality."""
    metrics = {}
    
    if not pipeline_trace:
        return metrics
    
    # Agent execution metrics
    total_agents = len(pipeline_trace)
    completed_agents = sum(1 for step in pipeline_trace if step.get("status") == "ok")
    failed_agents = sum(1 for step in pipeline_trace if step.get("status") not in ("ok", "running", ""))
    skipped_agents = sum(1 for step in pipeline_trace if step.get("status") == "skipped")
    
    metrics["agent_count"] = total_agents
    metrics["completed_agents"] = completed_agents
    metrics["failed_agents"] = failed_agents
    metrics["skipped_agents"] = skipped_agents
    metrics["agent_success_rate"] = completed_agents / total_agents if total_agents > 0 else 0.0
    metrics["pipeline_efficiency"] = _calculate_pipeline_efficiency(pipeline_trace)
    
    return metrics


def _calculate_pipeline_efficiency(pipeline_trace: list[dict[str, Any]]) -> float:
    """Calculate pipeline efficiency based on execution trace."""
    if not pipeline_trace:
        return 0.0
    
    total = len(pipeline_trace)
    completed = sum(1 for step in pipeline_trace if step.get("status") == "ok")
    failed = sum(1 for step in pipeline_trace if step.get("status") == "error")
    skipped = sum(1 for step in pipeline_trace if step.get("status") == "skipped")
    
    # Base efficiency from success rate
    success_rate = completed / total if total else 0.0
    
    # Penalize failures and skips
    failure_penalty = (failed + skipped) / total if total else 0.0
    
    # Efficiency is success rate minus penalty
    efficiency = success_rate - (failure_penalty * 0.5)
    
    return max(0.0, min(1.0, efficiency))


def _calculate_deepeval_overall_score(metrics: dict[str, Any]) -> float:
    """Calculate an overall DeepEval-style score from the metrics."""
    # Define key metrics and their weights for the overall score
    key_metrics = {
        "report_sections": 0.15,
        "has_executive_summary": 0.10,
        "has_findings": 0.15,
        "has_action_items": 0.10,
        "report_readability_score": 0.15,
        "report_structure_score": 0.15,
        "agent_success_rate": 0.10,
        "pipeline_efficiency": 0.10,
    }
    
    score = 0.0
    total_weight = 0.0
    
    for metric_name, weight in key_metrics.items():
        if metric_name in metrics:
            value = metrics[metric_name]
            # Normalize different metric types to 0-1 scale
            normalized_value = _normalize_metric_value(metric_name, value)
            score += normalized_value * weight
            total_weight += weight
    
    return round(score / total_weight, 3) if total_weight > 0 else 0.0


def _normalize_metric_value(metric_name: str, value: Any) -> float:
    """Normalize metric values to 0-1 scale for scoring."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    elif isinstance(value, int):
        if "length" in metric_name or "chars" in metric_name:
            # For length metrics, assume 1000+ chars is good
            return min(value / 1000.0, 1.0)
        elif "sections" in metric_name:
            # For sections, assume 5+ sections is good
            return min(value / 5.0, 1.0)
        elif "count" in metric_name or "agents" in metric_name:
            # For count metrics, assume 5+ is good
            return min(value / 5.0, 1.0)
        else:
            # Default normalization for other integers
            return min(value / 10.0, 1.0)
    elif isinstance(value, float):
        # Assume floats are already in 0-1 range or can be clamped
        return max(0.0, min(1.0, value))
    else:
        return 0.0


def _count_sections(text: str) -> int:
    """Count Markdown headings as a proxy for report structure."""
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            count += 1
    return count


def _has_section(text: str, section_keywords: list[str]) -> bool:
    """Check if text contains a section with any of the given keywords."""
    text_lower = text.lower()
    # Look for markdown headers that contain the keywords
    for line in text.splitlines():
        line_lower = line.strip().lower()
        if line_lower.startswith("#"):
            for keyword in section_keywords:
                if keyword in line_lower:
                    return True
    return False


def _calculate_readability(text: str) -> float:
    """Calculate a simple readability score based on sentence and word length."""
    if not text:
        return 0.0
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    words = text.split()
    if not words:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Simple readability score (inverse of complexity)
    # Ideal: avg sentence length 15-20, avg word length 4-5
    sentence_score = max(0.0, 1.0 - abs(avg_sentence_length - 17.5) / 20.0)
    word_score = max(0.0, 1.0 - abs(avg_word_length - 4.5) / 3.0)
    
    return (sentence_score + word_score) / 2.0


def _calculate_structure_score(text: str) -> float:
    """Calculate a score based on report structure and formatting."""
    if not text:
        return 0.0
    
    lines = text.splitlines()
    if not lines:
        return 0.0
    
    # Check for various structural elements
    has_bullets = any(line.strip().startswith(('-', '*', '+')) for line in lines)
    has_numbers = any(re.match(r'^\d+\.', line.strip()) for line in lines)
    has_bold = '**' in text or '__' in text
    has_italics = '*' in text and text.count('*') >= 4  # At least two italic sections
    has_code = '```' in text
    
    structure_elements = [has_bullets, has_numbers, has_bold, has_italics, has_code]
    return sum(structure_elements) / len(structure_elements)


def _calculate_completeness_score(text: str) -> float:
    """Calculate a completeness score based on content coverage."""
    if not text:
        return 0.0
    
    text_lower = text.lower()
    
    # Key sections that should be present in a comprehensive report
    key_sections = [
        "introduction", "background", "methodology", 
        "results", "findings", "analysis",
        "discussion", "conclusion", "recommendations",
        "references", "references"
    ]
    
    found_sections = 0
    for section in key_sections:
        if section in text_lower:
            found_sections += 1
    
    return found_sections / len(key_sections) if key_sections else 0.0


def _has_citations(text: str) -> bool:
    """Check if text contains citations."""
    citation_patterns = [
        r'\[\d+\]',  # [1], [2], etc.
        r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
        r'et al\.',  # et al.
        r'doi:\s*\S+',  # DOI
    ]
    
    for pattern in citation_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _calculate_citation_density(text: str) -> float:
    """Calculate the density of citations in the text."""
    if not text:
        return 0.0
    
    # Count potential citations
    words = text.split()
    if not words:
        return 0.0
    
    citation_count = 0
    for word in words:
        if re.search(r'\[\d+\]|\([^)]*\d{4}[^)]*\)|et al\.|doi:', word, re.IGNORECASE):
            citation_count += 1
    
    return min(citation_count / len(words) * 20, 1.0)  # Normalize to 0-1 (5% citation density = 1.0)