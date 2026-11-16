# backend/aegis/evaluation/ragas.py
"""Ragas integration for AegisAI.

Ragas is a framework for evaluating RAG pipelines. This module
provides evaluation functions that assess retrieval quality,
answer faithfulness, and context relevancy.
"""

from __future__ import annotations

import re
from typing import Any

from aegis.core.logging import get_logger

log = get_logger(__name__)


def evaluate_rag_pipeline(
    task: str,
    retrieved_context: str,
    report_markdown: str,
) -> dict[str, Any]:
    """Evaluate a RAG pipeline run using Ragas-style metrics.

    This implementation uses enhanced heuristic metrics that provide
    meaningful insights without requiring the ragas package.
    When ragas is installed and configured, these can be replaced
    with proper context_precision, faithfulness, and answer_relevancy metrics.
    """
    log.info("evaluate_rag_pipeline: ragas", task_preview=task[:80])

    try:
        metrics: dict[str, Any] = {}
        overall_score: float | None = None

        # Context quality metrics
        if retrieved_context:
            metrics.update(_evaluate_context_quality(retrieved_context))
        
        # Report quality metrics
        if report_markdown:
            metrics.update(_evaluate_report_ragas_quality(report_markdown, retrieved_context))

        # Task alignment metrics
        if task and retrieved_context:
            metrics.update(_evaluate_task_alignment(task, retrieved_context, report_markdown))

        # Calculate overall score from key metrics
        overall_score = _calculate_ragas_overall_score(metrics)

        return {
            "framework": "ragas",
            "status": "ready",
            "metrics": metrics,
            "overall_score": overall_score,
            "notes": "Enhanced heuristic RAG evaluation. Install ragas to enable context_precision, faithfulness, and answer_relevancy metrics.",
        }
    except Exception as exc:
        log.exception("evaluate_rag_pipeline failed")
        return {
            "framework": "ragas",
            "status": "error",
            "metrics": {},
            "overall_score": None,
            "notes": f"Evaluation failed: {exc}",
        }


def _evaluate_context_quality(context: str) -> dict[str, Any]:
    """Evaluate the quality of retrieved context."""
    metrics = {}
    
    if not context:
        return {
            "context_length_chars": 0,
            "context_relevance_score": 0.0,
            "context_information_density": 0.0,
            "context_coherence_score": 0.0,
        }
    
    # Basic context metrics
    metrics["context_length_chars"] = len(context)
    metrics["context_has_numbers"] = _has_numbers(context)
    metrics["context_sections"] = _count_sections(context)
    
    # Quality metrics
    metrics["context_relevance_score"] = _calculate_context_relevance(context)
    metrics["context_information_density"] = _calculate_information_density(context)
    metrics["context_coherence_score"] = _calculate_context_coherence(context)
    metrics["context_diversity_score"] = _calculate_context_diversity(context)
    
    return metrics


def _evaluate_report_ragas_quality(report: str, context: str) -> dict[str, Any]:
    """Evaluate report quality in relation to context (RAG-specific)."""
    metrics = {}
    
    if not report:
        return {
            "report_length_chars": 0,
            "report_faithfulness_score": 0.0,
            "report_context_utilization": 0.0,
            "report_answer_relevancy": 0.0,
        }
    
    # Report metrics
    metrics["report_length_chars"] = len(report)
    metrics["report_has_structure"] = _count_sections(report) >= 3
    
    # RAG-specific quality metrics
    metrics["report_faithfulness_score"] = _calculate_faithfulness(report, context)
    metrics["report_context_utilization"] = _calculate_context_utilization(report, context)
    metrics["report_answer_relevancy"] = _calculate_answer_relevancy(report, context)
    metrics["report_hallucination_score"] = 1.0 - metrics["report_faithfulness_score"]
    
    return metrics


def _evaluate_task_alignment(task: str, context: str, report: str) -> dict[str, Any]:
    """Evaluate how well the pipeline addressed the original task."""
    metrics = {}
    
    if not task:
        return metrics
    
    # Task alignment metrics
    metrics["context_task_alignment"] = _estimate_alignment(task, context)
    metrics["report_task_alignment"] = _estimate_alignment(task, report)
    metrics["pipeline_task_coverage"] = _calculate_task_coverage(task, context, report)
    
    return metrics


def _calculate_ragas_overall_score(metrics: dict[str, Any]) -> float:
    """Calculate an overall Ragas-style score from the metrics."""
    # Define key metrics and their weights for the overall score
    key_metrics = {
        "context_relevance_score": 0.20,
        "context_information_density": 0.15,
        "context_coherence_score": 0.10,
        "report_faithfulness_score": 0.25,
        "report_context_utilization": 0.15,
        "report_answer_relevancy": 0.15,
    }
    
    score = 0.0
    total_weight = 0.0
    
    for metric_name, weight in key_metrics.items():
        if metric_name in metrics:
            value = metrics[metric_name]
            # Normalize to 0-1 scale if needed
            normalized_value = max(0.0, min(1.0, float(value)))
            score += normalized_value * weight
            total_weight += weight
    
    return round(score / total_weight, 3) if total_weight > 0 else 0.0


def _has_numbers(text: str) -> bool:
    """Check if text contains numbers."""
    return bool(re.search(r"\b\d+(?:\.\d+)?\b", text))


def _count_sections(text: str) -> int:
    """Count Markdown headings as a proxy for structure."""
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            count += 1
    return count


def _calculate_context_relevance(context: str) -> float:
    """Calculate how relevant the context appears to be."""
    if not context:
        return 0.0
    
    # Simple heuristics for context relevance
    # Look for indicators of quality content
    quality_indicators = [
        len(context) > 100,  # Substantial content
        context.count('.') > 5,  # Multiple sentences
        ',' in context and ';' in context,  # Complex punctuation
        any(c.isupper() for c in context[:50]),  # Proper capitalization
        not context.isupper(),  # Not all caps (likely not garbage)
    ]
    
    return sum(quality_indicators) / len(quality_indicators)


def _calculate_information_density(context: str) -> float:
    """Calculate the information density of the context."""
    if not context:
        return 0.0
    
    # Remove whitespace and measure meaningful content vs total
    meaningful_chars = len(re.sub(r'\s+', '', context))
    total_chars = len(context)
    
    if total_chars == 0:
        return 0.0
    
    # Information density ratio (lower whitespace = higher density)
    density = meaningful_chars / total_chars
    # Normalize: 0.3-0.8 is good range for text density
    return max(0.0, min(1.0, (density - 0.2) / 0.6))


def _calculate_context_coherence(context: str) -> float:
    """Calculate the coherence of the context."""
    if not context:
        return 0.0
    
    sentences = re.split(r'[.!?]+', context)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if len(sentences) < 2:
        return 0.5 if sentences else 0.0
    
    # Check for logical flow between sentences
    coherence_indicators = 0
    total_checks = 0
    
    for i in range(len(sentences) - 1):
        total_checks += 1
        curr_sent = sentences[i].lower()
        next_sent = sentences[i + 1].lower()
        
        # Check for topic continuity (shared nouns, etc.)
        curr_words = set(re.findall(r'\b\w{4,}\b', curr_sent))
        next_words = set(re.findall(r'\b\w{4,}\b', next_sent))
        
        if curr_words and next_words:
            overlap = len(curr_words & next_words)
            total_unique = len(curr_words | next_words)
            if total_unique > 0:
                coherence_indicators += overlap / total_unique
    
    return coherence_indicators / total_checks if total_checks > 0 else 0.5


def _calculate_context_diversity(context: str) -> float:
    """Calculate the diversity of information in the context."""
    if not context:
        return 0.0
    
    # Simple diversity metric based on vocabulary richness
    words = re.findall(r'\b\w+\b', context.lower())
    if not words:
        return 0.0
    
    unique_words = set(words)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Type-token ratio (diversity measure)
    diversity = len(unique_words) / total_words
    # Scale to 0-1 where higher is more diverse
    return min(diversity * 2, 1.0)


def _calculate_faithfulness(report: str, context: str) -> float:
    """Calculate how faithful the report is to the provided context."""
    if not report or not context:
        return 0.0
    
    # Simple faithfulness heuristic: check if report claims are supported by context
    report_sentences = re.split(r'[.!?]+', report)
    report_sentences = [s.strip() for s in report_sentences if len(s.strip()) > 10]
    
    if not report_sentences:
        return 0.5
    
    supported_claims = 0
    total_claims = 0
    
    context_lower = context.lower()
    
    for sentence in report_sentences:
        # Skip very short sentences or questions
        if len(sentence) < 15 or sentence.endswith('?'):
            continue
            
        total_claims += 1
        sentence_lower = sentence.lower()
        
        # Extract key phrases/nouns from the sentence
        key_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
        key_phrases += re.findall(r'\b\d+(?:\.\d+)?\s*(?:percent|%|million|billion|thousand)\b', sentence_lower)
        
        if not key_phrases:
            # If no specific phrases found, check general relevance
            if any(word in context_lower for word in sentence_lower.split() if len(word) > 4):
                supported_claims += 1
        else:
            # Check if key phrases appear in context
            phrase_found = any(phrase.lower() in context_lower for phrase in key_phrases)
            if phrase_found:
                supported_claims += 1
    
    return supported_claims / total_claims if total_claims > 0 else 0.5


def _calculate_context_utilization(report: str, context: str) -> float:
    """Calculate how well the report utilizes the provided context."""
    if not report or not context:
        return 0.0
    
    # Extract meaningful chunks from context and see how many are referenced in report
    context_chunks = _extract_meaningful_chunks(context)
    if not context_chunks:
        return 0.5
    
    utilized_chunks = 0
    report_lower = report.lower()
    
    for chunk in context_chunks:
        if len(chunk.strip()) > 10 and chunk.lower() in report_lower:
            utilized_chunks += 1
    
    return utilized_chunks / len(context_chunks)


def _calculate_answer_relevancy(report: str, context: str) -> float:
    """Calculate how relevant the report is as an answer to the implied question in context."""
    if not report or not context:
        return 0.0
    
    # Simple heuristic: longer, well-structured reports that use context are more relevant
    length_score = min(len(report) / 500.0, 1.0)  # Optimal around 500 chars
    structure_score = min(_count_sections(report) / 5.0, 1.0)  # Optimal around 5 sections
    utilization_score = _calculate_context_utilization(report, context)
    
    return (length_score * 0.3 + structure_score * 0.3 + utilization_score * 0.4)


def _extract_meaningful_chunks(text: str) -> list[str]:
    """Extract meaningful chunks from text for utilization analysis."""
    if not text:
        return []
    
    # Split by paragraphs and filter for meaningful content
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    meaningful = []
    
    for para in paragraphs:
        # Skip very short paragraphs or those that are just formatting
        if len(para) < 20:
            continue
        # Skip paragraphs that are mostly whitespace or special chars
        if len(re.sub(r'[^\w\s]', '', para)) < len(para) * 0.3:
            continue
        meaningful.append(para)
    
    return meaningful


def _estimate_alignment(task: str, context: str) -> float:
    """Estimate alignment between task and context."""
    if not task or not context:
        return 0.0
    
    task_words = set(re.findall(r'\b\w{3,}\b', task.lower()))
    context_words = set(re.findall(r'\b\w{3,}\b', context.lower()))
    
    if not task_words:
        return 0.0
    
    overlap = task_words & context_words
    return len(overlap) / len(task_words)


def _calculate_task_coverage(task: str, context: str, report: str) -> float:
    """Calculate how well the pipeline covered the task across context and report."""
    if not task:
        return 0.0
    
    task_words = set(re.findall(r'\b\w{3,}\b', task.lower()))
    if not task_words:
        return 0.0
    
    context_words = set(re.findall(r'\b\w{3,}\b', context.lower())) if context else set()
    report_words = set(re.findall(r'\b\w{3,}\b', report.lower())) if report else set()
    
    # Words covered by either context or report
    covered_words = (context_words | report_words) & task_words
    
    return len(covered_words) / len(task_words) if task_words else 0.0