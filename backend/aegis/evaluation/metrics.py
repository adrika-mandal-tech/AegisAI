# backend/aegis/evaluation/metrics.py
"""Evaluation metrics for agent output quality.

This module implements actual evaluation metrics that can be used
for assessing RAG pipeline output quality, providing meaningful
alternatives to external libraries like Ragas and DeepEval.
"""

from __future__ import annotations

import re
from typing import Any


def faithfulness_score(response: str, context: str) -> float:
    """Return a faithfulness score (0.0–1.0) comparing *response* to *context*.

    Measures how much of the response is supported by the provided context.
    Higher scores indicate better faithfulness (less hallucination).
    """
    if not response or not context:
        return 0.0
    
    # Extract factual claims from response
    response_sentences = re.split(r'[.!?]+', response)
    response_sentences = [s.strip() for s in response_sentences if len(s.strip()) > 15]
    
    if not response_sentences:
        return 0.5  # Neutral score for minimal content
    
    supported_claims = 0
    total_claims = 0
    context_lower = context.lower()
    
    for sentence in response_sentences:
        # Skip questions and very short statements
        if sentence.endswith('?') or len(sentence) < 20:
            continue
            
        total_claims += 1
        sentence_lower = sentence.lower()
        
        # Extract key factual elements (proper nouns, numbers, dates)
        key_elements = []
        
        # Proper nouns and entities
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
        key_elements.extend(proper_nouns)
        
        # Numbers and measurements
        measurements = re.findall(r'\b\d+(?:\.\d+)?\s*(?:percent|%|million|billion|thousand|km|m|cm|mm|years?|months?|days?|hours?|minutes?|seconds?)\b', sentence_lower)
        key_elements.extend(measurements)
        
        # Years and dates
        years = re.findall(r'\b(?:19|20)\d{2}\b', sentence)
        key_elements.extend(years)
        
        if not key_elements:
            # Fallback: check if significant words from sentence appear in context
            significant_words = [w for w in sentence_lower.split() if len(w) > 4 and w not in {'the', 'and', 'for', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}]
            if significant_words:
                if any(word in context_lower for word in significant_words[:3]):  # Check first 3 significant words
                    supported_claims += 1
        else:
            # Check if key elements are supported by context
            element_found = any(element.lower() in context_lower for element in key_elements)
            if element_found:
                supported_claims += 1
    
    return supported_claims / total_claims if total_claims > 0 else 0.5


def answer_relevancy_score(question: str, answer: str) -> float:
    """Return an answer-relevancy score (0.0–1.0).

    Measures how relevant the answer is to the given question.
    Higher scores indicate better relevance.
    """
    if not question or not answer:
        return 0.0
    
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    # Extract key concepts from question
    question_words = set(re.findall(r'\b\w{3,}\b', question_lower))
    # Remove common stop words
    stop_words = {'the', 'and', 'for', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'this', 'that', 'with', 'from', 'they', 'what', 'when', 'where', 'why', 'how'}
    question_keywords = question_words - stop_words
    
    if not question_keywords:
        return 0.5  # Neutral if no meaningful keywords
    
    # Check how many question keywords appear in the answer
    matched_keywords = 0
    for keyword in question_keywords:
        if keyword in answer_lower:
            matched_keywords += 1
    
    # Base score on keyword matching
    keyword_score = matched_keywords / len(question_keywords)
    
    # Bonus for answer length and structure (but not too long)
    length_score = min(len(answer) / 200.0, 1.0)  # Optimal around 200 chars
    if len(answer) > 500:  # Penalize overly long answers
        length_score = max(0.0, 1.0 - (len(answer) - 500) / 1000.0)
    
    # Structure bonus (complete sentences, etc.)
    sentence_count = len(re.split(r'[.!?]+', answer))
    structure_score = min(sentence_count / 3.0, 1.0)  # Optimal around 3+ sentences
    
    # Combine scores
    final_score = (keyword_score * 0.5 + length_score * 0.3 + structure_score * 0.2)
    return max(0.0, min(1.0, final_score))