# backend/aegis/agents/evidence_gap_analyzer.py
"""Evidence gap analyzer - identifies gaps and weaknesses in evidence coverage."""

from __future__ import annotations

import re
from typing import Any

from aegis.core.logging import get_logger

log = get_logger(__name__)


class EvidenceGapAnalyzer:
    """Analyzes evidence coverage and identifies gaps in AI responses."""

    def analyze_evidence_gaps(
        self,
        response: str,
        evidence_items: list[dict[str, Any]],
        task: str = "",
    ) -> dict[str, Any]:
        """Analyze evidence gaps in an AI response.

        Args:
            response: The AI response text
            evidence_items: List of evidence items supporting the response
            task: Original task/query for context

        Returns:
            Dictionary containing:
            - gap_score: Overall gap score (0.0-1.0, lower is better)
            - identified_gaps: List of identified evidence gaps
            - coverage_metrics: Metrics about evidence coverage
            - recommendations: Suggestions for improving evidence
        """
        log.info("EvidenceGapAnalyzer.analyze_evidence_gaps", 
                response_length=len(response), evidence_count=len(evidence_items))

        if not response:
            return {
                "gap_score": 1.0,
                "identified_gaps": ["No response provided"],
                "coverage_metrics": {"coverage_ratio": 0.0},
                "recommendations": ["Provide a response for analysis"]
            }

        # Extract claims from response
        claims = self._extract_claims(response)
        
        # Analyze evidence coverage
        coverage_metrics = self._calculate_coverage_metrics(claims, evidence_items)
        
        # Identify specific gaps
        identified_gaps = self._identify_gaps(claims, evidence_items, coverage_metrics)
        
        # Calculate overall gap score
        gap_score = self._calculate_gap_score(coverage_metrics, identified_gaps)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(identified_gaps, coverage_metrics)

        return {
            "gap_score": gap_score,
            "identified_gaps": identified_gaps,
            "coverage_metrics": coverage_metrics,
            "recommendations": recommendations
        }

    def _extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from text."""
        claims = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue
            
            # Skip questions, commands, and metadata
            if (sentence.endswith('?') or 
                sentence.startswith(('Please', 'Could you', 'I recommend', '#', '-', '*')) or
                '|' in sentence or  # Skip table-like content
                sentence.startswith('http')):
                continue
            
            # Skip markdown formatting
            if '```' in sentence or '**' in sentence or '__' in sentence:
                continue
            
            claims.append(sentence)
        
        return claims

    def _is_factual_claim(self, text: str) -> bool:
        """Determine if text contains a factual claim."""
        # Skip questions and commands
        if text.endswith('?') or text.startswith(('Please', 'Could you', 'I recommend')):
            return False
        
        # Look for factual indicators
        factual_indicators = [
            'is', 'was', 'are', 'were', 'has', 'had', 'can', 'could',
            'report', 'show', 'indicate', 'demonstrate', 'prove',
            'according to', 'based on', 'because', 'due to',
            'data', 'study', 'research', 'analysis', 'results'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in factual_indicators)

    def _calculate_coverage_metrics(
        self, 
        claims: list[str], 
        evidence_items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Calculate evidence coverage metrics."""
        if not claims:
            return {
                "coverage_ratio": 0.0,
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "partially_supported_claims": 0,
                "evidence_density": 0.0
            }
        
        total_claims = len(claims)
        supported_claims = 0
        partially_supported_claims = 0
        unsupported_claims = 0
        
        for claim in claims:
            support_level = self._assess_claim_support(claim, evidence_items)
            if support_level == "full":
                supported_claims += 1
            elif support_level == "partial":
                partially_supported_claims += 1
            else:
                unsupported_claims += 1
        
        coverage_ratio = supported_claims / total_claims if total_claims > 0 else 0.0
        evidence_density = len(evidence_items) / total_claims if total_claims > 0 else 0.0
        
        return {
            "coverage_ratio": coverage_ratio,
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "partially_supported_claims": partially_supported_claims,
            "evidence_density": evidence_density
        }

    def _assess_claim_support(
        self, 
        claim: str, 
        evidence_items: list[dict[str, Any]]
    ) -> str:
        """Assess how well a claim is supported by evidence."""
        if not evidence_items:
            return "none"
        
        claim_lower = claim.lower()
        best_support = "none"
        
        for evidence in evidence_items:
            evidence_text = evidence.get("supporting_passage", "").lower()
            evidence_claim = evidence.get("claim", "").lower()
            
            # Check for keyword overlap using word-level matching
            claim_words = set(claim_lower.split())
            evidence_words = set(evidence_text.split())
            
            if claim_words and evidence_words:
                # Calculate Jaccard similarity
                intersection = claim_words & evidence_words
                union = claim_words | evidence_words
                similarity = len(intersection) / len(union) if union else 0
                
                if similarity > 0.2:
                    best_support = "full"
                elif similarity > 0.05:
                    best_support = "partial"
        
        return best_support

    def _identify_gaps(
        self, 
        claims: list[str], 
        evidence_items: list[dict[str, Any]],
        coverage_metrics: dict[str, Any]
    ) -> list[str]:
        """Identify specific evidence gaps."""
        gaps = []
        
        # Check for unsupported claims
        if coverage_metrics.get("unsupported_claims", 0) > 0:
            gaps.append(f"{coverage_metrics['unsupported_claims']} claims lack supporting evidence")
        
        # Check for partially supported claims
        if coverage_metrics.get("partially_supported_claims", 0) > 0:
            gaps.append(f"{coverage_metrics['partially_supported_claims']} claims have partial evidence")
        
        # Check evidence density
        if coverage_metrics.get("evidence_density", 0) < 0.5:
            gaps.append("Low evidence density - more evidence items needed")
        
        # Check coverage ratio
        if coverage_metrics.get("coverage_ratio", 0) < 0.7:
            gaps.append(f"Low coverage ratio ({coverage_metrics.get('coverage_ratio', 0):.2f}) - most claims unsupported")
        
        # Check for missing evidence types
        if evidence_items:
            sources = set(e.get("source", "") for e in evidence_items)
            if len(sources) < 2:
                gaps.append("Limited source diversity - consider adding more sources")
        
        # Check evidence recency
        if evidence_items:
            recent_evidence = sum(1 for e in evidence_items if self._is_recent_evidence(e))
            if recent_evidence < len(evidence_items) * 0.5:
                gaps.append("Some evidence may be outdated - consider adding recent sources")
        
        return gaps

    def _is_recent_evidence(self, evidence: dict[str, Any]) -> bool:
        """Check if evidence is recent (within last year)."""
        timestamp = evidence.get("timestamp", "")
        if not timestamp:
            return False
        
        # Simple check - if timestamp contains 2024 or 2025, consider it recent
        return any(year in timestamp for year in ["2024", "2025"])

    def _calculate_gap_score(
        self, 
        coverage_metrics: dict[str, Any], 
        identified_gaps: list[str]
    ) -> float:
        """Calculate overall gap score (0.0-1.0, lower is better)."""
        if not coverage_metrics:
            return 1.0
        
        # Base score from coverage ratio
        coverage_ratio = coverage_metrics.get("coverage_ratio", 0.0)
        base_score = 1.0 - coverage_ratio
        
        # Adjust for identified gaps
        gap_penalty = min(len(identified_gaps) * 0.1, 0.5)
        
        # Adjust for evidence density
        evidence_density = coverage_metrics.get("evidence_density", 0.0)
        density_bonus = max(0, 0.2 - evidence_density * 0.2)
        
        final_score = base_score + gap_penalty + density_bonus
        return min(max(final_score, 0.0), 1.0)

    def _generate_recommendations(
        self, 
        identified_gaps: list[str], 
        coverage_metrics: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations for improving evidence coverage."""
        recommendations = []
        
        if coverage_metrics.get("unsupported_claims", 0) > 0:
            recommendations.append("Add supporting evidence for unsupported claims")
        
        if coverage_metrics.get("evidence_density", 0) < 0.5:
            recommendations.append("Increase number of evidence items per claim")
        
        if coverage_metrics.get("coverage_ratio", 0) < 0.7:
            recommendations.append("Improve overall evidence coverage by adding more sources")
        
        # Check for source diversity
        recommendations.append("Consider adding diverse sources for better reliability")
        
        return recommendations


# Global instance
_evidence_gap_analyzer = EvidenceGapAnalyzer()

# Module-level functions
def analyze_evidence_gaps(
    response: str,
    evidence_items: list[dict[str, Any]] = None,
    task: str = ""
) -> dict[str, Any]:
    """Convenience function to analyze evidence gaps."""
    analyzer = _evidence_gap_analyzer
    return analyzer.analyze_evidence_gaps(response, evidence_items or [], task)