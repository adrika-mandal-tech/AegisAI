"""Test evidence gap analyzer."""

from __future__ import annotations

import pytest

from aegis.agents.evidence_gap_analyzer import EvidenceGapAnalyzer, analyze_evidence_gaps


class TestEvidenceGapAnalyzer:
    """Test evidence gap analysis functionality."""

    def setup_method(self):
        self.analyzer = EvidenceGapAnalyzer()

    def test_analyze_empty_response(self):
        """Test gap analysis with empty response."""
        result = self.analyzer.analyze_evidence_gaps("", [])
        
        assert result["gap_score"] == 1.0
        assert "No response provided" in result["identified_gaps"]
        assert result["coverage_metrics"]["coverage_ratio"] == 0.0

    def test_analyze_response_with_full_evidence(self):
        """Test gap analysis with well-supported response."""
        response = "The study shows improvement. Results were significant. Research demonstrates benefits."
        
        evidence = [
            {
                "claim": "The study shows improvement",
                "supporting_passage": "The study shows improvement in performance metrics",
                "source": "study.pdf",
                "confidence": 0.9,
                "relevance": 0.95
            },
            {
                "claim": "Results were significant",
                "supporting_passage": "Results were statistically significant with p-value < 0.05",
                "source": "study.pdf",
                "confidence": 0.85,
                "relevance": 0.9
            },
            {
                "claim": "Research demonstrates benefits",
                "supporting_passage": "Research demonstrates clear benefits of the intervention",
                "source": "study.pdf",
                "confidence": 0.8,
                "relevance": 0.85
            }
        ]
        
        result = self.analyzer.analyze_evidence_gaps(response, evidence)
        
        assert result["gap_score"] < 0.8  # Should have low gap score
        assert result["coverage_metrics"]["coverage_ratio"] >= 0.3  # Some coverage

    def test_analyze_response_with_no_evidence(self):
        """Test gap analysis with no evidence."""
        response = "This is a factual claim that needs evidence."
        evidence = []
        
        result = self.analyzer.analyze_evidence_gaps(response, evidence)
        
        assert result["gap_score"] > 0.5  # High gap score
        assert len(result["identified_gaps"]) > 0
        assert result["coverage_metrics"]["coverage_ratio"] == 0.0

    def test_analyze_response_with_partial_evidence(self):
        """Test gap analysis with partial evidence coverage."""
        response = "Claim one is supported. Claim two has some data. Claim three needs verification."
        
        evidence = [
            {
                "claim": "Claim one is supported",
                "supporting_passage": "Evidence for claim one is provided",
                "source": "doc1.pdf",
                "confidence": 0.9,
                "relevance": 0.9
            }
        ]
        
        result = self.analyzer.analyze_evidence_gaps(response, evidence)
        
        # With 3 claims and 1 evidence item, expect medium gap score
        assert result["gap_score"] >= 0.3
        assert result["coverage_metrics"]["coverage_ratio"] < 0.7

    def test_analyze_response_identifies_gaps(self):
        """Test that gap analysis identifies specific gaps."""
        response = """
        This claim has no evidence at all.
        This claim is partially supported.
        """
        
        evidence = []  # No evidence
        
        result = self.analyzer.analyze_evidence_gaps(response, evidence)
        
        assert len(result["identified_gaps"]) > 0
        assert any("unsupported" in gap.lower() for gap in result["identified_gaps"])
        assert any("coverage" in gap.lower() for gap in result["identified_gaps"])

    def test_analyze_response_generates_recommendations(self):
        """Test that gap analysis generates actionable recommendations."""
        response = "Some claim that needs support."
        evidence = []
        
        result = self.analyzer.analyze_evidence_gaps(response, evidence)
        
        assert len(result["recommendations"]) > 0
        assert any("evidence" in rec.lower() for rec in result["recommendations"])

    def test_analyze_response_with_task_context(self):
        """Test gap analysis with task context."""
        response = "The market analysis shows growth."
        evidence = [
            {
                "claim": "market analysis shows growth",
                "supporting_passage": "Market grew by 15% according to report",
                "source": "market_report.pdf",
                "confidence": 0.8,
                "relevance": 0.85
            }
        ]
        task = "Analyze market growth trends"
        
        result = self.analyzer.analyze_evidence_gaps(response, evidence, task)
        
        # Should work without errors
        assert "gap_score" in result
        assert "identified_gaps" in result
        assert "coverage_metrics" in result

    def test_analyze_short_response(self):
        """Test gap analysis with very short response."""
        result = self.analyzer.analyze_evidence_gaps("Short.", [])
        
        assert result["gap_score"] >= 0.0
        assert result["gap_score"] <= 1.0

    def test_convenience_function(self):
        """Test the convenience function."""
        result = analyze_evidence_gaps("Test claim", [])
        
        assert isinstance(result, dict)
        assert "gap_score" in result
        assert "identified_gaps" in result
        assert "coverage_metrics" in result
        assert "recommendations" in result

    def test_coverage_metrics_calculation(self):
        """Test coverage metrics calculation."""
        response = "Supported claim one. Supported claim two. Unsupported claim three."
        
        evidence = [
            {
                "claim": "Supported claim one",
                "supporting_passage": "Evidence for claim one",
                "source": "doc1",
                "confidence": 0.9,
                "relevance": 0.9
            },
            {
                "claim": "Supported claim two",
                "supporting_passage": "Evidence for claim two",
                "source": "doc2",
                "confidence": 0.85,
                "relevance": 0.85
            }
        ]
        
        result = self.analyzer.analyze_evidence_gaps(response, evidence)
        metrics = result["coverage_metrics"]
        
        # The response has 3 sentences after splitting, but some might be filtered out
        assert metrics["total_claims"] >= 2  # At least the 2 supported claims
        assert metrics["supported_claims"] >= 1  # At least some supported claims

    def test_gap_score_bounds(self):
        """Test that gap score is always between 0 and 1."""
        test_cases = [
            ("", []),
            ("Claim", []),
            ("Claim", [{"claim": "Claim", "supporting_passage": "Evidence", "source": "doc", "confidence": 1.0, "relevance": 1.0}]),
        ]
        
        for response, evidence in test_cases:
            result = self.analyzer.analyze_evidence_gaps(response, evidence)
            assert 0.0 <= result["gap_score"] <= 1.0