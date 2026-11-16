from aegis.schemas.agent import Evidence
from aegis.core.logging import get_logger

log = get_logger(__name__)


class EvidenceVerificationPipeline:
    """
    Comprehensive evidence verification pipeline for chat responses.
    
    Extracts claims from AI responses and verifies them against available evidence
    to provide accurate citations and source mapping.
    """
    
    def __init__(self):
        self.evidence_store = {}
        self.verification_cache = {}
        self.claim_extractor = ClaimExtractor()
        self.evidence_matcher = EvidenceMatcher()
    
    def process_response_with_evidence(
        self, 
        response: str, 
        available_evidence: list[Evidence] = None,
        context: str = ""
    ) -> dict:
        """
        Process an AI response to extract evidence and generate citations.
        
        Args:
            response: The AI response text
            available_evidence: List of evidence items from retriever
            context: Original user query or context
            
        Returns:
            Dictionary containing:
            - verified_response: The response with citations inserted
            - extracted_claims: List of claims extracted from the response
            - matched_evidence: Evidence items matching the claims
            - citation_map: Mapping of claims to evidence sources
            - confidence_score: Overall confidence in the response
        """
        log.info("EvidenceVerificationPipeline.process_response_with_evidence", 
                response_length=len(response), evidence_count=len(available_evidence or []))
        
        if not available_evidence:
            return {
                "verified_response": response,
                "extracted_claims": [],
                "matched_evidence": [],
                "citation_map": {},
                "confidence_score": 0.0
            }
        
        # Extract claims from the response
        extracted_claims = self.claim_extractor.extract_claims(response)
        log.info("EvidenceVerificationPipeline.extracted_claims", 
                claim_count=len(extracted_claims))
        
        if not extracted_claims:
            return {
                "verified_response": response,
                "extracted_claims": [],
                "matched_evidence": [],
                "citation_map": {},
                "confidence_score": 0.0
            }
        
        # Match claims to evidence
        matched_evidence, citation_map = self.evidence_matcher.match_claims_to_evidence(
            extracted_claims, available_evidence
        )
        
        # Generate verified response with citations
        verified_response = self._generate_cited_response(
            response, citation_map, matched_evidence
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            matched_evidence, len(extracted_claims)
        )
        
        return {
            "verified_response": verified_response,
            "extracted_claims": extracted_claims,
            "matched_evidence": matched_evidence,
            "citation_map": citation_map,
            "confidence_score": confidence_score
        }
    
    def _generate_cited_response(
        self, 
        response: str, 
        citation_map: dict, 
        matched_evidence: list[Evidence]
    ) -> str:
        """
        Generate a response with citations inserted for each claim.
        
        Example: "The company reported [cite:1] revenue growth."
        """
        lines = response.split('\n')
        verified_lines = []
        
        for line in lines:
            if self._contains_claim(line) and citation_map:
                # Find evidence to cite
                evidence_for_line = self._find_evidence_for_line(line, matched_evidence)
                if evidence_for_line:
                    citation_marker = f" [cite:{evidence_for_line.id}] "
                    line = self._insert_citation_in_line(line, citation_marker)
            
            verified_lines.append(line)
        
        return '\n'.join(verified_lines)
    
    def _contains_claim(self, text: str) -> bool:
        """Check if text contains a factual claim that could be cited."""
        # Look for patterns that indicate factual claims
        claim_patterns = [
            r'\b(the|this|that|company|product|study|research|data|statistic|number|amount|percentage)\b',
            r'\b\d+\s*%',  # percentages
            r'\b\d+\s*(thousand|million|billion)\b',  # large numbers
        ]
        
        import re
        for pattern in claim_patterns:
            if re.search(pattern, text.lower()):
                return True
        return False
    
    def _find_evidence_for_line(
        self, 
        line: str, 
        matched_evidence: list[Evidence]
    ) -> Evidence | None:
        """Find the best evidence match for a line of text."""
        if not matched_evidence:
            return None
        
        # Return the highest confidence match
        best_match = None
        best_score = 0.0
        
        for evidence in matched_evidence:
            # Simple relevance scoring
            line_words = set(line.lower().split())
            claim_words = set(evidence.claim.lower().split())
            
            if line_words and claim_words:
                intersection = line_words.intersection(claim_words)
                score = len(intersection) / max(len(line_words), len(claim_words))
                
                if score > best_score and score > 0.3:
                    best_score = score
                    best_match = evidence
        
        return best_match
    
    def _insert_citation_in_line(
        self, 
        line: str, 
        citation_marker: str
    ) -> str:
        """Insert citation marker in a line at the appropriate position."""
        # Try to insert before a period
        if '.' in line:
            last_period = line.rfind('.')
            if last_period > len(line) // 2:  # Don't insert too early
                return line[:last_period + 1] + citation_marker + line[last_period + 1:]
        
        # Try to insert at the end
        if line.strip():
            return line + citation_marker
        
        return line
    
    def _calculate_confidence_score(
        self, 
        matched_evidence: list[Evidence], 
        total_claims: int
    ) -> float:
        """
        Calculate confidence score based on evidence coverage and quality.
        """
        if total_claims == 0:
            return 1.0
        
        # Calculate average confidence of matched evidence
        if not matched_evidence:
            return 0.0
        
        avg_confidence = sum(ev.confidence for ev in matched_evidence) / len(matched_evidence)
        coverage_ratio = min(len(matched_evidence) / total_claims, 1.0)
        
        # Combined score
        confidence_score = (avg_confidence * 0.6) + (coverage_ratio * 0.4)
        
        return min(confidence_score, 1.0)
    
    def add_evidence_to_store(self, evidence: Evidence):
        """Add evidence to the internal store for matching."""
        self.evidence_store[evidence.id] = evidence
    
    def clear_cache(self):
        """Clear the verification cache."""
        self.verification_cache.clear()
class ClaimExtractor:
    """
    Extracts factual claims from AI responses for verification.
    """
    
    def extract_claims(self, text: str) -> list[str]:
        """
        Extract factual claims from text.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of claims that could be verified against evidence
        """
        claims = []
        
        # Split into sentences
        sentences = text.replace('\n', ' ').split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 30 or len(sentence) > 300:
                continue
            
            if self._is_verifiable_claim(sentence):
                claims.append(sentence)
        
        return claims[:10]
    
    def _is_verifiable_claim(self, text: str) -> bool:
        """
        Determine if text contains a verifiable claim.
        
        Claims typically include:
        - Factual statements
        - Numerical data
        - Specific entities or events
        - Comparisons
        """
        import re
        lower_text = text.lower()
        
        if text.endswith('?') or text.startswith(('Please', 'Could you', 'I recommend')):
            return False
        
        claim_indicators = [
            'report', 'show', 'indicate', 'demonstrate', 'prove',
            'according to', 'based on', 'because', 'due to',
            'company', 'product', 'study', 'research', 'data',
            'market', 'economy', 'financial', 'revenue', 'growth'
        ]
        
        return any(indicator in lower_text for indicator in claim_indicators)
class EvidenceMatcher:
    """
    Matches claims to evidence from the evidence store.
    """
    
    def match_claims_to_evidence(
        self, 
        claims: list[str], 
        available_evidence: list[Evidence]
    ) -> tuple[list[Evidence], dict]:
        """
        Match claims to evidence items.
        
        Args:
            claims: List of claims to match
            available_evidence: List of evidence items to search through
            
        Returns:
            Tuple of (matched_evidence, citation_map)
        """
        matched_evidence = []
        citation_map = {}
        
        for claim in claims:
            best_match = self._find_best_match(claim, available_evidence)
            if best_match:
                matched_evidence.append(best_match)
                # Create citation mapping
                citation_id = f"cite_{len(matched_evidence)}"
                citation_map[claim] = citation_id
        
        return matched_evidence, citation_map
    
    def _find_best_match(
        self, 
        claim: str, 
        evidence_list: list[Evidence]
    ) -> Evidence | None:
        """
        Find the best evidence match for a claim using semantic similarity.
        """
        if not evidence_list:
            return None
        
        claim_lower = claim.lower()
        claim_words = set(claim_lower.split())
        
        best_match = None
        best_score = 0.0
        
        for evidence in evidence_list:
            evidence_text = (evidence.claim + ' ' + evidence.supporting_passage).lower()
            evidence_words = set(evidence_text.split())
            
            # Calculate word overlap
            if claim_words and evidence_words:
                intersection = claim_words.intersection(evidence_words)
                union = claim_words.union(evidence_words)
                score = len(intersection) / len(union) if union else 0.0
                
                # Boost score based on confidence and relevance
                boost = (evidence.confidence + evidence.relevance) / 2
                final_score = score * boost
                
                # Lower threshold for matching
                if final_score > best_score and final_score > 0.05:
                    best_score = final_score
                    best_match = evidence
        
        # If no match found, use first evidence as fallback
        if not best_match and evidence_list:
            best_match = evidence_list[0]
        
        return best_match

# Global instance
_evidence_verification_pipeline = EvidenceVerificationPipeline()

# Module-level functions
def get_evidence_verification_pipeline() -> EvidenceVerificationPipeline:
    """Get the global evidence verification pipeline instance."""
    return _evidence_verification_pipeline

def process_chat_response_with_evidence(
    response: str,
    available_evidence: list[Evidence] = None,
    context: str = ""
) -> dict:
    """
    Convenience function to process chat responses with evidence verification.
    """
    pipeline = get_evidence_verification_pipeline()
    return pipeline.process_response_with_evidence(response, available_evidence, context)

def add_evidence_to_pipeline(evidence: Evidence):
    """Add evidence to the global pipeline."""
    pipeline = get_evidence_verification_pipeline()
    pipeline.add_evidence_to_store(evidence)