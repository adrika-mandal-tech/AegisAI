# backend/aegis/agents/researcher.py
"""Researcher agent — multi-source research with strict zero-fabrication policy (Section 7.3)."""

from __future__ import annotations

from typing import cast

from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider
from aegis.llm.factory import get_default_llm
from aegis.schemas.agent import ResearchResult
from aegis.tools.tavily_search import TavilySearch

log = get_logger(__name__)

_RESEARCHER_SYSTEM_PROMPT: str = """\
You are the Researcher agent in AegisAI.
Synthesize findings strictly from the provided document excerpts and/or
verified web search results.

STRICT ANTI-HALLUCINATION POLICY:
- NEVER invent URLs, DOIs, author names, dates, or statistics.
- Distinguish clearly between uploaded-document evidence and external web sources.
- If the provided excerpts do not answer a facet of the question, state the gap explicitly.

Return ONLY valid JSON matching the ResearchResult schema:
{
  "query": "original research question",
  "content": "structured research synthesis",
  "source": "document_excerpts | web_search | multi_source",
  "relevance_score": 0.85
}
"""


class Researcher:
    """Gathers and synthesizes multi-source information without fabricating citations."""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        search: TavilySearch | None = None,
    ) -> None:
        self._llm = llm or get_default_llm()
        self._search = search or TavilySearch()

    def research(
        self,
        topic: str,
        retrieved_context: str = "",
        *,
        enable_web_search: bool = True,
    ) -> ResearchResult:
        """Perform grounded research on *topic* combining retrieved & external sources."""
        log.info("Researcher.research", topic_preview=(topic or "")[:80], web_search=enable_web_search)

        greetings = {"hi", "hello", "hey", "thanks", "thank you"}
        if (topic or "").strip().lower().strip("!.? ") in greetings and not retrieved_context:
            return ResearchResult(query=topic, content="", source="none", relevance_score=0.0, evidence_items=[])

        raw_results = self._search.search(topic, max_results=4) if enable_web_search else []
        valid_web = [
            r for r in (raw_results or [])
            if isinstance(r, dict) and r.get("content") and not str(r.get("content", "")).startswith("[dry-run]")
        ]
        web_text = "\n\n".join(
            f"[Web: {r.get('title', 'Source')} | {r.get('url', '')}]\n{str(r.get('content', ''))[:500]}"
            for r in valid_web
        )
        web_evidence = [
            {
                "claim": str(r.get("title", ""))[:140] or topic[:100],
                "source": str(r.get("url", "web_search")),
                "supporting_passage": str(r.get("content", ""))[:400],
                "confidence": 0.78,
                "relevance": float(r.get("score", 0.8) or 0.8),
            }
            for r in valid_web
        ][:5]

        has_docs = bool(retrieved_context and len(retrieved_context.strip()) > 40)
        has_web = bool(web_text.strip())

        if has_docs or has_web:
            source_label = (
                "multi_source"
                if (has_docs and has_web)
                else ("web_search" if has_web else "document_research")
            )
            prompt_parts = [f"Research Objective: {topic[:600]}"]
            if has_docs:
                prompt_parts.append(f"Retrieved Document Passages:\n{retrieved_context[:4500]}")
            if has_web:
                prompt_parts.append(f"Verified External Web Sources:\n{web_text[:3000]}")
            prompt_parts.append(
                "Perform in-depth domain research synthesis for the objective above.\n"
                "- Extract key findings, quantitative facts, technical metrics, and specific methods.\n"
                "- Categorize by core thematic pillars.\n"
                "- Identify relationships, cross-references, and any information gaps.\n"
                "- Cite specific source documents/URLs accurately."
            )
            prompt = "\n\n".join(prompt_parts)
            try:
                result: ResearchResult = cast(
                    ResearchResult,
                    self._llm.generate_model_safe(
                        prompt,
                        ResearchResult,
                        system_instruction=_RESEARCHER_SYSTEM_PROMPT,
                    ),
                )
                if result and result.content and result.content.strip():
                    result.source = result.source or source_label
                    if web_evidence:
                        result.evidence_items = web_evidence
                    return result
            except Exception as exc:
                log.warning("Researcher LLM synthesis fallback triggered", error=str(exc)[:200])

            fallback_body = "\n\n".join(p for p in [retrieved_context[:3500], web_text[:3500]] if p.strip())
            return ResearchResult(
                query=topic,
                content=fallback_body,
                source=source_label,
                relevance_score=0.86 if has_docs else 0.75,
                evidence_items=web_evidence,
            )

        return ResearchResult(
            query=topic,
            content="",
            source="none",
            relevance_score=0.0,
            evidence_items=[],
        )
