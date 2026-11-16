# backend/aegis/agents/report_generator.py
"""Report Generator & Deliverable Engine (Section 7.8).

Synthesizes all validated agent outputs, Python dataset computations, and
grounded citations into a structured 12-section Markdown report with real
visualizations rendered via :class:`VisualizationEngine`:
1. Title
2. Executive Summary
3. Objective & Scope
4. Methodology
5. Key Findings
6. Detailed Analysis
7. Supporting Evidence
8. Data Insights & Visualizations
9. Risks, Limitations & Assumptions
10. Recommendations
11. Conclusion
12. References & Sources
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider
from aegis.llm.factory import get_default_llm
from aegis.schemas.agent import ChartData, ReportOutput
from aegis.services.visualization import get_visualization_engine

log = get_logger(__name__)

_SYNTHESIS_SYSTEM_PROMPT: str = """\
You are the Principal Analyst & Report Generator of AegisAI.
Produce a comprehensive, publication-grade Markdown report grounded strictly
in the provided document excerpts, computed Python statistics, and verified research.

Structure your report with these exact Markdown headings:
# <Descriptive Report Title>
## 1. Executive Summary
## 2. Objective & Scope
## 3. Methodology
## 4. Key Findings
## 5. Detailed Analysis
## 6. Supporting Evidence
## 7. Data Insights & Visualizations
## 8. Risks, Limitations & Assumptions
## 9. Recommendations
## 10. Conclusion
## 11. References & Sources

Rules:
- Cite concrete figures, passages, and source names from the context.
- NEVER fabricate statistics, URLs, or citations.
- If a specific aspect is absent from the uploaded documents, state that limitation transparently under Risks, Limitations & Assumptions.
"""

_CHART_CANDIDATE_PROMPT: str = """\
Extract up to 3 real numerical series explicitly present in the context below.
Return ONLY a JSON object: {"charts": [{"type": "bar"|"line"|"pie", "title": "...", "labels": [...], "values": [...]}]}
If no real multi-point numerical series exist in the text, return {"charts": []}.

Context:
{context}
"""


def _has_numeric_pattern(text: str) -> bool:
    if not text:
        return False
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    return len(numbers) >= 3


def _render_bar_chart_png(
    labels: list[str],
    values: list[float],
    width: int = 800,
    height: int = 480,
    title: str = "",
    library: str = "matplotlib",
) -> str | None:
    """Backward-compatible chart helper delegating to VisualizationEngine."""
    chart = get_visualization_engine().render_chart(
        "bar",
        title=title or "Extracted Metrics",
        labels=labels,
        values=values,
    )
    return chart.png_base64 if chart else None


def _extract_sections_dict(markdown: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_key = "Overview"
    buf: list[str] = []
    for line in (markdown or "").splitlines():
        m = re.match(r"^#{1,3}\s+(.+?)\s*$", line.strip())
        if m:
            if buf:
                sections[current_key] = "\n".join(buf).strip()
                buf = []
            current_key = m.group(1).strip()
        else:
            buf.append(line)
    if buf:
        sections[current_key] = "\n".join(buf).strip()
    return sections


def _extract_title(markdown: str, fallback_task: str) -> str:
    for line in (markdown or "").splitlines():
        m = re.match(r"^#\s+(.+?)\s*$", line.strip())
        if m:
            return re.sub(r"[*_`]", "", m.group(1)).strip()[:140]
    cleaned = (fallback_task or "AegisAI Intelligence Report").strip()
    return (cleaned[:110] + "...") if len(cleaned) > 110 else cleaned


def _extract_fallback_chart_candidates(context: str) -> list[dict[str, Any]]:
    """Deterministically extract real labeled numerical series from text/tables."""
    if not context:
        return []

    # Look for "Label: 123.4" pairs across lines or pipe-delimited rows
    kv_pairs: list[tuple[str, float]] = []
    for m in re.finditer(r"([A-Za-z][A-Za-z0-9 _\-/]{1,28})\s*[:=]\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)\b", context):
        lbl = m.group(1).strip()
        try:
            val = float(m.group(2).replace(",", ""))
            if not (val.is_integer() and 1990 <= int(val) <= 2035):
                kv_pairs.append((lbl, val))
        except Exception:
            continue

    if 2 <= len(kv_pairs) <= 24:
        # Deduplicate labels keeping first occurrence
        seen: dict[str, float] = {}
        for k, v in kv_pairs:
            if k not in seen:
                seen[k] = v
            if len(seen) >= 10:
                break
        if len(seen) >= 2:
            return [
                {
                    "type": "bar",
                    "title": "Extracted Quantitative Metrics",
                    "labels": list(seen.keys()),
                    "values": list(seen.values()),
                }
            ]
    return []


def _build_structured_fallback_markdown(data: Mapping[str, Any]) -> str:
    """Build a deterministic 12-section grounded report when LLM is in dry-run/offline mode."""
    task = str(data.get("task", "") or "Analytical Request")
    task_type = str(data.get("task_type", "hybrid"))
    retrieved = str(data.get("retrieved_context", "") or "").strip()
    research = str(data.get("research_results", "") or "").strip()
    reasoning = str(data.get("reasoning_details", "") or data.get("reasoning_summary", "") or "").strip()
    critique = str(data.get("critique_assessment", "") or "").strip()
    weaknesses = list(data.get("critique_weaknesses", []) or [])
    suggestions = list(data.get("validation_suggestions", []) or [])
    evidence_items = list(data.get("evidence_items", []) or [])
    analysis_results = data.get("analysis_results") or {}
    subtasks = list(data.get("subtasks", []) or data.get("plan_steps", []) or [])

    if task_type == "conversational" and not retrieved and not research and not analysis_results:
        return (
            "# AegisAI AgentSwarm Workspace\n\n"
            "Hello! I am **AegisAI AgentSwarm** — an 8-agent autonomous research, document "
            "intelligence, dataset analytics, and deliverable generation platform.\n\n"
            "You can:\n"
            "- **Upload documents or datasets** (`PDF`, scanned `PDF`/images via OCR, `DOCX`, `PPTX`, `XLSX`, `CSV`, `JSON`, `HTML`, `MD`, `ZIP`)\n"
            "- **Ask complex multi-step questions** across single or multiple documents\n"
            "- **Run statistical profiling & generate charts** on tabular datasets\n"
            "- **Export real deliverables** in `PDF`, `DOCX`, `PPTX`, `XLSX`, `CSV`, `HTML`, and `Markdown`"
        )

    exec_summary = (
        reasoning[:1800]
        or research[:1800]
        or retrieved[:1800]
        or "Analysis executed over available inputs; no matching indexed document passages were found."
    )

    method_lines = [
        f"- **Execution Mode**: `{task_type}` orchestrated via LangGraph 8-agent StateGraph.",
        f"- **Subtasks Planned**: {len(subtasks)} structured steps with dependency validation.",
        f"- **Evidence Retrieved**: {len(evidence_items)} grounded passage(s) from Qdrant hybrid vector+BM25 search.",
    ]
    if analysis_results:
        method_lines.append(
            f"- **Dataset Profiling**: Deterministic Python (`pandas` / `scipy` / `sklearn`) over {len(analysis_results)} dataset(s)."
        )

    analysis_block = ""
    if isinstance(analysis_results, dict) and analysis_results:
        parts = []
        for ds_name, ds_info in analysis_results.items():
            if isinstance(ds_info, dict) and ds_info.get("summary"):
                parts.append(f"### Dataset: {ds_name}\n{ds_info['summary']}")
        analysis_block = "\n\n".join(parts)

    ev_lines: list[str] = []
    sources_seen: list[str] = []
    for idx, ev in enumerate(evidence_items[:12], start=1):
        if isinstance(ev, dict):
            src = str(ev.get("source", "Document"))
            if src not in sources_seen:
                sources_seen.append(src)
            conf = float(ev.get("confidence", 0.0) or 0.0)
            passage = str(ev.get("supporting_passage") or ev.get("claim") or "")[:320]
            ev_lines.append(f"- **[{idx}] {src}** *(confidence {conf:.2f})*: {passage}")

    risk_lines = [f"- {w}" for w in weaknesses[:5]]
    if not risk_lines:
        risk_lines = [
            "- Conclusions are bounded by the scope and recency of uploaded documents and configured search sources."
        ]

    rec_lines = [f"- {s}" for s in suggestions[:6]] or [
        "- Cross-validate high-impact operational metrics against primary source tables.",
        "- Export the generated XLSX/PDF deliverables for stakeholder review.",
    ]

    ref_lines = [f"{i}. `{src}`" for i, src in enumerate(sources_seen, start=1)] or [
        "1. Internal AegisAI execution trace (no external citations required)."
    ]

    return "\n\n".join([
        f"# {task[:120]}",
        f"## 1. Executive Summary\n\n{exec_summary}",
        f"## 2. Objective & Scope\n\nPrimary objective: **{task}**",
        "## 3. Methodology\n\n" + "\n".join(method_lines),
        f"## 4. Key Findings\n\n{(research[:2500] or retrieved[:2500] or exec_summary)}",
        f"## 5. Detailed Analysis\n\n{(reasoning or retrieved[:4000] or 'See Executive Summary above.')}",
        "## 6. Supporting Evidence\n\n" + ("\n".join(ev_lines) if ev_lines else "*No indexed evidence chunks matched this query.*"),
        "## 7. Data Insights & Visualizations\n\n" + (analysis_block or "*Quantitative charts are attached where numerical series were detected.*"),
        "## 8. Risks, Limitations & Assumptions\n\n" + "\n".join(risk_lines),
        "## 9. Recommendations\n\n" + "\n".join(rec_lines),
        f"## 10. Conclusion\n\n{critique or 'Multi-agent verification and synthesis completed successfully.'}",
        "## 11. References & Sources\n\n" + "\n".join(ref_lines),
    ])


_CONVERSATIONAL_SYSTEM_PROMPT: str = """\
You are AegisAI AgentSwarm's Principal Synthesis Engine responding in Conversational Swarm mode.
Produce a refined, easily understandable, high-clarity response synthesized from BOTH the Retriever agent's grounded passages and the Researcher agent's domain insights.

FORMATTING & CLARITY RULES:
1. DO NOT scatter raw ASCII symbols like '*#_|' in sentences or headings.
2. Structure your response into clear, numbered sections:
   ## 1. Executive Summary & Direct Answer
   Provide an immediate, plain-language answer addressing the user's objective with key figures.
   ## 2. Core Findings & Data Breakdown
   Whenever comparisons, numerical metrics, categories, or parameters are present, ALWAYS format them as a clean, properly structured GitHub Flavored Markdown table:
   | Metric / Parameter | Finding / Value | Source / Relevance |
   |---|---|---|
   | [Item] | [Value] | [Source] |
   Ensure the table has a valid '|---|---|' separator row so it renders as a visual table.
   ## 3. In-Depth Analysis & Nuances
   Provide clear paragraphs and clean bullet points (- Point) detailing technical implications, method details, and rationale.
   ## 4. Key Takeaways & Recommendations
   Provide 2-4 actionable next steps or decisions.
   ## 5. Verified Sources & Provenance
   List the exact documents, chunk references, or verified search sources used.
3. NEVER fabricate numbers, URLs, or claims. If information was not found in the retrieved chunks or research, state so directly.
"""


def clean_output_symbols(text: str) -> str:
    """Refine generated markdown output, eliminating raw ASCII symbols (*#_| noise) while preserving clean tables and headings."""
    if not text:
        return ""
    lines = text.splitlines()
    cleaned: list[str] = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        # Detect table line
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2:
            in_table = True
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            # Check if this is the separator row
            if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                cleaned.append("| " + " | ".join("---" for _ in cells) + " |")
            else:
                cleaned.append("| " + " | ".join(cells) + " |")
            continue

        in_table = False
        # Remove decorative symbol divider lines like ***, ___, ###, ===, *#*#*
        if re.match(r"^[ \t]*[*#_=-]{3,}[ \t]*$", stripped) and not stripped.startswith("#"):
            cleaned.append("")
            continue

        # Clean trailing hashes from headings: "## Heading ##" -> "## Heading"
        line = re.sub(r"^(#{1,6}\s+.*?)\s+#+\s*$", r"\1", line)

        # Outside of tables, replace isolated pipes with clean middle dots: "Option A | Option B" -> "Option A • Option B"
        line = re.sub(r"(\S)\s*\|\s*(\S)", r"\1 • \2", line)

        # Clean stray leading or trailing pipes on regular text lines
        line = re.sub(r"^\s*\|\s*", "", line)
        line = re.sub(r"\s*\|\s*$", "", line)

        # Clean excess raw symbols combinations
        line = re.sub(r"\*{3,}(.*?)\*{3,}", r"**\1**", line)
        line = re.sub(r"_{2,}(.*?)_{2,}", r"*\1*", line)

        cleaned.append(line)

    res = "\n".join(cleaned)
    res = re.sub(r"\n{3,}", "\n\n", res)
    return res.strip()


def _build_markdown(data: Mapping[str, object], llm: LLMProvider | None = None) -> str:
    """Synthesize the grounded Markdown report (12-section formal or fast conversational)."""
    task = str(data.get("task", ""))
    task_type = str(data.get("task_type", "hybrid"))
    conversational_mode = bool(data.get("conversational_mode", False))
    retrieved_context = str(data.get("retrieved_context", "") or "")
    research_results = str(data.get("research_results", "") or "")
    reasoning_details = str(data.get("reasoning_details", "") or data.get("reasoning_summary", "") or "")
    critique_assessment = str(data.get("critique_assessment", "") or "")
    analysis_results = data.get("analysis_results") or {}
    evidence_items = list(data.get("evidence_items", []) or [])  # type: ignore[arg-type]

    # Fast conversational greeting response
    if task_type == "conversational" and not retrieved_context and not research_results and not analysis_results:
        return clean_output_symbols(_build_structured_fallback_markdown(data))

    context_parts: list[str] = []
    if retrieved_context:
        context_parts.append(f"Retrieved Document Passages (Retriever Agent):\n{retrieved_context[:5500]}")
    if research_results:
        context_parts.append(f"Multi-Source Research Synthesis (Researcher Agent):\n{research_results[:4000]}")
    if analysis_results:
        context_parts.append(f"Verified Python Dataset Statistics (Analytics Engine):\n{str(analysis_results)[:3500]}")
    if reasoning_details:
        context_parts.append(f"Reasoner & Analyst Conclusions:\n{reasoning_details[:3000]}")
    if critique_assessment:
        context_parts.append(f"Critic Quality Audit:\n{critique_assessment[:1000]}")

    if llm and context_parts:
        if conversational_mode:
            prompt = (
                f"User Question / Objective: {task}\n\n"
                + "\n\n".join(context_parts)
                + "\n\nSynthesize BOTH the Retriever agent's document passages and Researcher agent's findings into a refined, human-readable, beautifully formatted answer. Use standard Markdown tables for any structured comparisons or metrics."
            )
            sys_prompt = _CONVERSATIONAL_SYSTEM_PROMPT
            max_tokens = 2000
        else:
            prompt = (
                f"User Objective: {task}\n\n"
                + "\n\n".join(context_parts)
                + "\n\nWrite the complete 12-section Markdown report answering the User Objective "
                "using ONLY verified facts, numbers, and citations from above. Include structured tables."
            )
            sys_prompt = _SYNTHESIS_SYSTEM_PROMPT
            max_tokens = 3600

        try:
            generated = llm.generate(
                prompt,
                system_instruction=sys_prompt,
                temperature=0.25,
                max_output_tokens=max_tokens,
            )
            if generated and not generated.startswith("[dry-run]") and len(generated.strip()) > 60:
                md = clean_output_symbols(generated.strip())
                if (
                    not conversational_mode
                    and evidence_items
                    and "References & Sources" not in md
                    and "Supporting Evidence" not in md
                ):
                    ev_block = "\n\n## References & Sources\n\n"
                    for i, ev in enumerate(evidence_items[:12], start=1):
                        if isinstance(ev, dict):
                            ev_block += (
                                f"- **[{i}] {ev.get('source', 'Source')}** "
                                f"(confidence {float(ev.get('confidence', 0.0) or 0.0):.2f}): "
                                f"{str(ev.get('supporting_passage') or ev.get('claim') or '')[:260]}\n"
                            )
                    md += ev_block
                return clean_output_symbols(md)
        except Exception as exc:
            log.warning("ReportGenerator LLM synthesis fell back to deterministic builder", error=str(exc)[:200])

    if conversational_mode and (reasoning_details or retrieved_context or research_results):
        direct = reasoning_details or research_results[:2000] or retrieved_context[:2000]
        sources = []
        for ev in evidence_items[:6]:
            if isinstance(ev, dict) and ev.get("source") and ev["source"] not in sources:
                sources.append(str(ev["source"]))
        src_footer = ("\n\n**Sources:** " + ", ".join(f"`{s}`" for s in sources)) if sources else ""
        return clean_output_symbols(f"### Grounded Swarm Synthesis\n\n{direct}{src_footer}")

    return clean_output_symbols(_build_structured_fallback_markdown(data))


def _extract_action_items(markdown: str, data: Mapping[str, object]) -> list[str]:
    items: list[str] = []
    in_rec_section = False
    for line in (markdown or "").splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,3}\s+.*(?:Recommendation|Action|Next Step)", stripped, flags=re.IGNORECASE):
            in_rec_section = True
            continue
        if in_rec_section and stripped.startswith("#"):
            in_rec_section = False
        if in_rec_section and stripped.startswith(("- ", "* ")):
            cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped[2:]).strip()
            if cleaned and cleaned not in items:
                items.append(cleaned)

    for s in (data.get("validation_suggestions") or []):  # type: ignore[union-attr]
        s_str = str(s).strip()
        if s_str and s_str not in items:
            items.append(s_str)

    if not items:
        items = [
            "Review key findings and supporting citations against operational objectives.",
            "Download generated PDF, DOCX, PPTX, or XLSX deliverables for stakeholder distribution.",
        ]
    return items[:15]


class ReportGenerator:
    """Compiles the final 12-section report and verified visualizations."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_default_llm()
        self._vis = get_visualization_engine()

    def generate(self, data: Mapping[str, object]) -> ReportOutput:
        """Produce a structured :class:`ReportOutput` from aggregated pipeline state."""
        log.info("ReportGenerator.generate")

        conversational_mode = bool(data.get("conversational_mode", False))
        if conversational_mode:
            # 1. Clean, refined, structured output for the conversational chat interface
            conv_resp = clean_output_symbols(_build_markdown(data, llm=self._llm))
            # 2. Detailed, exhaustive 12-section document for file downloads (PDF, PPTX, DOCX, MD)
            detailed_data = dict(data)
            detailed_data["conversational_mode"] = False
            detailed_md = clean_output_symbols(_build_structured_fallback_markdown(detailed_data))
            markdown = detailed_md
            conversational_response = conv_resp
        else:
            full_md = clean_output_symbols(_build_markdown(data, llm=self._llm))
            markdown = full_md
            conversational_response = full_md

        title = _extract_title(conversational_response or markdown, str(data.get("task", "AegisAI Report")))
        sections_map = _extract_sections_dict(markdown)
        sections = [{"heading": k, "content": v} for k, v in sections_map.items()]
        action_items = _extract_action_items(conversational_response or markdown, data)

        # 1. Reuse charts already computed by DataAnalysisEngine / ToolAgent
        charts: list[ChartData] = []
        precomputed = data.get("chart_data") or []
        if isinstance(precomputed, list):
            for ch in precomputed:
                if isinstance(ch, ChartData) and ch.png_base64:
                    charts.append(ch)
                elif isinstance(ch, dict):
                    img_b64 = ch.get("image_data") or ch.get("png_base64")
                    if img_b64:
                        payload = {
                            k: v for k, v in ch.items()
                            if k in ChartData.model_fields
                        }
                        payload["image_data"] = str(img_b64)
                        charts.append(ChartData(**payload))
                    elif ch.get("labels") and ch.get("values"):
                        rendered = self._vis.render_chart(
                            str(ch.get("chart_type", "bar")),
                            title=str(ch.get("title", "Data Visualization")),
                            labels=[str(x) for x in ch["labels"]],
                            values=[float(v) for v in ch["values"]],
                            metadata=ch.get("metadata") or {},
                        )
                        if rendered:
                            charts.append(rendered)

        # 2. If no dataset charts exist yet, deterministically extract real numerical series from text
        combined_text = f"{data.get('retrieved_context', '')}\n{data.get('research_results', '')}"
        if not charts and _has_numeric_pattern(combined_text):
            candidates = _extract_fallback_chart_candidates(combined_text)

            for cand in candidates[:3]:
                lbls = [str(x) for x in (cand.get("labels") or [])]
                vals_raw = cand.get("values") or []
                try:
                    vals = [float(v) for v in vals_raw]
                except Exception:
                    continue
                if len(lbls) >= 2 and len(lbls) == len(vals):
                    rendered = self._vis.render_chart(
                        str(cand.get("type", "bar")),
                        title=str(cand.get("title", "Quantitative Summary")),
                        labels=lbls,
                        values=vals,
                    )
                    if rendered:
                        charts.append(rendered)

        return ReportOutput(
            title=title,
            markdown=markdown,
            conversational_response=conversational_response,
            sections=sections,
            action_items=action_items,
            charts=charts,
        )
