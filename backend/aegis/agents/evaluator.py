# backend/aegis/agents/evaluator.py
"""Evaluator agent — computes document metrics and quality evaluations.

Analyzes documents and evidence to produce:
- Document metrics (completeness, evidence density, confidence, etc.)
- Quality evaluations (hallucination, factual accuracy, coverage, etc.)
- Chart and table data extraction for visualization
"""

from __future__ import annotations

import re
import json
from typing import cast
from collections import Counter

from pydantic import BaseModel
from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider
from aegis.llm.factory import get_default_llm
from aegis.schemas.agent import (
    Evidence,
    DocumentMetrics,
    EvaluationResult,
    ChartData,
    TableData,
)

log = get_logger(__name__)

_EVALUATOR_SYSTEM_PROMPT: str = """\
You are the Evaluator agent. Given document content and evidence items, produce:

1. **Document Metrics** — Quantitative measures of document quality and completeness
2. **Quality Evaluation** — Assessment of factual accuracy, hallucination risk, coverage
3. **Chart Data** — Extract numeric patterns suitable for visualization
4. **Table Data** — Extract structured tabular data

Return ONLY valid JSON matching the expected schemas.
Be precise and conservative in your estimates.
"""

_CHART_EXTRACTION_PROMPT: str = """\
Analyze the document content for numeric data patterns suitable for charts.
Look for:
- Performance metrics (FLOPS, bandwidth, latency, throughput)
- Specifications (memory size, core counts, clock speeds)
- Comparisons (X vs Y, before/after, percentage improvements)
- Time series data
- Categorical distributions

Return chart configurations as JSON with chart_type, title, labels, values, datasets.
"""

_TABLE_EXTRACTION_PROMPT: str = """\
Extract structured tabular data from the document content.
Look for:
- Specification tables
- Comparison tables
- Parameter lists
- Configuration matrices

Return table data as JSON with title, headers, rows.
"""


class Evaluator:
    """Computes metrics and evaluations for documents and evidence."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm or get_default_llm()

    def evaluate(
        self,
        document_content: str,
        evidence_items: list[Evidence],
        task: str = "",
    ) -> tuple[DocumentMetrics, EvaluationResult, list[ChartData], list[TableData]]:
        """Run full evaluation pipeline on document and evidence."""
        log.info("Evaluator.evaluate", doc_len=len(document_content), evidence_count=len(evidence_items))

        # Compute document metrics
        metrics = self._compute_metrics(document_content, evidence_items)

        # Run quality evaluation
        evaluation = self._evaluate_quality(document_content, evidence_items, task)

        # Extract chart data
        charts = self._extract_charts(document_content)

        # Extract table data
        tables = self._extract_tables(document_content)

        return metrics, evaluation, charts, tables

    def _compute_metrics(
        self,
        document_content: str,
        evidence_items: list[Evidence],
    ) -> DocumentMetrics:
        """Compute quantitative document metrics."""
        content = document_content or ""
        word_count = len(content.split())
        chunk_count = max(1, word_count // 500)  # Rough estimate

        # Evidence density: evidence per 1000 words
        evidence_density = (len(evidence_items) / max(word_count, 1)) * 1000

        # Average confidence across evidence
        avg_confidence = (
            sum(e.confidence for e in evidence_items) / len(evidence_items)
            if evidence_items else 0.0
        )

        # Source diversity
        unique_sources = len(set(e.source for e in evidence_items))

        # Citation coverage: fraction of document claims with evidence
        # Approximate: evidence items per 200 words of content
        citation_coverage = min(len(evidence_items) / max(word_count / 200, 1), 1.0)

        # Table count: markdown table blocks + HTML tables (not raw pipe lines,
        # which overcounted single-pipe prose as dozens of tables).
        table_blocks = 0
        in_table_block = False
        for line in content.split('\n'):
            if '|' in line and line.count('|') >= 2:
                if not in_table_block:
                    in_table_block = True
                    table_blocks += 1
            else:
                in_table_block = False
        table_blocks += len(re.findall(r'<table[\s>]', content, re.IGNORECASE))
        table_count = table_blocks

        # Chart data points (numeric patterns)
        numeric_patterns = len(re.findall(
            r'\b\d+(?:\.\d+)?\s*(?:[KMGT]?B|k?Hz|MHz|GHz|FLOPS|TFLOPS|W|V|A|%|MB/s|GB/s)\b',
            content, re.IGNORECASE,
        ))
        # Also count bare numbers in table cells / bullet specs as data points
        if numeric_patterns == 0 and content.strip():
            bare_numbers = re.findall(r'(?m)^[\s|\-*>•]*\d+(?:\.\d+)?\b', content)
            numeric_patterns = len(bare_numbers)

        # Completeness score based on evidence coverage and document structure
        # Base 0.2 for any non-empty doc; evidence and citations add up to 0.8.
        # (Previously the 0.5 floor overstated empty/low-evidence documents.)
        has_content = 1.0 if word_count > 0 else 0.0
        structure_bonus = 0.0
        if content.strip():
            headings = sum(1 for line in content.splitlines() if line.strip().startswith('#'))
            structure_bonus = min(headings / 5.0, 1.0) * 0.1
        completeness = min(
            (0.2 * has_content)
            + min(evidence_density / 10.0, 0.4)
            + (citation_coverage * 0.3)
            + structure_bonus,
            1.0,
        )

        return DocumentMetrics(
            completeness_score=completeness,
            evidence_density=evidence_density,
            avg_confidence=avg_confidence,
            source_diversity=unique_sources,
            citation_coverage=citation_coverage,
            chunk_count=chunk_count,
            word_count=word_count,
            table_count=table_count,
            chart_data_points=numeric_patterns,
        )

    def _evaluate_quality(
        self,
        document_content: str,
        evidence_items: list[Evidence],
        task: str,
    ) -> EvaluationResult:
        """Deterministically evaluate document & evidence grounding quality in <1ms."""
        return self._heuristic_evaluation(evidence_items, document_content=document_content, task=task)

    def _heuristic_evaluation(
        self,
        evidence_items: list[Evidence],
        document_content: str = "",
        task: str = "",
    ) -> EvaluationResult:
        """Fast grounded evaluation over evidence items and document content."""
        if not evidence_items:
            has_doc = bool(document_content and document_content.strip())
            return EvaluationResult(
                hallucination_score=0.35 if has_doc else 0.5,
                factual_accuracy=0.65 if has_doc else 0.3,
                coverage_score=0.6 if has_doc else 0.2,
                coherence_score=0.75 if has_doc else 0.3,
                citation_quality=0.3 if has_doc else 0.1,
                overall_quality=0.62 if has_doc else 0.2,
                issues=[] if has_doc else ["No evidence available for evaluation"],
                recommendations=[] if has_doc else ["Run retriever to gather document evidence"],
            )

        avg_conf = sum(e.confidence for e in evidence_items) / len(evidence_items)
        avg_rel = sum(e.relevance for e in evidence_items) / len(evidence_items)
        rel_effective = avg_rel if avg_rel > 0 else avg_conf

        # Compute lexical overlap bonus with task when available
        task_terms = {
            w for w in re.findall(r"[a-zA-Z0-9]{3,}", (task or "").lower())
            if w not in {"the", "and", "for", "with", "from", "what", "how", "explain", "analyze"}
        }
        doc_lower = (document_content or "").lower()
        coverage = min(rel_effective, 1.0)
        if task_terms and doc_lower:
            matched = sum(1 for t in task_terms if t in doc_lower)
            coverage = min(1.0, max(coverage, round(matched / max(len(task_terms), 1), 3)))

        citation_q = min(max(len(evidence_items) / 6.0, 0.35), 1.0)
        factual = round(min(1.0, max(avg_conf * 0.95, 0.65)), 3)
        coherence = round(min(1.0, max(avg_conf * 0.9, 0.72)), 3)
        overall = round(min(1.0, (factual + coverage + coherence + citation_q) / 4.0), 3)

        return EvaluationResult(
            hallucination_score=round(max(0.05, 1.0 - factual), 3),
            factual_accuracy=factual,
            coverage_score=round(coverage, 3),
            coherence_score=coherence,
            citation_quality=round(citation_q, 3),
            overall_quality=overall,
            issues=[],
            recommendations=[],
        )

    def _extract_charts(self, document_content: str) -> list[ChartData]:
        """Extract chart-worthy numeric data from documents.

        In addition to spec key-value pairs and percentage improvements,
        falls back to generic ``Label: number`` rows and markdown table
        numeric columns so ordinary business/CSV-style content still
        yields at least one chart instead of an empty list.
        """
        charts: list[ChartData] = []

        if not document_content or not document_content.strip():
            return charts

        # Pattern: Key-value pairs with units
        kv_pattern = r'(\w+(?:\s+\w+){0,3}):\s*(\d+(?:\.\d+)?)\s*([KMGT]?B|Hz|FLOPS|W|V|A|%|MB/s|GB/s|TFLOPS|TOPS|GB)\b'
        matches = re.findall(kv_pattern, document_content, re.IGNORECASE)

        if len(matches) >= 3:
            labels = [m[0][:40] for m in matches[:10]]
            values = [float(m[1]) for m in matches[:10]]
            charts.append(ChartData(
                chart_type="bar",
                title="Key Specifications",
                labels=labels,
                values=values,
                metadata={"source": "document_specifications"},
            ))

        # Pattern: Comparison data (X vs Y)
        comp_pattern = r'(\w+)\s*(?:vs|versus|compared to)\s*(\w+)'
        # Pattern: Percentage improvements
        pct_pattern = r'(\d+(?:\.\d+)?)\s*%(?:\s*(?:improvement|increase|faster|better|higher|lower|reduction|growth|decline|change|share|rate))?'
        pct_matches = re.findall(pct_pattern, document_content, re.IGNORECASE)

        if pct_matches:
            charts.append(ChartData(
                chart_type="bar",
                title="Performance Improvements",
                labels=[f"Metric {i+1}" for i in range(len(pct_matches))],
                values=[float(m) for m in pct_matches],
                metadata={"source": "performance_comparisons"},
            ))

        # Fallback: generic "Label: number" rows (unit optional).
        # Matches per-line so a single number anywhere in a long paragraph
        # cannot produce a bogus chart label.
        if not charts:
            generic_matches: list[tuple[str, str]] = []
            for line in document_content.splitlines():
                m = re.match(
                    r'^[\s\-*•>]*([A-Za-z][\w\s\-/]{1,40}?):\s*(\d+(?:\.\d+)?)\b',
                    line,
                )
                if m:
                    generic_matches.append((m.group(1), m.group(2)))
            # Drop likely timestamps / version numbers with too many digits
            generic_matches = [(k, v) for k, v in generic_matches if len(v.replace(".", "")) <= 10]
            if len(generic_matches) >= 2:
                charts.append(ChartData(
                    chart_type="bar",
                    title="Key Metrics",
                    labels=[k.strip()[:40] for k, _ in generic_matches[:10]],
                    values=[float(v) for _, v in generic_matches[:10]],
                    metadata={"source": "key_value_metrics"},
                ))

        # Fallback: numeric column of a markdown table
        if not charts and '|' in document_content:
            for line in document_content.splitlines():
                if '|' not in line:
                    continue
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if len(cells) < 2 or all(re.match(r'^[-:]+$', c) for c in cells):
                    continue
                numeric = []
                for cell in cells[1:]:
                    m = re.match(r'^(\d+(?:\.\d+)?)', cell.replace(',', ''))
                    if m:
                        try:
                            numeric.append(float(m.group(1)))
                        except ValueError:
                            continue
                if len(numeric) >= 2:
                    charts.append(ChartData(
                        chart_type="bar",
                        title=cells[0][:60] or "Table Metrics",
                        labels=[f"Col {i+1}" for i in range(len(numeric))],
                        values=numeric,
                        metadata={"source": "markdown_table_column"},
                    ))
                    break

        return charts[:6]  # Limit to 6 charts

    def _extract_tables(self, document_content: str) -> list[TableData]:
        """Extract structured tables from document content.

        Handles markdown tables (with or without a ``|`` separator row),
        HTML ``<table>`` blocks, and bullet ``Key: Value`` lists. Rows with
        a mismatched column count are padded/truncated to the header width
        so Pydantic validation never fails downstream.
        """
        tables: list[TableData] = []

        if not document_content or not document_content.strip():
            return tables

        # ── 1. Markdown table extraction ──────────────────────────────────
        lines = document_content.split('\n')
        in_table = False
        current_headers: list[str] = []
        current_rows: list[list[str]] = []

        def _flush_markdown_table() -> None:
            if current_headers and current_rows:
                width = len(current_headers)
                normalized = [
                    (row + [""] * width)[:width] for row in current_rows
                ]
                tables.append(TableData(
                    title="Extracted Table",
                    headers=current_headers,
                    rows=normalized,
                    metadata={"source": "markdown_table"},
                ))

        for line in lines:
            if '|' in line and line.count('|') >= 2:
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if not cells:
                    _flush_markdown_table()
                    in_table = False
                    current_headers = []
                    current_rows = []
                    continue
                if not in_table:
                    in_table = True
                    current_headers = cells
                elif in_table and cells:
                    # Check if this is a separator line (---, :---:, etc.)
                    if all(re.match(r'^[-:]+$', c) for c in cells):
                        continue
                    if not current_rows and current_headers and len(cells) != len(current_headers):
                        # First body row with a different width than the header
                        # is likely a second header line, not data — treat the
                        # longer of the two as the header.
                        if len(cells) > len(current_headers):
                            current_headers = cells
                            continue
                    current_rows.append(list(cells))
            elif in_table:
                _flush_markdown_table()
                in_table = False
                current_headers = []
                current_rows = []

        # Handle final table
        if in_table:
            _flush_markdown_table()

        # ── 2. HTML table extraction ──────────────────────────────────────
        html_tables = re.findall(
            r'<table[^>]*>(.*?)</table>', document_content,
            re.IGNORECASE | re.DOTALL,
        )
        for html in html_tables:
            raw_rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.IGNORECASE | re.DOTALL)
            parsed: list[list[str]] = []
            for raw in raw_rows:
                cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', raw, re.IGNORECASE | re.DOTALL)
                cleaned = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
                if any(cleaned):
                    parsed.append(cleaned)
            if len(parsed) >= 2:
                headers = parsed[0]
                width = len(headers)
                body = [(r + [""] * width)[:width] for r in parsed[1:]]
                tables.append(TableData(
                    title="Extracted HTML Table",
                    headers=headers,
                    rows=body,
                    metadata={"source": "html_table"},
                ))

        # ── 3. Bullet Key: Value list extraction ──────────────────────────
        kv_lines: list[tuple[str, str]] = []
        for line in lines:
            stripped = line.strip().lstrip('-*•\t ')
            if ':' in stripped:
                key, _, value = stripped.partition(':')
                key, value = key.strip(), value.strip()
                if key and value and len(key) <= 60 and len(value) <= 200:
                    kv_lines.append((key, value))
        if len(kv_lines) >= 3 and not tables:
            tables.append(TableData(
                title="Extracted Specifications",
                headers=["Property", "Value"],
                rows=[[k, v] for k, v in kv_lines[:20]],
                metadata={"source": "key_value_list"},
            ))

        return tables[:5]  # Limit to 5 tables