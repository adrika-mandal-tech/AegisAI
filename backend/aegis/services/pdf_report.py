# backend/aegis/services/pdf_report.py
"""PDF report generation service.

Builds branded multi-page PDF documents from Markdown reports with:
- Title & Execution Metadata Header
- Table of Contents
- Heading hierarchy, bullets, tables, and embedded PNG charts
- Centered "Aegis AI" footer & page numbering compatible with both fpdf 1.7.x and fpdf2
"""

from __future__ import annotations

import base64
import os
import re
import tempfile
from typing import Any

from fpdf import FPDF

from aegis.core.logging import get_logger

log = get_logger(__name__)

_WATERMARK = "Aegis AI — Autonomous Multi-Agent Intelligence"


class AegisPDFReport(FPDF):
    """Custom FPDF subclass with watermark and page numbering."""

    def footer(self) -> None:
        self.set_y(-18)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(119, 115, 131)
        self.cell(0, 8, _sanitize_text(_WATERMARK), 0, 0, "C")
        self.set_y(-10)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, "R")


def _sanitize_text(text: str) -> str:
    """Normalize Unicode punctuation to clean Latin-1 so core PDF fonts never fail."""
    replacements = {
        "–": "-",
        "—": "--",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "•": "-",
        "→": "->",
        "↔": "<->",
        "×": "x",
        "²": "^2",
        "≥": ">=",
        "≤": "<=",
        "✅": "[PASS]",
        "⚠️": "[WARN]",
        "§": "Sec.",
    }
    out = text or ""
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    # Break absurdly long unbroken tokens (URLs/hashes > 75 chars) so multi_cell never overflows
    out = re.sub(r"(\S{70})(?=\S)", r"\1 ", out)
    return out.encode("latin-1", "replace").decode("latin-1")


def _parse_markdown_headings(markdown: str) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for line in (markdown or "").splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if match:
            level = len(match.group(1))
            title = _sanitize_text(re.sub(r"[*_`]", "", match.group(2).strip()))
            if title:
                headings.append({"level": level, "title": title})
    return headings


def _render_table_block(pdf: AegisPDFReport, table_lines: list[str], max_width: float) -> None:
    """Render a parsed markdown table as a styled, clean PDF grid."""
    if not table_lines:
        return
    parsed_rows: list[list[str]] = []
    for raw in table_lines:
        s = raw.strip()
        if re.match(r"^\|\s*[-:]+[-| :]*\|$", s):
            continue  # Separator row
        cells = [re.sub(r"[*_`]", "", c).strip() for c in s.strip("|").split("|")]
        if any(bool(c) for c in cells):
            parsed_rows.append(cells)
    if not parsed_rows:
        return

    col_count = max(len(r) for r in parsed_rows)
    if col_count == 0:
        return

    col_w = max_width / col_count
    pdf.ln(1.5)

    # If table might exceed page bottom, add page break
    needed_h = (len(parsed_rows) * 6.0) + 6.0
    if pdf.get_y() + needed_h > pdf.h - pdf.b_margin:
        pdf.add_page()

    pdf.set_draw_color(216, 204, 242)  # #D8CCF2

    for row_idx, row in enumerate(parsed_rows):
        is_header = row_idx == 0
        if is_header:
            pdf.set_fill_color(240, 234, 251)  # #F0EAFB
            pdf.set_text_color(79, 54, 133)   # #4F3685
            pdf.set_font("Helvetica", "B", 8)
        else:
            if row_idx % 2 == 1:
                pdf.set_fill_color(255, 255, 255)
            else:
                pdf.set_fill_color(250, 250, 244)  # #FAFAF4
            pdf.set_text_color(39, 36, 54)
            pdf.set_font("Helvetica", "", 7.5)

        padded_row = row + [""] * (col_count - len(row))
        for cell_text in padded_row:
            clean_cell = _sanitize_text(cell_text)[:50]
            pdf.cell(col_w, 6.0, clean_cell, border=1, ln=0, align="L", fill=True)
        pdf.ln(6.0)

    pdf.set_text_color(39, 36, 54)
    pdf.ln(2.0)


def _render_markdown(pdf: AegisPDFReport, markdown: str, max_width: float) -> int:
    heading_count = 0
    table_buffer: list[str] = []

    def _flush_table() -> None:
        nonlocal table_buffer
        if table_buffer:
            _render_table_block(pdf, table_buffer, max_width)
            table_buffer = []

    for raw_line in (markdown or "").splitlines():
        clean_md = re.sub(r"\*\*(.*?)\*\*", r"\1", raw_line.rstrip())
        line = _sanitize_text(clean_md)
        stripped = line.strip()

        # Check for table line
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2:
            table_buffer.append(stripped)
            continue
        else:
            _flush_table()

        if stripped in {"---", "***", "___"}:
            pdf.ln(3)
            y = pdf.get_y()
            pdf.set_draw_color(220, 216, 230)
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(3)
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            heading_count += 1
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip() or "(Untitled)"
            # Clean any trailing hashes
            title = re.sub(r"\s+#+$", "", title)
            pdf.set_text_color(79, 54, 133)  # #4F3685 deep purple
            if level == 1:
                pdf.set_font("Helvetica", "B", 15)
                pdf.ln(5)
                pdf.multi_cell(max_width, 8, title)
                pdf.ln(2)
            elif level == 2:
                pdf.set_font("Helvetica", "B", 12)
                pdf.ln(4)
                pdf.multi_cell(max_width, 7, title)
                pdf.ln(1.5)
            else:
                pdf.set_font("Helvetica", "B", 10.5)
                pdf.ln(3)
                pdf.multi_cell(max_width, 6, title)
                pdf.ln(1)
            pdf.set_text_color(39, 36, 54)  # #272436 primary text
            continue

        if stripped.startswith(("- ", "* ")):
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_x(pdf.l_margin + 4)
            clean_item = stripped[2:].lstrip("*_`#|")
            pdf.multi_cell(max_width - 4, 5.5, f"- {clean_item}")
            continue

        if not stripped:
            pdf.ln(2.5)
            continue

        pdf.set_font("Helvetica", "", 9.5)
        clean_para = re.sub(r"(\S)\s*\|\s*(\S)", r"\1 - \2", stripped)
        pdf.multi_cell(max_width, 5.5, clean_para)
        pdf.ln(1)

    _flush_table()
    return heading_count


def generate_report_pdf(
    markdown: str,
    title: str = "AegisAI Report",
    charts: list[Any] | None = None,
    execution_id: str = "",
) -> bytes:
    """Generate a branded multi-page PDF report with optional embedded PNG charts."""
    log.info("generate_report_pdf", title=title[:80])
    pdf = AegisPDFReport()
    pdf.set_auto_page_break(auto=True, margin=22)
    pdf.set_left_margin(16)
    pdf.set_right_margin(16)
    max_width = pdf.w - pdf.l_margin - pdf.r_margin

    headings = _parse_markdown_headings(markdown)
    sanitized_title = _sanitize_text(title or "AegisAI Report")

    # Cover / Header Banner
    pdf.add_page()
    pdf.set_fill_color(79, 54, 133)  # #4F3685
    pdf.rect(0, 0, pdf.w, 38, "F")
    pdf.set_y(11)
    pdf.set_text_color(243, 230, 107)  # #F3E66B lemon accent
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(max_width, 6, "AEGISAI AGENTSWARM - EXECUTIVE DELIVERABLE", 0, 1, "L")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(max_width, 7, sanitized_title[:110])

    pdf.set_y(44)
    pdf.set_text_color(39, 36, 54)
    if execution_id:
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(119, 115, 131)
        pdf.cell(max_width, 5, _sanitize_text(f"Execution ID: {execution_id}"), 0, 1, "L")
        pdf.ln(3)
        pdf.set_text_color(39, 36, 54)

    if headings:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(79, 54, 133)
        pdf.cell(max_width, 7, "Contents Overview", 0, 1, "L")
        pdf.set_text_color(39, 36, 54)
        pdf.set_font("Helvetica", "", 9.5)
        for heading in headings[:24]:
            indent = min(heading["level"] - 1, 3) * 5
            pdf.set_x(pdf.l_margin + indent)
            pdf.multi_cell(max_width - indent, 5.2, f"- {heading['title']}")
        pdf.ln(5)

    _render_markdown(pdf, markdown, max_width)

    # Append rendered PNG charts
    if charts:
        for idx, ch in enumerate(charts[:8]):
            b64 = ch.png_base64 if hasattr(ch, "png_base64") else (ch.get("png_base64") if isinstance(ch, dict) else "")
            ch_title = ch.title if hasattr(ch, "title") else (ch.get("title", f"Chart {idx+1}") if isinstance(ch, dict) else f"Chart {idx+1}")
            if not b64:
                continue
            tmp_img: str | None = None
            try:
                img_bytes = base64.b64decode(b64)
                fd, tmp_img = tempfile.mkstemp(suffix=".png", prefix="aegis_pdf_chart_")
                os.close(fd)
                with open(tmp_img, "wb") as fh:
                    fh.write(img_bytes)
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(79, 54, 133)
                pdf.multi_cell(max_width, 7, _sanitize_text(f"Visualization {idx + 1}: {ch_title}"))
                pdf.ln(3)
                pdf.image(tmp_img, x=pdf.l_margin, w=max_width)
                pdf.set_text_color(39, 36, 54)
            except Exception as exc:
                log.warning("PDF chart embed skipped", error=str(exc)[:160])
            finally:
                if tmp_img and os.path.exists(tmp_img):
                    try:
                        os.unlink(tmp_img)
                    except Exception:
                        pass

    output = pdf.output(dest="S")
    if isinstance(output, str):
        return output.encode("latin-1", "replace")
    return bytes(output)
