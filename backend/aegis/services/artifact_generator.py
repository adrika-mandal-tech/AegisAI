# backend/aegis/services/artifact_generator.py
"""Deliverable & Artifact Generation Engine (Section 11).

Generates real, verified, downloadable files linked to ``execution_id`` across
all 7 mandatory formats:
1. ``pdf`` — Branded multi-page PDF report with embedded charts
2. ``docx`` — Microsoft Word report with headings, tables, and embedded PNG charts
3. ``pptx`` — Widescreen executive presentation deck (16:9) with charts & tables
4. ``xlsx`` — Multi-sheet Excel workbook (Summary, Action Items, Dataset Stats, Tables)
5. ``csv`` — Structured findings / dataset statistics CSV export
6. ``html`` — Standalone styled HTML report in the Lemon & Purple light palette
7. ``md`` — Clean 12-section Markdown deliverable
"""

from __future__ import annotations

import base64
import csv
import html as html_lib
import io
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aegis.core.config import settings
from aegis.core.logging import get_logger
from aegis.schemas.agent import ArtifactItem
from aegis.services.pdf_report import generate_report_pdf

log = get_logger(__name__)

ALL_DELIVERABLE_FORMATS: list[str] = ["pdf", "docx", "pptx", "xlsx", "csv", "html", "md"]

MIME_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "html": "text/html",
    "md": "text/markdown",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (title or "aegis_report").lower()).strip("_")
    return (slug[:42] or "aegis_report")


def _extract_chart_list(charts: list[Any] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for ch in charts or []:
        if hasattr(ch, "model_dump"):
            normalized.append(ch.model_dump())
        elif isinstance(ch, dict):
            normalized.append(ch)
    return normalized


def _parse_sections_from_markdown(markdown: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "Executive Summary"
    current_lines: list[str] = []
    for line in (markdown or "").splitlines():
        m = re.match(r"^(#{1,3})\s+(.+?)\s*$", line.strip())
        if m:
            if current_lines:
                sections.append((current_title, current_lines))
                current_lines = []
            current_title = re.sub(r"[*_`]", "", m.group(2)).strip()
        else:
            if line.strip():
                current_lines.append(re.sub(r"\*\*(.*?)\*\*", r"\1", line.strip()))
    if current_lines:
        sections.append((current_title, current_lines))
    return sections


class ArtifactEngine:
    """Produces and verifies downloadable deliverables under ``AEGIS_ARTIFACT_ROOT``."""

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root = Path(root_dir or getattr(settings, "AEGIS_ARTIFACT_ROOT", "./data/artifacts"))
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def storage_root(self) -> Path:
        return self._root

    def execution_dir(self, execution_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "", execution_id) or "default"
        d = self._root / safe_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def list_artifacts(self, execution_id: str | None = None) -> list[dict[str, Any]]:
        """List verified artifacts on disk for a specific execution or all executions."""
        items: list[dict[str, Any]] = []
        if not self._root.exists():
            return items

        dirs = [self.execution_dir(execution_id)] if execution_id else sorted(
            [p for p in self._root.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in dirs[:50]:
            exec_id = d.name
            for f in sorted(d.iterdir()):
                if not f.is_file() or f.name.startswith("chart_") or f.stat().st_size <= 0:
                    continue
                ext = f.suffix.lower().lstrip(".")
                if ext not in MIME_TYPES:
                    continue
                items.append(
                    ArtifactItem(
                        artifact_id=f"{exec_id}:{f.name}",
                        execution_id=exec_id,
                        filename=f.name,
                        format=ext,
                        mime_type=MIME_TYPES[ext],
                        size_bytes=f.stat().st_size,
                        download_url=f"/api/artifacts/{exec_id}/{f.name}",
                        created_at=datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
                    ).model_dump()
                )
        return items

    def resolve_file(self, execution_id: str, filename: str) -> Path | None:
        safe_exec = re.sub(r"[^a-zA-Z0-9_\-]", "", execution_id)
        safe_name = Path(filename).name
        candidate = (self._root / safe_exec / safe_name).resolve()
        if not str(candidate).startswith(str(self._root.resolve())):
            return None
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
        return None

    def generate_all(
        self,
        *,
        execution_id: str,
        title: str,
        markdown: str,
        sections: dict[str, str] | None = None,
        action_items: list[str] | None = None,
        charts: list[Any] | None = None,
        tables: list[dict[str, Any]] | None = None,
        analysis_results: dict[str, Any] | None = None,
        evidence_items: list[dict[str, Any]] | None = None,
        requested_formats: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate & verify requested formats (defaults to all 7 formats)."""
        formats = [
            f.lower().lstrip(".")
            for f in (requested_formats or ALL_DELIVERABLE_FORMATS)
            if f.lower().lstrip(".") in MIME_TYPES
        ]
        if not formats:
            formats = list(ALL_DELIVERABLE_FORMATS)
        # Ensure core formats are always available alongside any requested format
        for core_fmt in ALL_DELIVERABLE_FORMATS:
            if core_fmt not in formats:
                formats.append(core_fmt)

        out_dir = self.execution_dir(execution_id)
        slug = _slugify(title)
        norm_charts = _extract_chart_list(charts)
        created: list[dict[str, Any]] = []

        generators = {
            "pdf": lambda p: self._write_pdf(p, title, markdown, norm_charts, execution_id),
            "docx": lambda p: self._write_docx(p, title, markdown, norm_charts, tables or [], execution_id),
            "pptx": lambda p: self._write_pptx(p, title, markdown, action_items or [], norm_charts, tables or [], execution_id),
            "xlsx": lambda p: self._write_xlsx(p, title, markdown, action_items or [], tables or [], analysis_results or {}, evidence_items or []),
            "csv": lambda p: self._write_csv(p, title, action_items or [], tables or [], analysis_results or {}, evidence_items or []),
            "html": lambda p: self._write_html(p, title, markdown, norm_charts, tables or [], execution_id),
            "md": lambda p: p.write_text(markdown or f"# {title}\n", encoding="utf-8"),
        }

        for fmt in formats:
            fname = f"{slug}.{fmt}"
            fpath = out_dir / fname
            try:
                generators[fmt](fpath)
                if fpath.is_file() and fpath.stat().st_size > 0:
                    created.append(
                        ArtifactItem(
                            artifact_id=f"{execution_id}:{fname}",
                            execution_id=execution_id,
                            filename=fname,
                            format=fmt,
                            mime_type=MIME_TYPES[fmt],
                            size_bytes=fpath.stat().st_size,
                            download_url=f"/api/artifacts/{execution_id}/{fname}",
                            created_at=_utc_now_iso(),
                        ).model_dump()
                    )
            except Exception as exc:
                log.warning("Artifact generation failed for format", format=fmt, error=str(exc)[:220])

        return created

    # ── Format Writers ────────────────────────────────────────────────────

    @staticmethod
    def _write_pdf(
        path: Path,
        title: str,
        markdown: str,
        charts: list[dict[str, Any]],
        execution_id: str,
    ) -> None:
        pdf_bytes = generate_report_pdf(
            markdown=markdown or f"# {title}",
            title=title,
            charts=charts,
            execution_id=execution_id,
        )
        path.write_bytes(pdf_bytes)

    @staticmethod
    def _write_docx(
        path: Path,
        title: str,
        markdown: str,
        charts: list[dict[str, Any]],
        tables: list[dict[str, Any]],
        execution_id: str,
    ) -> None:
        from docx import Document  # noqa: PLC0415
        from docx.shared import Inches, Pt, RGBColor  # noqa: PLC0415

        doc = Document()
        h0 = doc.add_heading(title or "AegisAI Executive Report", level=0)
        for run in h0.runs:
            run.font.color.rgb = RGBColor(0x4F, 0x36, 0x85)

        meta_p = doc.add_paragraph()
        r_meta = meta_p.add_run(
            f"Generated by AegisAI AgentSwarm | Execution ID: {execution_id} | {_utc_now_iso()[:19]} UTC"
        )
        r_meta.italic = True
        r_meta.font.size = Pt(9)
        r_meta.font.color.rgb = RGBColor(0x77, 0x73, 0x83)

        table_buffer: list[str] = []

        def _flush_docx_table() -> None:
            nonlocal table_buffer
            if not table_buffer:
                return
            parsed_rows: list[list[str]] = []
            for raw in table_buffer:
                s = raw.strip()
                if re.match(r"^\|\s*[-:]+[-| :]*\|$", s):
                    continue
                cells = [re.sub(r"[*_`]", "", c).strip() for c in s.strip("|").split("|")]
                if any(bool(c) for c in cells):
                    parsed_rows.append(cells)
            table_buffer = []
            if not parsed_rows:
                return
            col_count = max(len(r) for r in parsed_rows)
            tbl = doc.add_table(rows=len(parsed_rows), cols=col_count)
            tbl.style = "Table Grid"
            for r_idx, r in enumerate(parsed_rows):
                padded = r + [""] * (col_count - len(r))
                for c_idx, cell_text in enumerate(padded):
                    cell = tbl.rows[r_idx].cells[c_idx]
                    cell.text = cell_text
                    if r_idx == 0:
                        for p in cell.paragraphs:
                            for run in p.runs:
                                run.font.bold = True
                                run.font.color.rgb = RGBColor(0x4F, 0x36, 0x85)

        for raw_line in (markdown or "").splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2:
                table_buffer.append(stripped)
                continue
            else:
                _flush_docx_table()

            if not stripped or stripped in {"---", "***", "___"}:
                continue
            m = re.match(r"^(#{1,4})\s+(.+?)\s*$", stripped)
            if m:
                lvl = min(len(m.group(1)), 3)
                clean_heading = re.sub(r"\s+#+$", "", re.sub(r"[*_`]", "", m.group(2)))
                h = doc.add_heading(clean_heading, level=lvl)
                for run in h.runs:
                    run.font.color.rgb = RGBColor(0x4F, 0x36, 0x85)
                continue
            if stripped.startswith(("- ", "* ")):
                clean_item = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped[2:])
                clean_item = re.sub(r"[*_`#|]", "", clean_item).strip()
                doc.add_paragraph(clean_item, style="List Bullet")
                continue

            clean_line = re.sub(r"(\S)\s*\|\s*(\S)", r"\1 - \2", stripped)
            clean_line = re.sub(r"\*\*(.*?)\*\*", r"\1", clean_line)
            doc.add_paragraph(clean_line)

        _flush_docx_table()

        # Structured tables
        for tbl_data in tables[:6]:
            headers = tbl_data.get("headers") or []
            rows = tbl_data.get("rows") or []
            if not headers:
                continue
            doc.add_heading(str(tbl_data.get("title", "Data Table")), level=2)
            tbl = doc.add_table(rows=1, cols=len(headers))
            tbl.style = "Table Grid"
            for i, h_text in enumerate(headers):
                tbl.rows[0].cells[i].text = str(h_text)
            for row in rows[:80]:
                row_cells = tbl.add_row().cells
                for i in range(min(len(headers), len(row))):
                    row_cells[i].text = str(row[i])

        # Embedded PNG charts
        for idx, ch in enumerate(charts[:8]):
            b64 = ch.get("png_base64", "")
            if not b64:
                continue
            doc.add_heading(str(ch.get("title", f"Chart {idx + 1}")), level=2)
            img_stream = io.BytesIO(base64.b64decode(b64))
            doc.add_picture(img_stream, width=Inches(5.8))

        doc.save(str(path))

    @staticmethod
    def _write_pptx(
        path: Path,
        title: str,
        markdown: str,
        action_items: list[str],
        charts: list[dict[str, Any]],
        tables: list[dict[str, Any]],
        execution_id: str,
    ) -> None:
        from pptx import Presentation  # noqa: PLC0415
        from pptx.dml.color import RGBColor  # noqa: PLC0415
        from pptx.util import Inches, Pt  # noqa: PLC0415

        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        blank_layout = prs.slide_layouts[6]

        def _add_styled_header(slide: Any, slide_title: str, subtitle: str = "") -> None:
            bg = slide.background
            fill = bg.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(0xFA, 0xFA, 0xF4)  # #FAFAF4

            band = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
            band.fill.solid()
            band.fill.fore_color.rgb = RGBColor(0x4F, 0x36, 0x85)  # #4F3685
            band.line.fill.background()

            accent = slide.shapes.add_shape(1, Inches(0), Inches(1.15), Inches(13.333), Inches(0.08))
            accent.fill.solid()
            accent.fill.fore_color.rgb = RGBColor(0xF3, 0xE6, 0x6B)  # #F3E66B
            accent.line.fill.background()

            tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.22), Inches(12.0), Inches(0.75))
            tf = tb.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = slide_title[:95]
            p.font.size = Pt(22)
            p.font.bold = True
            p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            if subtitle:
                p2 = tf.add_paragraph()
                p2.text = subtitle[:120]
                p2.font.size = Pt(11)
                p2.font.color.rgb = RGBColor(0xF3, 0xE6, 0x6B)

        # Slide 1: Cover
        cover = prs.slides.add_slide(blank_layout)
        bg = cover.background.fill
        bg.solid()
        bg.fore_color.rgb = RGBColor(0x4F, 0x36, 0x85)
        pill = cover.shapes.add_shape(1, Inches(0.8), Inches(1.5), Inches(4.2), Inches(0.45))
        pill.fill.solid()
        pill.fill.fore_color.rgb = RGBColor(0xF3, 0xE6, 0x6B)
        pill.line.fill.background()
        ptf = pill.text_frame.paragraphs[0]
        ptf.text = "AEGISAI AGENTSWARM BRIEFING"
        ptf.font.size = Pt(11)
        ptf.font.bold = True
        ptf.font.color.rgb = RGBColor(0x27, 0x24, 0x36)

        ctb = cover.shapes.add_textbox(Inches(0.8), Inches(2.3), Inches(11.6), Inches(2.8))
        ctf = ctb.text_frame
        ctf.word_wrap = True
        cp = ctf.paragraphs[0]
        cp.text = (title or "Executive Intelligence Presentation")[:140]
        cp.font.size = Pt(34)
        cp.font.bold = True
        cp.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cp2 = ctf.add_paragraph()
        cp2.text = f"Execution ID: {execution_id}  •  Generated {_utc_now_iso()[:10]}"
        cp2.font.size = Pt(13)
        cp2.font.color.rgb = RGBColor(0xF0, 0xEA, 0xFB)

        # Content & Table slides from Markdown sections
        parsed_secs = _parse_sections_from_markdown(markdown)
        for sec_title, sec_lines in parsed_secs[:8]:
            table_lines = [ln.strip() for ln in sec_lines if ln.strip().startswith("|") and ln.strip().endswith("|")]
            text_lines = [ln for ln in sec_lines if not (ln.strip().startswith("|") and ln.strip().endswith("|"))]

            if text_lines:
                slide = prs.slides.add_slide(blank_layout)
                _add_styled_header(slide, sec_title)
                box = slide.shapes.add_textbox(Inches(0.7), Inches(1.55), Inches(11.9), Inches(5.4))
                tf = box.text_frame
                tf.word_wrap = True
                clean_count = 0
                for ln in text_lines[:9]:
                    clean_ln = re.sub(r"[*_`#|]", "", ln.lstrip("-* ")).strip()[:210]
                    if not clean_ln:
                        continue
                    para = tf.paragraphs[0] if clean_count == 0 else tf.add_paragraph()
                    clean_count += 1
                    para.text = f"• {clean_ln}"
                    para.space_after = Pt(8)
                    para.font.size = Pt(14)
                    para.font.color.rgb = RGBColor(0x27, 0x24, 0x36)

            if table_lines and len(table_lines) >= 2:
                p_rows = []
                for tl in table_lines:
                    if re.match(r"^\|\s*[-:]+[-| :]*\|$", tl):
                        continue
                    p_cells = [re.sub(r"[*_`]", "", c).strip() for c in tl.strip("|").split("|")]
                    if any(bool(c) for c in p_cells):
                        p_rows.append(p_cells)
                if p_rows:
                    t_slide = prs.slides.add_slide(blank_layout)
                    _add_styled_header(t_slide, f"Data Breakdown: {sec_title[:55]}")
                    t_cols = max(len(r) for r in p_rows)
                    t_shape = t_slide.shapes.add_table(
                        min(len(p_rows), 9),
                        t_cols,
                        Inches(0.8),
                        Inches(1.55),
                        Inches(11.7),
                        Inches(0.55 * min(len(p_rows), 9)),
                    )
                    t_obj = t_shape.table
                    for r_idx, r in enumerate(p_rows[:9]):
                        padded = r + [""] * (t_cols - len(r))
                        for c_idx, c_val in enumerate(padded):
                            cell = t_obj.cell(r_idx, c_idx)
                            cell.text = c_val[:60]
                            cp = cell.text_frame.paragraphs[0]
                            cp.font.size = Pt(11)
                            if r_idx == 0:
                                cell.fill.solid()
                                cell.fill.fore_color.rgb = RGBColor(0x4F, 0x36, 0x85)
                                cp.font.bold = True
                                cp.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                            else:
                                cell.fill.solid()
                                cell.fill.fore_color.rgb = RGBColor(0xFA, 0xFA, 0xF4) if r_idx % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
                                cp.font.color.rgb = RGBColor(0x27, 0x24, 0x36)

        # Chart slides
        for idx, ch in enumerate(charts[:6]):
            b64 = ch.get("png_base64", "")
            if not b64:
                continue
            slide = prs.slides.add_slide(blank_layout)
            _add_styled_header(slide, str(ch.get("title", f"Visualization {idx + 1}")), f"Chart Type: {ch.get('chart_type', 'bar')}")
            img_stream = io.BytesIO(base64.b64decode(b64))
            slide.shapes.add_picture(img_stream, Inches(1.8), Inches(1.55), width=Inches(9.6))

        # Action Items / Recommendations slide
        if action_items:
            slide = prs.slides.add_slide(blank_layout)
            _add_styled_header(slide, "Strategic Recommendations & Next Steps")
            box = slide.shapes.add_textbox(Inches(0.7), Inches(1.55), Inches(11.9), Inches(5.4))
            tf = box.text_frame
            tf.word_wrap = True
            for idx, item in enumerate(action_items[:8]):
                para = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
                para.text = f"{idx + 1}. {str(item)[:200]}"
                para.space_after = Pt(10)
                para.font.size = Pt(16)
                para.font.color.rgb = RGBColor(0x27, 0x24, 0x36)

        prs.save(str(path))

    @staticmethod
    def _write_xlsx(
        path: Path,
        title: str,
        markdown: str,
        action_items: list[str],
        tables: list[dict[str, Any]],
        analysis_results: dict[str, Any],
        evidence_items: list[dict[str, Any]],
    ) -> None:
        import openpyxl  # noqa: PLC0415
        from openpyxl.styles import Font, PatternFill  # noqa: PLC0415

        wb = openpyxl.Workbook()
        header_fill = PatternFill(start_color="4F3685", end_color="4F3685", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)

        # Sheet 1: Executive Summary & Sections
        ws_sum = wb.active
        ws_sum.title = "Executive_Summary"
        ws_sum.append(["Section", "Content"])
        for cell in ws_sum[1]:
            cell.fill = header_fill
            cell.font = header_font
        ws_sum.append(["Report Title", title])
        for sec_title, sec_lines in _parse_sections_from_markdown(markdown):
            ws_sum.append([sec_title, "\n".join(sec_lines[:40])[:30000]])
        ws_sum.column_dimensions["A"].width = 32
        ws_sum.column_dimensions["B"].width = 95

        # Sheet 2: Action Items
        ws_act = wb.create_sheet("Recommendations")
        ws_act.append(["Priority #", "Recommended Action"])
        for cell in ws_act[1]:
            cell.fill = header_fill
            cell.font = header_font
        for i, item in enumerate(action_items, start=1):
            ws_act.append([i, str(item)])
        ws_act.column_dimensions["B"].width = 90

        # Sheet 3: Dataset Statistics (if computed)
        if analysis_results:
            ws_stat = wb.create_sheet("Dataset_Statistics")
            ws_stat.append(["Dataset", "Column", "Count", "Mean", "Median", "Std", "Variance", "Min", "Q25", "Q75", "Max", "IQR_Outliers"])
            for cell in ws_stat[1]:
                cell.fill = header_fill
                cell.font = header_font
            for ds_name, ds_prof in analysis_results.items():
                if not isinstance(ds_prof, dict):
                    continue
                d_stats = ds_prof.get("descriptive_statistics") or {}
                outliers = ds_prof.get("outliers") or {}
                for col_name, st in d_stats.items():
                    ws_stat.append([
                        ds_name,
                        col_name,
                        st.get("count"),
                        st.get("mean"),
                        st.get("median"),
                        st.get("std"),
                        st.get("variance"),
                        st.get("min"),
                        st.get("q25"),
                        st.get("q75"),
                        st.get("max"),
                        outliers.get(col_name, {}).get("iqr_outlier_count", 0),
                    ])

        # Sheet 4: Structured Tables
        for idx, tbl in enumerate(tables[:5], start=1):
            headers = tbl.get("headers") or []
            rows = tbl.get("rows") or []
            if not headers:
                continue
            ws_t = wb.create_sheet(f"Table_{idx}")
            ws_t.append([str(h) for h in headers])
            for cell in ws_t[1]:
                cell.fill = header_fill
                cell.font = header_font
            for r in rows[:500]:
                ws_t.append([str(c) for c in r])

        # Sheet 5: Supporting Evidence
        if evidence_items:
            ws_ev = wb.create_sheet("Supporting_Evidence")
            ws_ev.append(["ID", "Source", "Confidence", "Claim", "Supporting Passage"])
            for cell in ws_ev[1]:
                cell.fill = header_fill
                cell.font = header_font
            for ev in evidence_items[:200]:
                if isinstance(ev, dict):
                    ws_ev.append([
                        str(ev.get("id", "")),
                        str(ev.get("source", "")),
                        float(ev.get("confidence", 0.0) or 0.0),
                        str(ev.get("claim", ""))[:500],
                        str(ev.get("supporting_passage", ""))[:2000],
                    ])

        wb.save(str(path))

    @staticmethod
    def _write_csv(
        path: Path,
        title: str,
        action_items: list[str],
        tables: list[dict[str, Any]],
        analysis_results: dict[str, Any],
        evidence_items: list[dict[str, Any]],
    ) -> None:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            # Prefer dataset statistics table when present; otherwise export findings & evidence
            if tables and tables[0].get("headers") and tables[0].get("rows"):
                writer.writerow(tables[0]["headers"])
                for r in tables[0]["rows"]:
                    writer.writerow(r)
                return

            writer.writerow(["Category", "Source_or_Index", "Confidence", "Detail"])
            for idx, act in enumerate(action_items, start=1):
                writer.writerow(["Recommendation", f"Action #{idx}", "1.00", str(act)])
            for ev in evidence_items[:200]:
                if isinstance(ev, dict):
                    writer.writerow([
                        "Evidence",
                        str(ev.get("source", "document")),
                        f"{float(ev.get('confidence', 0.0) or 0.0):.2f}",
                        str(ev.get("supporting_passage") or ev.get("claim") or "")[:500],
                    ])
            if not action_items and not evidence_items:
                writer.writerow(["Report", title, "1.00", "Generated by AegisAI AgentSwarm"])

    @staticmethod
    def _write_html(
        path: Path,
        title: str,
        markdown: str,
        charts: list[dict[str, Any]],
        tables: list[dict[str, Any]],
        execution_id: str,
    ) -> None:
        body_parts: list[str] = []
        in_list = False
        table_buf: list[str] = []

        def _flush_html_table() -> None:
            nonlocal table_buf
            if not table_buf:
                return
            p_rows: list[list[str]] = []
            for raw in table_buf:
                s_raw = raw.strip()
                if re.match(r"^\|\s*[-:]+[-| :]*\|$", s_raw):
                    continue
                cells = [re.sub(r"[*_`]", "", c).strip() for c in s_raw.strip("|").split("|")]
                if any(bool(c) for c in cells):
                    p_rows.append(cells)
            table_buf = []
            if not p_rows:
                return
            th = "".join(f"<th>{html_lib.escape(c)}</th>" for c in p_rows[0])
            trs = "".join(
                "<tr>" + "".join(f"<td>{html_lib.escape(c)}</td>" for c in r) + "</tr>"
                for r in p_rows[1:]
            )
            body_parts.append(f"<div class='table-responsive'><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>")

        for raw_line in (markdown or "").splitlines():
            s = raw_line.strip()
            if s.startswith("|") and s.endswith("|") and len(s) > 2:
                if in_list:
                    body_parts.append("</ul>")
                    in_list = False
                table_buf.append(s)
                continue
            else:
                _flush_html_table()

            if not s:
                if in_list:
                    body_parts.append("</ul>")
                    in_list = False
                continue
            m = re.match(r"^(#{1,4})\s+(.+?)\s*$", s)
            if m:
                if in_list:
                    body_parts.append("</ul>")
                    in_list = False
                lvl = len(m.group(1))
                txt = html_lib.escape(re.sub(r"\s+#+$", "", re.sub(r"[*_`]", "", m.group(2))))
                body_parts.append(f"<h{lvl}>{txt}</h{lvl}>")
                continue
            if s.startswith(("- ", "* ")):
                if not in_list:
                    body_parts.append("<ul>")
                    in_list = True
                clean_item = re.sub(r"[*_`#|]", "", s[2:]).strip()
                item_html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html_lib.escape(clean_item))
                body_parts.append(f"<li>{item_html}</li>")
                continue
            if in_list:
                body_parts.append("</ul>")
                in_list = False
            clean_p = re.sub(r"(\S)\s*\|\s*(\S)", r"\1 • \2", s)
            p_html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html_lib.escape(clean_p))
            body_parts.append(f"<p>{p_html}</p>")
        if in_list:
            body_parts.append("</ul>")
        _flush_html_table()

        charts_html = ""
        if charts:
            cards = []
            for ch in charts[:8]:
                b64 = ch.get("png_base64", "")
                if not b64:
                    continue
                c_title = html_lib.escape(str(ch.get("title", "Visualization")))
                cards.append(
                    f'<div class="chart-card"><h3>{c_title}</h3>'
                    f'<img src="data:image/png;base64,{b64}" alt="{c_title}" /></div>'
                )
            if cards:
                charts_html = "<h2>Data Visualizations</h2><div class='chart-grid'>" + "".join(cards) + "</div>"

        tables_html = ""
        if tables:
            t_blocks = []
            for tbl in tables[:5]:
                headers = tbl.get("headers") or []
                rows = tbl.get("rows") or []
                if not headers:
                    continue
                th = "".join(f"<th>{html_lib.escape(str(h))}</th>" for h in headers)
                trs = "".join(
                    "<tr>" + "".join(f"<td>{html_lib.escape(str(c))}</td>" for c in r) + "</tr>"
                    for r in rows[:100]
                )
                t_blocks.append(
                    f"<h3>{html_lib.escape(str(tbl.get('title', 'Table')))}</h3>"
                    f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"
                )
            tables_html = "".join(t_blocks)

        doc_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{html_lib.escape(title)}</title>
<style>
  body {{ font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; background: #FAFAF4; color: #272436; margin: 0; padding: 32px; line-height: 1.65; }}
  .container {{ max-width: 980px; margin: 0 auto; background: #FFFFFF; border: 1px solid #EAE7EF; border-radius: 16px; padding: 40px; box-shadow: 0 8px 28px rgba(39,36,54,0.05); }}
  .badge {{ display: inline-block; background: #FFF9CC; color: #4F3685; border: 1px solid #F3E66B; font-weight: 600; font-size: 12px; padding: 4px 12px; border-radius: 999px; margin-bottom: 12px; }}
  h1, h2, h3 {{ color: #4F3685; margin-top: 1.4em; }}
  h1 {{ font-size: 28px; border-bottom: 3px solid #F3E66B; padding-bottom: 12px; margin-top: 0; }}
  h2 {{ font-size: 20px; border-bottom: 1px solid #EAE7EF; padding-bottom: 6px; }}
  .meta {{ color: #777383; font-size: 13px; margin-bottom: 24px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13.5px; }}
  th {{ background: #F0EAFB; color: #4F3685; text-align: left; padding: 10px 12px; border: 1px solid #EAE7EF; }}
  td {{ padding: 8px 12px; border: 1px solid #EAE7EF; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-top: 16px; }}
  .chart-card {{ background: #FAFAF4; border: 1px solid #EAE7EF; border-radius: 12px; padding: 16px; }}
  .chart-card img {{ max-width: 100%; height: auto; border-radius: 8px; }}
</style>
</head>
<body>
  <div class="container">
    <span class="badge">AEGISAI AGENTSWARM DELIVERABLE</span>
    <h1>{html_lib.escape(title)}</h1>
    <div class="meta">Execution ID: {html_lib.escape(execution_id)} &bull; Generated {_utc_now_iso()[:19]} UTC</div>
    {"".join(body_parts)}
    {tables_html}
    {charts_html}
  </div>
</body>
</html>"""
        path.write_text(doc_html, encoding="utf-8")


_ARTIFACT_ENGINE: ArtifactEngine | None = None


def get_artifact_engine() -> ArtifactEngine:
    global _ARTIFACT_ENGINE
    if _ARTIFACT_ENGINE is None:
        _ARTIFACT_ENGINE = ArtifactEngine()
    return _ARTIFACT_ENGINE


def generate_execution_artifacts(
    *,
    execution_id: str,
    title: str,
    markdown: str,
    sections: dict[str, str] | None = None,
    action_items: list[str] | None = None,
    charts: list[Any] | None = None,
    tables: list[dict[str, Any]] | None = None,
    analysis_results: dict[str, Any] | None = None,
    evidence_items: list[dict[str, Any]] | None = None,
    requested_formats: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Top-level helper invoked by ``report_node`` and ``/api/artifacts/generate``."""
    return get_artifact_engine().generate_all(
        execution_id=execution_id or f"exec-{uuid.uuid4().hex[:10]}",
        title=title,
        markdown=markdown,
        sections=sections,
        action_items=action_items,
        charts=charts,
        tables=tables,
        analysis_results=analysis_results,
        evidence_items=evidence_items,
        requested_formats=requested_formats,
    )
