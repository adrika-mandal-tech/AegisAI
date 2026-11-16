# backend/aegis/services/visualization.py
"""Reusable Visualization Engine (Section 10).

Generates clean, publication-grade charts in AegisAI's light Lemon-Yellow &
Purple design language solely from real computed numerical/tabular data:
- ``bar``
- ``line``
- ``pie``
- ``histogram``
- ``scatter``
- ``boxplot``
- ``heatmap``
- ``correlation_matrix``
- ``time_series``
- ``comparative``
- ``distribution``
- ``data_quality``
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aegis.core.logging import get_logger
from aegis.schemas.agent import ChartData

log = get_logger(__name__)

# Light Lemon-Yellow & Purple Executive Palette
PALETTE: list[str] = [
    "#7454B8",  # Primary purple
    "#E5C832",  # Rich readable lemon gold
    "#4F3685",  # Deep purple
    "#3B82F6",  # Crisp blue
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#EC4899",  # Rose
    "#6366F1",  # Indigo
]
BG_COLOR = "#FAFAF4"
SURFACE_COLOR = "#FFFFFF"
TEXT_PRIMARY = "#272436"
TEXT_SECONDARY = "#777383"
GRID_COLOR = "#EAE7EF"

SUPPORTED_CHART_TYPES: set[str] = {
    "bar",
    "line",
    "pie",
    "histogram",
    "scatter",
    "boxplot",
    "heatmap",
    "correlation_matrix",
    "time_series",
    "comparative",
    "distribution",
    "data_quality",
}


class VisualizationEngine:
    """Renders verified PNG charts (base64 + optional disk file) from real data."""

    @staticmethod
    def _apply_theme(fig: Any, ax: Any, title: str, xlabel: str = "", ylabel: str = "") -> None:
        fig.patch.set_facecolor(SURFACE_COLOR)
        ax.set_facecolor(BG_COLOR)
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_PRIMARY, pad=12)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9.5, color=TEXT_SECONDARY, labelpad=6)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9.5, color=TEXT_SECONDARY, labelpad=6)
        ax.tick_params(colors=TEXT_PRIMARY, labelsize=8.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID_COLOR)
        ax.grid(True, linestyle="--", linewidth=0.7, color=GRID_COLOR, alpha=0.85)
        ax.set_axisbelow(True)

    @staticmethod
    def _finalize(
        fig: Any,
        *,
        chart_type: str,
        title: str,
        labels: list[str],
        values: list[float],
        save_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChartData:
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        png_bytes = buf.getvalue()

        meta = dict(metadata or {})
        saved_file = ""
        if save_path is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(png_bytes)
            saved_file = str(out_path)
            meta["file_path"] = saved_file

        b64_str = base64.b64encode(png_bytes).decode("ascii")
        meta["library"] = "matplotlib"
        return ChartData(
            chart_type=chart_type,
            title=title,
            labels=[str(lbl)[:48] for lbl in labels],
            values=[round(float(v), 4) for v in values],
            image_data=b64_str,
            file_path=saved_file,
            metadata=meta,
        )

    def render_chart(
        self,
        chart_type: str,
        *,
        title: str,
        labels: list[str] | None = None,
        values: list[float] | None = None,
        x_values: list[float] | None = None,
        y_values: list[float] | None = None,
        series: dict[str, list[float]] | None = None,
        matrix: list[list[float]] | None = None,
        x_labels: list[str] | None = None,
        y_labels: list[str] | None = None,
        xlabel: str = "",
        ylabel: str = "",
        save_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChartData | None:
        """Render any of the 12 supported chart types from structured numerical inputs."""
        ctype = (chart_type or "bar").lower().strip()
        if ctype not in SUPPORTED_CHART_TYPES:
            ctype = "bar"

        try:
            if ctype == "bar":
                return self._render_bar(title, labels or [], values or [], xlabel, ylabel, save_path, metadata)
            if ctype in {"line", "time_series"}:
                return self._render_line(ctype, title, labels or [], values or [], series, xlabel, ylabel, save_path, metadata)
            if ctype == "pie":
                return self._render_pie(title, labels or [], values or [], save_path, metadata)
            if ctype in {"histogram", "distribution"}:
                return self._render_distribution(ctype, title, values or [], xlabel, ylabel, save_path, metadata)
            if ctype == "scatter":
                return self._render_scatter(title, x_values or [], y_values or values or [], xlabel, ylabel, save_path, metadata)
            if ctype == "boxplot":
                return self._render_boxplot(title, series or {"Values": values or []}, ylabel, save_path, metadata)
            if ctype in {"heatmap", "correlation_matrix"}:
                return self._render_heatmap(ctype, title, matrix or [], x_labels or labels or [], y_labels or labels or [], save_path, metadata)
            if ctype == "comparative":
                return self._render_comparative(title, labels or [], series or {}, xlabel, ylabel, save_path, metadata)
            if ctype == "data_quality":
                return self._render_data_quality(title, labels or [], values or [], save_path, metadata)
        except Exception as exc:
            log.warning("VisualizationEngine.render_chart failed", chart_type=ctype, error=str(exc)[:200])
        return None

    # ── Individual renderers ──────────────────────────────────────────────

    def _render_bar(
        self,
        title: str,
        labels: list[str],
        values: list[float],
        xlabel: str,
        ylabel: str,
        save_path: str | Path | None,
        metadata: dict[str, Any] | None,
    ) -> ChartData | None:
        if not labels or not values or len(labels) != len(values):
            return None
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        self._apply_theme(fig, ax, title, xlabel, ylabel or "Value")
        short_labels = [str(x)[:28] for x in labels[:16]]
        vals = [float(v) for v in values[:16]]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(vals))]
        bars = ax.bar(short_labels, vals, color=colors, edgecolor="#4F3685", linewidth=0.5, width=0.62)
        if len(short_labels) > 5:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        span = max(abs(v) for v in vals) if vals else 1.0
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + span * 0.015,
                f"{val:,.2f}" if abs(val) < 1000 else f"{val:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_PRIMARY,
            )
        return self._finalize(fig, chart_type="bar", title=title, labels=short_labels, values=vals, save_path=save_path, metadata=metadata)

    def _render_line(
        self,
        ctype: str,
        title: str,
        labels: list[str],
        values: list[float],
        series: dict[str, list[float]] | None,
        xlabel: str,
        ylabel: str,
        save_path: str | Path | None,
        metadata: dict[str, Any] | None,
    ) -> ChartData | None:
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        self._apply_theme(fig, ax, title, xlabel or ("Time / Index" if ctype == "time_series" else ""), ylabel or "Value")
        if series:
            first_vals: list[float] = []
            x_idx = range(len(labels)) if labels else None
            for idx, (name, s_vals) in enumerate(series.items()):
                if not s_vals:
                    continue
                if not first_vals:
                    first_vals = [float(v) for v in s_vals]
                xs = list(range(len(s_vals)))
                ax.plot(xs, s_vals, marker="o", markersize=4, linewidth=2.1, color=PALETTE[idx % len(PALETTE)], label=name)
            if labels and x_idx is not None and len(labels) <= 16:
                ax.set_xticks(list(range(len(labels))))
                ax.set_xticklabels([str(x)[:20] for x in labels], rotation=30, ha="right")
            ax.legend(frameon=True, facecolor=SURFACE_COLOR, edgecolor=GRID_COLOR, fontsize=8.5)
            return self._finalize(
                fig,
                chart_type=ctype,
                title=title,
                labels=labels or [str(i + 1) for i in range(len(first_vals))],
                values=first_vals,
                save_path=save_path,
                metadata=metadata,
            )

        if not values:
            plt.close(fig)
            return None
        vals = [float(v) for v in values[:80]]
        lbls = [str(x)[:20] for x in (labels[:80] if labels else [str(i + 1) for i in range(len(vals))])]
        xs = list(range(len(vals)))
        ax.plot(xs, vals, color="#7454B8", linewidth=2.3, marker="o", markersize=4, markerfacecolor="#F3E66B", markeredgecolor="#4F3685")
        ax.fill_between(xs, vals, alpha=0.12, color="#7454B8")
        if len(lbls) <= 16:
            ax.set_xticks(xs)
            ax.set_xticklabels(lbls, rotation=30, ha="right")
        else:
            step = max(1, len(lbls) // 8)
            ticks = xs[::step]
            ax.set_xticks(ticks)
            ax.set_xticklabels([lbls[i] for i in ticks], rotation=30, ha="right")
        return self._finalize(fig, chart_type=ctype, title=title, labels=lbls, values=vals, save_path=save_path, metadata=metadata)

    def _render_pie(
        self,
        title: str,
        labels: list[str],
        values: list[float],
        save_path: str | Path | None,
        metadata: dict[str, Any] | None,
    ) -> ChartData | None:
        pos_pairs = [(str(l)[:26], float(v)) for l, v in zip(labels, values) if float(v) > 0][:8]
        if not pos_pairs:
            return None
        lbls, vals = zip(*pos_pairs)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        fig.patch.set_facecolor(SURFACE_COLOR)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(vals))]
        wedges, texts, autotexts = ax.pie(
            vals,
            labels=lbls,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": SURFACE_COLOR, "linewidth": 1.5},
            textprops={"color": TEXT_PRIMARY, "fontsize": 8.5},
        )
        for at in autotexts:
            at.set_color("#FFFFFF")
            at.set_fontweight("bold")
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_PRIMARY, pad=12)
        return self._finalize(fig, chart_type="pie", title=title, labels=list(lbls), values=list(vals), save_path=save_path, metadata=metadata)

    def _render_distribution(
        self,
        ctype: str,
        title: str,
        values: list[float],
        xlabel: str,
        ylabel: str,
        save_path: str | Path | None,
        metadata: dict[str, Any] | None,
    ) -> ChartData | None:
        arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float)
        if arr.size < 2:
            return None
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        self._apply_theme(fig, ax, title, xlabel or "Value", ylabel or ("Density" if ctype == "distribution" else "Frequency"))
        bins = min(25, max(5, int(np.sqrt(arr.size))))
        density = ctype == "distribution"
        counts, edges, _ = ax.hist(
            arr,
            bins=bins,
            density=density,
            color="#7454B8",
            alpha=0.78,
            edgecolor=SURFACE_COLOR,
            linewidth=0.8,
        )
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        ax.axvline(mean_val, color="#E5C832", linestyle="--", linewidth=2.0, label=f"Mean: {mean_val:,.2f}")
        if density and std_val > 1e-9:
            xs = np.linspace(float(arr.min()), float(arr.max()), 120)
            pdf = (1.0 / (std_val * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((xs - mean_val) / std_val) ** 2)
            ax.plot(xs, pdf, color="#4F3685", linewidth=2.0, label="Normal Fit")
        ax.legend(frameon=True, facecolor=SURFACE_COLOR, edgecolor=GRID_COLOR, fontsize=8.5)
        bin_labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(counts))]
        return self._finalize(
            fig,
            chart_type=ctype,
            title=title,
            labels=bin_labels,
            values=[float(c) for c in counts],
            save_path=save_path,
            metadata=metadata,
        )

    def _render_scatter(
        self,
        title: str,
        x_values: list[float],
        y_values: list[float],
        xlabel: str,
        ylabel: str,
        save_path: str | Path | None,
        metadata: dict[str, Any] | None,
    ) -> ChartData | None:
        n = min(len(x_values), len(y_values))
        if n < 2:
            return None
        xs = np.asarray([float(v) for v in x_values[:n]], dtype=float)
        ys = np.asarray([float(v) for v in y_values[:n]], dtype=float)
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        self._apply_theme(fig, ax, title, xlabel or "X", ylabel or "Y")
        ax.scatter(xs, ys, color="#7454B8", edgecolors="#4F3685", alpha=0.8, s=42)
        if np.std(xs) > 1e-9:
            slope, intercept = np.polyfit(xs, ys, 1)
            line_x = np.linspace(float(xs.min()), float(xs.max()), 60)
            ax.plot(line_x, slope * line_x + intercept, color="#E5C832", linewidth=2.2, linestyle="--", label=f"Trend (slope={slope:.3f})")
            ax.legend(frameon=True, facecolor=SURFACE_COLOR, edgecolor=GRID_COLOR, fontsize=8.5)
        return self._finalize(
            fig,
            chart_type="scatter",
            title=title,
            labels=[f"x={x:.2f}" for x in xs[:25]],
            values=[float(y) for y in ys[:25]],
            save_path=save_path,
            metadata=metadata,
        )

    def _render_boxplot(
        self,
        title: str,
        series: dict[str, list[float]],
        ylabel: str,
        save_path: str | Path | None,
        metadata: dict[str, Any] | None,
    ) -> ChartData | None:
        clean_series = {
            str(k)[:24]: [float(v) for v in vals if np.isfinite(float(v))]
            for k, vals in series.items()
            if vals
        }
        clean_series = {k: v for k, v in clean_series.items() if len(v) >= 2}
        if not clean_series:
            return None
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        self._apply_theme(fig, ax, title, "", ylabel or "Distribution")
        names = list(clean_series.keys())[:10]
        data = [clean_series[k] for k in names]
        bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
        for idx, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE[idx % len(PALETTE)])
            patch.set_alpha(0.75)
            patch.set_edgecolor("#4F3685")
        for med in bp["medians"]:
            med.set_color("#272436")
            med.set_linewidth(2.0)
        if len(names) > 5:
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        medians = [float(np.median(d)) for d in data]
        return self._finalize(fig, chart_type="boxplot", title=title, labels=names, values=medians, save_path=save_path, metadata=metadata)

    def _render_heatmap(
        self,
        ctype: str,
        title: str,
        matrix: list[list[float]],
        x_labels: list[str],
        y_labels: list[str],
        save_path: str | Path | None,
        metadata: dict[str, Any] | None,
    ) -> ChartData | None:
        if not matrix or not matrix[0]:
            return None
        arr = np.asarray(matrix, dtype=float)
        rows, cols = arr.shape
        xl = [str(x)[:18] for x in (x_labels[:cols] if x_labels else [f"C{i+1}" for i in range(cols)])]
        yl = [str(y)[:18] for y in (y_labels[:rows] if y_labels else [f"R{i+1}" for i in range(rows)])]

        fig, ax = plt.subplots(figsize=(max(6.8, cols * 0.85), max(4.8, rows * 0.65)))
        fig.patch.set_facecolor(SURFACE_COLOR)
        vmin, vmax = (-1.0, 1.0) if ctype == "correlation_matrix" else (float(np.nanmin(arr)), float(np.nanmax(arr)))
        im = ax.imshow(arr, cmap="Purples" if ctype != "correlation_matrix" else "PuOr", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(xl)))
        ax.set_xticklabels(xl, rotation=35, ha="right", fontsize=8.5, color=TEXT_PRIMARY)
        ax.set_yticks(range(len(yl)))
        ax.set_yticklabels(yl, fontsize=8.5, color=TEXT_PRIMARY)
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_PRIMARY, pad=12)

        if rows <= 12 and cols <= 12:
            for r in range(rows):
                for c in range(cols):
                    val = arr[r, c]
                    if np.isfinite(val):
                        ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=8, color=TEXT_PRIMARY)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        flat_vals = [float(v) for v in arr[0]]
        return self._finalize(fig, chart_type=ctype, title=title, labels=xl, values=flat_vals, save_path=save_path, metadata=metadata)

    def _render_comparative(
        self,
        title: str,
        labels: list[str],
        series: dict[str, list[float]],
        xlabel: str,
        ylabel: str,
        save_path: str | Path | None,
        metadata: dict[str, Any] | None,
    ) -> ChartData | None:
        valid = {k: [float(v) for v in vals] for k, vals in series.items() if vals}
        if not labels or not valid:
            return None
        n_groups = min(len(labels), 12)
        lbls = [str(l)[:22] for l in labels[:n_groups]]
        s_names = list(valid.keys())[:5]
        x = np.arange(n_groups)
        width = 0.78 / max(1, len(s_names))

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        self._apply_theme(fig, ax, title, xlabel, ylabel or "Value")
        for idx, s_name in enumerate(s_names):
            vals = valid[s_name][:n_groups]
            if len(vals) < n_groups:
                vals = vals + [0.0] * (n_groups - len(vals))
            offset = (idx - (len(s_names) - 1) / 2) * width
            ax.bar(x + offset, vals, width=width * 0.92, label=s_name, color=PALETTE[idx % len(PALETTE)])
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, rotation=25, ha="right")
        ax.legend(frameon=True, facecolor=SURFACE_COLOR, edgecolor=GRID_COLOR, fontsize=8.5)
        first_series = valid[s_names[0]][:n_groups]
        return self._finalize(fig, chart_type="comparative", title=title, labels=lbls, values=first_series, save_path=save_path, metadata=metadata)

    def _render_data_quality(
        self,
        title: str,
        labels: list[str],
        completeness_pcts: list[float],
        save_path: str | Path | None,
        metadata: dict[str, Any] | None,
    ) -> ChartData | None:
        if not labels or not completeness_pcts or len(labels) != len(completeness_pcts):
            return None
        lbls = [str(l)[:26] for l in labels[:18]]
        comp = [max(0.0, min(100.0, float(v))) for v in completeness_pcts[:18]]
        missing = [100.0 - c for c in comp]

        fig, ax = plt.subplots(figsize=(8.2, max(4.2, len(lbls) * 0.34)))
        self._apply_theme(fig, ax, title, "Percentage of Rows (%)", "")
        y = np.arange(len(lbls))
        ax.barh(y, comp, color="#7454B8", height=0.6, label="Populated (%)")
        ax.barh(y, missing, left=comp, color="#F3E66B", edgecolor="#EAE7EF", height=0.6, label="Missing (%)")
        ax.set_yticks(y)
        ax.set_yticklabels(lbls)
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.legend(loc="lower right", frameon=True, facecolor=SURFACE_COLOR, edgecolor=GRID_COLOR, fontsize=8.5)
        return self._finalize(fig, chart_type="data_quality", title=title, labels=lbls, values=comp, save_path=save_path, metadata=metadata)


_VIS_ENGINE: VisualizationEngine | None = None


def get_visualization_engine() -> VisualizationEngine:
    global _VIS_ENGINE
    if _VIS_ENGINE is None:
        _VIS_ENGINE = VisualizationEngine()
    return _VIS_ENGINE
