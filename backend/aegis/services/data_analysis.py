# backend/aegis/services/data_analysis.py
"""Data Analysis & Statistical Profiling Engine (Section 9).

Performs 100% real Python computation (pandas, numpy, scipy.stats, scikit-learn)
over tabular files (CSV, XLSX, XLS, JSON) — never LLM-estimated numbers:
- Dimensions, column data types, missing values, duplicates, unique counts
- Descriptive statistics: mean, median, mode, std, variance, min, Q25, Q75, max, IQR, skewness
- Pearson & Spearman correlation matrices + top correlated pairs with p-values
- Outlier & anomaly detection via IQR fences (1.5x IQR) and Z-score (|z| > 3.0)
- Linear trend & regression analysis (slope, intercept, R^2, p-value, direction)
- Grouped categorical comparisons
- Automatic generation of real charts via :class:`VisualizationEngine`
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression

from aegis.core.config import settings
from aegis.core.logging import get_logger
from aegis.services.visualization import get_visualization_engine

log = get_logger(__name__)

TABULAR_EXTENSIONS: set[str] = {".csv", ".xlsx", ".xls", ".json"}


def _safe_float(val: Any, digits: int = 4) -> float | None:
    try:
        f = float(val)
        if np.isfinite(f):
            return round(f, digits)
    except Exception:
        pass
    return None


class DataAnalysisEngine:
    """Deterministic statistical and analytical profiling engine for datasets."""

    def __init__(self) -> None:
        self._vis = get_visualization_engine()

    @staticmethod
    def load_dataframe(path: str | Path) -> pd.DataFrame:
        """Load CSV, XLSX, XLS, or JSON into a pandas DataFrame."""
        file_path = Path(str(path))
        ext = file_path.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(file_path, encoding_errors="replace")
        if ext == ".xlsx":
            return pd.read_excel(file_path, engine="openpyxl")
        if ext == ".xls":
            return pd.read_excel(file_path, engine="xlrd")
        if ext == ".json":
            raw = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(raw, list):
                return pd.DataFrame(raw)
            if isinstance(raw, dict):
                for v in raw.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        return pd.DataFrame(v)
                return pd.DataFrame([raw])
        raise ValueError(f"Unsupported dataset format for profiling: {ext}")

    def profile_dataset(
        self,
        path: str | Path,
        *,
        dataset_name: str | None = None,
        generate_charts: bool = True,
        artifact_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run complete statistical profiling and chart generation on *path*."""
        file_path = Path(str(path))
        raw_name = dataset_name or file_path.name
        # Clean UUID and numeric timestamp prefixes (e.g. 9eb51991-4870-5f52-b1db-c39a6165ce3b_customers.csv -> customers.csv)
        clean_name = re.sub(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_?", "", raw_name)
        clean_name = re.sub(r"^\d{10,}_?", "", clean_name).strip() or raw_name
        name = clean_name
        df = self.load_dataframe(file_path)

        # Coerce string columns that are predominantly numeric
        for col in df.columns:
            if df[col].dtype == object:
                cleaned = df[col].astype(str).str.replace(r"[$,%\s]", "", regex=True)
                converted = pd.to_numeric(cleaned, errors="coerce")
                non_null_orig = int(df[col].notna().sum())
                if non_null_orig > 0 and int(converted.notna().sum()) >= 0.8 * non_null_orig:
                    df[col] = converted

        row_count, col_count = int(df.shape[0]), int(df.shape[1])
        duplicate_rows = int(df.duplicated().sum()) if row_count > 0 else 0
        duplicate_pct = round((duplicate_rows / row_count) * 100.0, 2) if row_count > 0 else 0.0

        numeric_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns]
        datetime_cols: list[str] = []
        categorical_cols: list[str] = []
        for col in df.columns:
            c_str = str(col)
            if c_str in numeric_cols:
                continue
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(c_str)
            else:
                categorical_cols.append(c_str)

        # Missing & unique summary per column
        column_profiles: dict[str, Any] = {}
        completeness_labels: list[str] = []
        completeness_values: list[float] = []

        for col in df.columns:
            c_str = str(col)
            missing_cnt = int(df[col].isna().sum())
            missing_pct = round((missing_cnt / row_count) * 100.0, 2) if row_count > 0 else 0.0
            unique_cnt = int(df[col].nunique(dropna=True))
            completeness_labels.append(c_str)
            completeness_values.append(round(100.0 - missing_pct, 2))
            column_profiles[c_str] = {
                "dtype": str(df[col].dtype),
                "missing_count": missing_cnt,
                "missing_pct": missing_pct,
                "completeness_pct": round(100.0 - missing_pct, 2),
                "unique_count": unique_cnt,
            }

        # Descriptive statistics & outlier detection on numeric columns
        descriptive_stats: dict[str, dict[str, Any]] = {}
        outliers_summary: dict[str, dict[str, Any]] = {}

        for col in numeric_cols:
            s = df[col].dropna().astype(float)
            if s.empty:
                continue
            arr = s.to_numpy()
            q25, q50, q75 = np.percentile(arr, [25, 50, 75])
            iqr = float(q75 - q25)
            modes = s.mode()
            mode_val = _safe_float(modes.iloc[0]) if not modes.empty else None
            std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            var_val = float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0
            skew_val = float(sp_stats.skew(arr)) if arr.size >= 3 and std_val > 0 else 0.0

            descriptive_stats[col] = {
                "count": int(arr.size),
                "mean": _safe_float(np.mean(arr)),
                "median": _safe_float(q50),
                "mode": mode_val,
                "std": _safe_float(std_val),
                "variance": _safe_float(var_val),
                "min": _safe_float(np.min(arr)),
                "q25": _safe_float(q25),
                "q75": _safe_float(q75),
                "max": _safe_float(np.max(arr)),
                "iqr": _safe_float(iqr),
                "sum": _safe_float(np.sum(arr)),
                "skewness": _safe_float(skew_val),
            }

            # Outliers: IQR fence + Z-score
            lower_fence = q25 - 1.5 * iqr
            upper_fence = q75 + 1.5 * iqr
            iqr_mask = (s < lower_fence) | (s > upper_fence)
            if std_val > 1e-9 and arr.size >= 3:
                z_scores = np.abs((s - float(np.mean(arr))) / std_val)
                z_mask = z_scores > 3.0
            else:
                z_mask = pd.Series(False, index=s.index)

            outliers_summary[col] = {
                "iqr_outlier_count": int(iqr_mask.sum()),
                "zscore_outlier_count": int(z_mask.sum()),
                "lower_fence": _safe_float(lower_fence),
                "upper_fence": _safe_float(upper_fence),
                "sample_outlier_values": [
                    _safe_float(v) for v in s[iqr_mask].head(5).tolist()
                ],
            }

        # Categorical top-value distributions
        categorical_summary: dict[str, Any] = {}
        for col in categorical_cols[:8]:
            vc = df[col].dropna().astype(str).value_counts().head(8)
            categorical_summary[col] = {
                "top_values": {str(k): int(v) for k, v in vc.items()},
            }

        # Correlation analysis (Pearson & Spearman)
        correlations: dict[str, Any] = {"matrix_columns": [], "pearson_matrix": [], "top_pairs": []}
        if len(numeric_cols) >= 2:
            sub = df[numeric_cols[:10]].dropna(how="all")
            corr_df = sub.corr(method="pearson").fillna(0.0)
            correlations["matrix_columns"] = list(corr_df.columns)
            correlations["pearson_matrix"] = [
                [round(float(v), 4) for v in row] for row in corr_df.to_numpy()
            ]
            pairs: list[dict[str, Any]] = []
            cols_sub = list(corr_df.columns)
            for i in range(len(cols_sub)):
                for j in range(i + 1, len(cols_sub)):
                    c1, c2 = cols_sub[i], cols_sub[j]
                    valid = df[[c1, c2]].dropna()
                    if len(valid) >= 3:
                        try:
                            r_val, p_val = sp_stats.pearsonr(valid[c1].astype(float), valid[c2].astype(float))
                            if np.isfinite(r_val):
                                pairs.append({
                                    "feature_a": c1,
                                    "feature_b": c2,
                                    "pearson_r": round(float(r_val), 4),
                                    "p_value": round(float(p_val), 6),
                                })
                        except Exception:
                            pass
            pairs.sort(key=lambda item: abs(item["pearson_r"]), reverse=True)
            correlations["top_pairs"] = pairs[:8]

        # Trend & Linear Regression analysis
        trends: list[dict[str, Any]] = []
        for col in numeric_cols[:4]:
            s = df[col].dropna().astype(float)
            if len(s) >= 4 and float(s.std()) > 1e-9:
                xs = np.arange(len(s), dtype=float)
                ys = s.to_numpy(dtype=float)
                reg = sp_stats.linregress(xs, ys)
                # Verify with sklearn LinearRegression R^2
                lr = LinearRegression().fit(xs.reshape(-1, 1), ys)
                r2 = float(lr.score(xs.reshape(-1, 1), ys))
                direction = (
                    "increasing"
                    if reg.slope > 0 and reg.pvalue < 0.10
                    else "decreasing"
                    if reg.slope < 0 and reg.pvalue < 0.10
                    else "stable"
                )
                trends.append({
                    "column": col,
                    "slope_per_row": _safe_float(reg.slope, 6),
                    "intercept": _safe_float(reg.intercept, 4),
                    "r_squared": _safe_float(r2, 4),
                    "p_value": _safe_float(reg.pvalue, 6),
                    "direction": direction,
                })

        # Grouped aggregations
        id_tokens = {"id", "key", "index", "code", "row", "number", "postal"}
        continuous_numeric = [
            c for c in numeric_cols
            if not any(token in c.lower() for token in id_tokens) and df[c].nunique() > 5
        ]
        primary_num_cols = continuous_numeric if continuous_numeric else numeric_cols

        grouped_comparisons: list[dict[str, Any]] = []
        if primary_num_cols:
            num_col = primary_num_cols[0]
            candidate_cats = [c for c in categorical_cols if 2 <= int(df[c].nunique(dropna=True)) <= 25]
            if not candidate_cats:
                candidate_cats = [
                    c for c in df.columns
                    if 2 <= int(df[c].nunique(dropna=True)) <= 15 and c != num_col
                ]
            if candidate_cats:
                cat_col = candidate_cats[0]
                grp = (
                    df[[cat_col, num_col]]
                    .dropna()
                    .groupby(cat_col, observed=False)[num_col]
                    .agg(["count", "mean", "sum", "median"])
                    .sort_values("sum", ascending=False)
                    .head(10)
                )
                if not grp.empty:
                    grouped_comparisons.append({
                        "group_by": cat_col,
                        "metric": num_col,
                        "groups": [
                            {
                                "category": str(idx),
                                "count": int(row["count"]),
                                "mean": _safe_float(row["mean"]),
                                "sum": _safe_float(row["sum"]),
                                "median": _safe_float(row["median"]),
                            }
                            for idx, row in grp.iterrows()
                        ],
                    })

        # Generate real visualizations via VisualizationEngine
        charts: list[dict[str, Any]] = []
        if generate_charts:
            art_dir = Path(artifact_dir) if artifact_dir else None

            # 1. Data quality chart
            overall_hygiene = round(100.0 - (float(df.isna().sum().sum()) / max(1, row_count * col_count) * 100.0), 1)
            dq_chart = self._vis.render_chart(
                "data_quality",
                title=f"Column Data Completeness — {name}",
                labels=completeness_labels,
                values=completeness_values,
                save_path=(art_dir / "chart_data_quality.png") if art_dir else None,
                metadata={
                    "source": name,
                    "heading": "Column Data Completeness & Quality",
                    "explanation": f"Evaluates the percentage of populated, non-null records across all {col_count} columns in {name}. Identifies fields with missing data that require imputation or validation before downstream modeling.",
                    "key_metric": f"{overall_hygiene}% Overall Data Hygiene",
                    "takeaway": "Columns with 100% bars have zero null entries; shorter bars indicate missing values requiring attention.",
                },
            )
            if dq_chart:
                charts.append(dq_chart.model_dump())

            # 2. Categorical frequency chart for top categorical column(s)
            candidate_cats = [c for c in categorical_cols if 2 <= int(df[c].nunique(dropna=True)) <= 35]
            if not candidate_cats and categorical_cols:
                candidate_cats = [categorical_cols[0]]
            for idx_cat, best_cat in enumerate(candidate_cats[:2]):
                top_counts = df[best_cat].value_counts(dropna=True).head(10)
                cat_labels = [str(k) for k in top_counts.index]
                cat_values = [int(v) for v in top_counts.values]
                if cat_labels:
                    cat_pct = round((cat_values[0] / max(1, row_count)) * 100.0, 1)
                    cat_chart = self._vis.render_chart(
                        "bar",
                        title=f"Top Categories in {best_cat} — {name}",
                        labels=cat_labels,
                        values=cat_values,
                        xlabel=best_cat,
                        ylabel="Record Count",
                        save_path=(art_dir / f"chart_cat_{best_cat}.png") if art_dir else None,
                        metadata={
                            "source": name,
                            "heading": f"Category Distribution: {best_cat}",
                            "explanation": f"Illustrates the record counts for the top 10 most frequent categories in '{best_cat}' from {name}. Highlights dominant clusters and category variance.",
                            "key_metric": f"Leading segment '{cat_labels[0]}' represents {cat_values[0]:,} records ({cat_pct}% of total).",
                            "takeaway": f"Concentrate operations and segment targeting on top clusters ({', '.join(cat_labels[:3])}).",
                        },
                    )
                    if cat_chart:
                        charts.append(cat_chart.model_dump())

            # 3. Grouped bar comparison
            if grouped_comparisons:
                gc = grouped_comparisons[0]
                g_labels = [g["category"] for g in gc["groups"]]
                g_vals = [float(g["mean"] or 0.0) for g in gc["groups"]]
                bar_chart = self._vis.render_chart(
                    "bar",
                    title=f"Average {gc['metric']} by {gc['group_by']}",
                    labels=g_labels,
                    values=g_vals,
                    xlabel=gc["group_by"],
                    ylabel=f"Mean {gc['metric']}",
                    save_path=(art_dir / "chart_grouped_bar.png") if art_dir else None,
                    metadata={
                        "source": name,
                        "heading": f"Comparative Average: {gc['metric']} by {gc['group_by']}",
                        "explanation": f"Compares average '{gc['metric']}' values across different '{gc['group_by']}' segments in {name} to reveal performance disparities and high-value groupings.",
                        "key_metric": f"Top group '{g_labels[0]}' leads with average {g_vals[0]:.2f} {gc['metric']}.",
                        "takeaway": "Allocating resources to top-performing segments will maximize operational impact.",
                    },
                )
                if bar_chart:
                    charts.append(bar_chart.model_dump())

            # 4. Distribution chart on primary numeric column
            if primary_num_cols:
                primary_num = primary_num_cols[0]
                vals = df[primary_num].dropna().astype(float).tolist()
                st_data = descriptive_stats.get(primary_num, {})
                mean_val = float(st_data.get("mean") or (np.mean(vals) if vals else 0.0))
                med_val = float(st_data.get("median") or (np.median(vals) if vals else 0.0))
                std_val = float(st_data.get("std") or 0.0)
                skew_val = float(st_data.get("skewness") or 0.0)
                shape_text = "right-skewed" if skew_val > 0.5 else "left-skewed" if skew_val < -0.5 else "symmetrical"

                dist_chart = self._vis.render_chart(
                    "distribution",
                    title=f"Distribution Profile of {primary_num}",
                    values=vals,
                    xlabel=primary_num,
                    save_path=(art_dir / "chart_distribution.png") if art_dir else None,
                    metadata={
                        "source": name,
                        "heading": f"Statistical Distribution: {primary_num}",
                        "explanation": f"Shows the probability density curve and frequency histogram for '{primary_num}' across {len(vals):,} rows in {name}. Identifies central tendency, variance, and distributional skewness.",
                        "key_metric": f"Mean: {mean_val:.2f} | Median: {med_val:.2f} | Std Dev: {std_val:.2f}",
                        "takeaway": f"The data shows a {shape_text} pattern centered around {med_val:.2f}, indicating typical operating ranges.",
                    },
                )
                if dist_chart:
                    charts.append(dist_chart.model_dump())

                # 5. Boxplot across top numeric columns
                plot_cols = primary_num_cols[:6] if len(primary_num_cols) >= 2 else numeric_cols[:6]
                if len(plot_cols) >= 2:
                    box_series = {
                        c: df[c].dropna().astype(float).tolist()
                        for c in plot_cols
                    }
                    total_outliers = sum(outliers_summary.get(c, {}).get("iqr_outlier_count", 0) for c in plot_cols)
                    box_chart = self._vis.render_chart(
                        "boxplot",
                        title=f"Outlier & Quartile Spread — {name}",
                        series=box_series,
                        save_path=(art_dir / "chart_boxplot.png") if art_dir else None,
                        metadata={
                            "source": name,
                            "heading": "Interquartile Range & Outlier Spread",
                            "explanation": f"Plots the median (center line), interquartile range (IQR 25th-75th percentiles), and statistical anomalies exceeding 1.5× IQR across {len(plot_cols)} key numeric columns.",
                            "key_metric": f"{total_outliers} statistical outliers detected across analyzed features.",
                            "takeaway": "Audit extreme values outside the whiskers to prevent biased predictive modeling.",
                        },
                    )
                    if box_chart:
                        charts.append(box_chart.model_dump())

            # 6. Correlation matrix + Scatter on strongest correlated pair
            if correlations["pearson_matrix"] and len(correlations["matrix_columns"]) >= 2:
                corr_chart = self._vis.render_chart(
                    "correlation_matrix",
                    title=f"Correlation Matrix — {name}",
                    matrix=correlations["pearson_matrix"],
                    x_labels=correlations["matrix_columns"],
                    y_labels=correlations["matrix_columns"],
                    save_path=(art_dir / "chart_correlation.png") if art_dir else None,
                    metadata={
                        "source": name,
                        "heading": "Pearson Correlation Heatmap",
                        "explanation": f"Evaluates linear correlations (-1.0 to +1.0) among all {len(correlations['matrix_columns'])} numeric features in {name}. Identifies collinearity and strong co-moving variables.",
                        "key_metric": f"{len(correlations['matrix_columns'])}×{len(correlations['matrix_columns'])} Variable Dependency Grid",
                        "takeaway": "Strong positive (dark purple) and strong negative pairs indicate high predictive dependencies.",
                    },
                )
                if corr_chart:
                    charts.append(corr_chart.model_dump())

                if correlations["top_pairs"]:
                    non_id_pairs = [
                        p for p in correlations["top_pairs"]
                        if not any(token in p["feature_a"].lower() for token in id_tokens)
                        and not any(token in p["feature_b"].lower() for token in id_tokens)
                    ]
                    top_pair = non_id_pairs[0] if non_id_pairs else correlations["top_pairs"][0]
                    fa, fb = top_pair["feature_a"], top_pair["feature_b"]
                    r_score = float(top_pair["pearson_r"])
                    valid_pair = df[[fa, fb]].dropna()
                    if len(valid_pair) >= 3:
                        rel_strength = "Strong positive" if r_score > 0.6 else "Moderate positive" if r_score > 0.3 else "Strong inverse" if r_score < -0.6 else "Moderate inverse" if r_score < -0.3 else "Weak"
                        sc_chart = self._vis.render_chart(
                            "scatter",
                            title=f"{fa} vs {fb} (r = {r_score:.2f})",
                            x_values=valid_pair[fa].astype(float).tolist()[:300],
                            y_values=valid_pair[fb].astype(float).tolist()[:300],
                            xlabel=fa,
                            ylabel=fb,
                            save_path=(art_dir / "chart_scatter.png") if art_dir else None,
                            metadata={
                                "source": name,
                                "heading": f"Bivariate Scatter: {fa} vs {fb}",
                                "explanation": f"Plots individual data points of '{fa}' against '{fb}' with an ordinary least squares regression line. Confirms direct linear co-dependency.",
                                "key_metric": f"Pearson r = {r_score:.2f} ({rel_strength} correlation)",
                                "takeaway": f"Variations in {fa} explain meaningful variance in {fb}; useful for regression and forecasting.",
                            },
                        )
                        if sc_chart:
                            charts.append(sc_chart.model_dump())

        # Build structured summary tables for report & XLSX export
        tables: list[dict[str, Any]] = []
        if descriptive_stats:
            stat_headers = ["Column", "Count", "Mean", "Median", "Std Dev", "Min", "Max", "IQR Outliers"]
            stat_rows = [
                [
                    col,
                    str(st["count"]),
                    f"{st['mean']}" if st["mean"] is not None else "N/A",
                    f"{st['median']}" if st["median"] is not None else "N/A",
                    f"{st['std']}" if st["std"] is not None else "N/A",
                    f"{st['min']}" if st["min"] is not None else "N/A",
                    f"{st['max']}" if st["max"] is not None else "N/A",
                    str(outliers_summary.get(col, {}).get("iqr_outlier_count", 0)),
                ]
                for col, st in descriptive_stats.items()
            ]
            tables.append({
                "title": f"Descriptive Statistics — {name}",
                "headers": stat_headers,
                "rows": stat_rows,
                "metadata": {"source": name, "type": "descriptive_statistics"},
            })

        # Concise natural-language summary of exact computed numbers
        summary_lines = [
            f"Dataset '{name}': {row_count:,} rows × {col_count} columns "
            f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical; "
            f"{duplicate_rows} duplicate rows [{duplicate_pct}%])."
        ]
        for col, st in list(descriptive_stats.items())[:5]:
            out_cnt = outliers_summary.get(col, {}).get("iqr_outlier_count", 0)
            summary_lines.append(
                f"- {col}: mean={st['mean']}, median={st['median']}, std={st['std']}, "
                f"range=[{st['min']}, {st['max']}], IQR outliers={out_cnt}."
            )
        if correlations["top_pairs"]:
            tp = correlations["top_pairs"][0]
            summary_lines.append(
                f"- Strongest correlation: {tp['feature_a']} ↔ {tp['feature_b']} "
                f"(r={tp['pearson_r']}, p={tp['p_value']})."
            )
        for tr in trends[:2]:
            summary_lines.append(
                f"- Trend ({tr['column']}): {tr['direction']} (slope={tr['slope_per_row']}, R²={tr['r_squared']})."
            )

        return {
            "dataset_name": name,
            "row_count": row_count,
            "column_count": col_count,
            "duplicate_rows": duplicate_rows,
            "duplicate_pct": duplicate_pct,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
            "column_profiles": column_profiles,
            "descriptive_statistics": descriptive_stats,
            "outliers": outliers_summary,
            "categorical_summary": categorical_summary,
            "correlations": correlations,
            "trends": trends,
            "grouped_comparisons": grouped_comparisons,
            "summary": "\n".join(summary_lines),
            "visualizations": charts,
            "tables": tables,
        }


_ENGINE: DataAnalysisEngine | None = None


def get_data_analysis_engine() -> DataAnalysisEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = DataAnalysisEngine()
    return _ENGINE


def _lookup_registered_document_paths(doc_ids: list[str] | None = None) -> list[tuple[str, str, str]]:
    """Return list of ``(doc_id, filename, local_path)`` for persisted tabular files."""
    candidates: list[tuple[str, str, str]] = []
    registry_path = Path("./data/document_registry.json")
    entries: dict[str, dict[str, Any]] = {}

    if registry_path.exists():
        try:
            entries.update(json.loads(registry_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    try:
        from aegis.api.routes import _document_registry  # noqa: PLC0415

        entries.update(_document_registry)
    except Exception:
        pass

    target_set = {d for d in (doc_ids or []) if d}
    for d_id, info in entries.items():
        if target_set and d_id not in target_set:
            continue
        fname = str(info.get("filename", ""))
        local_path = str(info.get("file_path", ""))
        ext = Path(fname or local_path).suffix.lower()
        if ext in TABULAR_EXTENSIONS and local_path and Path(local_path).is_file():
            candidates.append((d_id, fname or Path(local_path).name, local_path))

    # Also scan AEGIS_DOCUMENT_ROOT if doc_ids matched a saved file prefix
    if not candidates and target_set:
        doc_root = Path(getattr(settings, "AEGIS_DOCUMENT_ROOT", "./data/documents"))
        if doc_root.is_dir():
            for d_id in target_set:
                for p in doc_root.glob(f"{d_id}_*"):
                    if p.suffix.lower() in TABULAR_EXTENSIONS and p.is_file():
                        candidates.append((d_id, p.name.split("_", 1)[-1], str(p)))

    return candidates


def analyze_registered_documents_for_state(state: dict[str, Any]) -> dict[str, Any] | None:
    """Automatically profile targeted CSV/XLSX/XLS/JSON documents during LangGraph execution."""
    doc_ids = list(state.get("document_ids") or [])
    if state.get("doc_id") and state["doc_id"] not in doc_ids:
        doc_ids.append(state["doc_id"])

    # Only scan ALL registered tabular docs if user explicitly asks for dataset/data analysis
    if not doc_ids and not state.get("requires_data_analysis"):
        return None

    matches = _lookup_registered_document_paths(doc_ids if doc_ids else None)
    if not matches:
        return None

    engine = get_data_analysis_engine()
    exec_id = state.get("execution_id") or ""
    art_root = Path(getattr(settings, "AEGIS_ARTIFACT_ROOT", "./data/artifacts"))
    art_dir = (art_root / exec_id) if exec_id else None

    combined_analysis: dict[str, Any] = {}
    all_charts: list[dict[str, Any]] = []
    all_tables: list[dict[str, Any]] = []
    summaries: list[str] = []

    for d_id, fname, fpath in matches[:3]:
        try:
            prof = engine.profile_dataset(
                fpath,
                dataset_name=fname,
                generate_charts=True,
                artifact_dir=art_dir,
            )
            combined_analysis[fname] = {
                k: v for k, v in prof.items() if k not in {"visualizations", "tables"}
            }
            all_charts.extend(prof.get("visualizations", []))
            all_tables.extend(prof.get("tables", []))
            summaries.append(prof.get("summary", ""))
        except Exception as exc:
            log.warning("Dataset profiling failed", doc_id=d_id, path=fpath, error=str(exc)[:200])

    if not combined_analysis:
        return None

    return {
        "analysis_results": combined_analysis,
        "visualizations": all_charts,
        "tables": all_tables,
        "summary": "\n\n".join(summaries),
    }

