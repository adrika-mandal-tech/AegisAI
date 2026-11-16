# backend/aegis/services/__init__.py
"""AegisAI service layer."""

from __future__ import annotations

from aegis.services.pdf_report import generate_report_pdf

__all__ = ["generate_report_pdf"]
