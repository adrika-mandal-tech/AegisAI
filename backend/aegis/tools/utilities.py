# backend/aegis/tools/utilities.py
"""Miscellaneous utility functions used across agents."""

from __future__ import annotations

import json
from pathlib import Path


def read_file(path: str | Path) -> str:
    """Return the text content of a file, or an error message."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: file not found — {path}"
    except Exception as exc:
        return f"Error reading {path}: {exc}"


def to_json(obj: object, *, indent: int = 2) -> str:
    """Pretty-print an object as JSON."""
    try:
        return json.dumps(obj, indent=indent, default=str)
    except (TypeError, ValueError) as exc:
        return f"Error serialising to JSON: {exc}"
