# backend/aegis/cli.py
"""CLI utilities for AegisAI."""

from __future__ import annotations

import sys

from aegis.core.logging import setup_logging
from aegis.llm.client import GeminiClient


def smoke_gemini() -> int:
    """Run a tiny strict Gemini output contract as a smoke test.

    Returns ``0`` on success, ``1`` on failure.
    """
    setup_logging()
    client = GeminiClient()
    response = client.generate("Say 'AegisAI smoke test passed' in exactly 5 words.")
    print(f"Gemini response: {response}")
    if "AegisAI" in response and "smoke test passed" in response:
        print("Smoke test passed.")
        return 0
    print("Smoke test failed: unexpected response.")
    return 1


if __name__ == "__main__":
    sys.exit(smoke_gemini())
