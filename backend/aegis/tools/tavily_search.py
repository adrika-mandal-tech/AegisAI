# backend/aegis/tools/tavily_search.py
"""Wrapper around the Tavily search API.

Used by the Researcher agent to gather up-to-date web information.
"""

from __future__ import annotations

from typing import Any

from tavily import TavilyClient

from aegis.core.config import settings
from aegis.core.constants import TAVILY_DEFAULT_MAX_RESULTS
from aegis.core.logging import get_logger

log = get_logger(__name__)


class TavilySearch:
    """Thin wrapper over ``tavily.TavilyClient``."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or settings.AEGIS_TAVILY_API_KEY
        if not key:
            log.warning(
                "TAVILY_API_KEY not set — TavilySearch running in dry-run mode."
            )
            self._client = None
            self._dry_run = True
        else:
            self._client = TavilyClient(api_key=key)
            self._dry_run = False

    def search(self, query: str, max_results: int = TAVILY_DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:  # noqa: E501
        """Run a web search and return a list of result dicts.

        Each dict contains at least ``title``, ``url``, and ``content``.
        """
        if self._dry_run or not query or not query.strip():
            return []
        trimmed_query = query.strip()[:380]
        log.info("Tavily search", query=trimmed_query[:80], max_results=max_results)
        try:
            try:
                response = self._client.search(
                    query=trimmed_query,
                    max_results=max_results,
                    search_depth="basic",
                    timeout=6,
                )
            except TypeError:
                response = self._client.search(query=trimmed_query, max_results=max_results)
            results: list[dict[str, Any]] = response.get("results", [])
            return results
        except Exception as exc:
            log.warning("Tavily search failed or timed out", error=str(exc)[:160])
            return []
