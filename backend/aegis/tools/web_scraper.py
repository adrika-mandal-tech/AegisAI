# backend/aegis/tools/web_scraper.py
"""Simple web scraper that extracts readable text from a URL."""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup

from aegis.core.logging import get_logger

log = get_logger(__name__)

_USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)


class WebScraper:
    """Extract clean text content from a web page."""

    def __init__(self, user_agent: str = _USER_AGENT) -> None:
        self._headers = {"User-Agent": user_agent}

    def scrape(self, url: str) -> str:
        """Fetch *url* and return visible text (scripts/styles stripped)."""
        log.info("Scraping URL", url=url)
        try:
            resp = requests.get(url, headers=self._headers, timeout=15)
            resp.raise_for_status()
        except requests.RequestException:
            log.exception("Failed to fetch URL", url=url)
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Collapse repeated blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
