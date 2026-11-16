# backend/aegis/tools/mcp_client.py
"""Model Context Protocol (MCP) Client Integration (Section 2 & 8).

Connects to HTTP JSON-RPC MCP tool servers when ``server_url`` /
``AEGIS_MCP_SERVER_URL`` is configured, and degrades cleanly to transparent
offline status reporting when no external MCP server is configured.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from aegis.core.logging import get_logger

log = get_logger(__name__)


class MCPClient:
    """Modular JSON-RPC client for Model Context Protocol (MCP) servers."""

    def __init__(self, server_url: str = "", api_key: str = "") -> None:
        self._server_url = (server_url or os.environ.get("AEGIS_MCP_SERVER_URL", "")).strip()
        self._api_key = (api_key or os.environ.get("AEGIS_MCP_API_KEY", "")).strip()
        log.info("MCPClient initialised", configured=bool(self._server_url))

    @property
    def is_configured(self) -> bool:
        return bool(self._server_url)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke *tool_name* on the configured MCP server via JSON-RPC 2.0."""
        if not self._server_url:
            return {
                "tool": tool_name,
                "arguments": arguments,
                "result": None,
                "status": "unconfigured",
                "message": "Set AEGIS_MCP_SERVER_URL to connect an external MCP server.",
            }

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(self._server_url, json=payload, headers=headers)
                resp.raise_for_status()
                body = resp.json()
            return {
                "tool": tool_name,
                "arguments": arguments,
                "result": body.get("result"),
                "error": body.get("error"),
                "status": "ok" if "error" not in body else "error",
            }
        except Exception as exc:
            log.warning("MCPClient.call_tool failed", tool=tool_name, error=str(exc)[:200])
            return {
                "tool": tool_name,
                "arguments": arguments,
                "result": None,
                "status": "error",
                "message": str(exc)[:250],
            }

    def list_tools(self) -> list[str]:
        """Query available tools from the configured MCP server."""
        if not self._server_url:
            return []
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(self._server_url, json=payload, headers=headers)
                resp.raise_for_status()
                tools = (resp.json().get("result") or {}).get("tools") or []
            return [str(t.get("name")) for t in tools if isinstance(t, dict) and t.get("name")]
        except Exception:
            return []
