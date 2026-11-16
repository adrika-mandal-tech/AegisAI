# backend/aegis/agents/tool_agent.py
"""Controlled Tool Agent — executes allowlisted tools safely (Sections 8, 9, 10, 11).

Allowlisted tools (NO unrestricted shell execution):
- ``tavily_search`` — web search via Tavily
- ``web_scrape`` — scrape an HTTP(S) URL
- ``load_document`` — extract normalized text from a local file path
- ``analyze_dataset`` — run real pandas/scipy statistical profiling on CSV/XLSX/XLS/JSON
- ``generate_chart`` — render any of the 12 VisualizationEngine chart types
- ``generate_artifact`` — generate PDF/DOCX/PPTX/XLSX/CSV/HTML/MD deliverables
- ``mcp_call`` — invoke a configured Model Context Protocol tool
- ``read_file`` — read a local text file
- ``to_json`` — parse/serialize JSON safely
"""

from __future__ import annotations

from typing import Any

from aegis.core.logging import get_logger
from aegis.llm.base import LLMProvider
from aegis.llm.factory import get_default_llm
from aegis.schemas.agent import AgentOutput
from aegis.tools.document_loader import DocumentLoader, get_document_loader
from aegis.tools.mcp_client import MCPClient
from aegis.tools.tavily_search import TavilySearch
from aegis.tools.utilities import read_file, to_json
from aegis.tools.web_scraper import WebScraper

log = get_logger(__name__)

ALLOWLISTED_TOOLS: tuple[str, ...] = (
    "tavily_search",
    "web_scrape",
    "load_document",
    "analyze_dataset",
    "generate_chart",
    "generate_artifact",
    "mcp_call",
    "read_file",
    "to_json",
)


class ToolAgent:
    """Executes allowlisted tool calls requested by upstream agents."""

    def __init__(
        self,
        llm: LLMProvider | None = None,
        search: TavilySearch | None = None,
        scraper: WebScraper | None = None,
        loader: DocumentLoader | None = None,
        mcp: MCPClient | None = None,
    ) -> None:
        self._llm = llm or get_default_llm()
        self._search = search or TavilySearch()
        self._scraper = scraper or WebScraper()
        self._loader = loader or get_document_loader()
        self._mcp = mcp or MCPClient()

    def run(self, tool: str, payload: dict[str, Any]) -> AgentOutput:
        """Execute *tool* with *payload* and return structured :class:`AgentOutput`."""
        log.info("ToolAgent.run", tool=tool, payload_keys=list(payload.keys()))

        try:
            if tool == "tavily_search":
                query = str(payload.get("query", ""))
                max_results = int(payload.get("max_results", 5))
                results = self._search.search(query, max_results=max_results)
                return AgentOutput(
                    success=True,
                    summary=f"Tavily search returned {len(results)} results",
                    details=to_json(results),
                )

            if tool == "web_scrape":
                url = str(payload.get("url", ""))
                if not url:
                    return AgentOutput(success=False, errors=["Missing 'url' in payload"])
                content = self._scraper.scrape(url)
                return AgentOutput(
                    success=True,
                    summary=f"Scraped {url}",
                    details=content[:4000],
                )

            if tool == "load_document":
                path = str(payload.get("path", ""))
                if not path:
                    return AgentOutput(success=False, errors=["Missing 'path' in payload"])
                loaded = self._loader.load(path)
                if isinstance(loaded, list):
                    text = "\n\n".join(getattr(d, "page_content", str(d)) for d in loaded)
                else:
                    text = str(loaded or "")
                return AgentOutput(
                    success=bool(text.strip()),
                    summary=f"Loaded document from {path} ({len(text)} chars)",
                    details=text[:4000],
                    errors=[] if text.strip() else [f"No extractable text from {path}"],
                )

            if tool == "analyze_dataset":
                path = str(payload.get("path", ""))
                if not path:
                    return AgentOutput(success=False, errors=["Missing 'path' in payload"])
                from aegis.services.data_analysis import get_data_analysis_engine  # noqa: PLC0415

                prof = get_data_analysis_engine().profile_dataset(path)
                return AgentOutput(
                    success=True,
                    summary=prof.get("summary", f"Profiled {path}"),
                    details=to_json({k: v for k, v in prof.items() if k != "visualizations"})[:6000],
                    metadata={"profile": prof},
                )

            if tool == "generate_chart":
                from aegis.services.visualization import get_visualization_engine  # noqa: PLC0415

                chart = get_visualization_engine().render_chart(
                    str(payload.get("chart_type", "bar")),
                    title=str(payload.get("title", "Chart")),
                    labels=payload.get("labels"),
                    values=payload.get("values"),
                    x_values=payload.get("x_values"),
                    y_values=payload.get("y_values"),
                    series=payload.get("series"),
                    matrix=payload.get("matrix"),
                )
                if chart is None:
                    return AgentOutput(success=False, errors=["Insufficient numerical data for chart"])
                return AgentOutput(
                    success=True,
                    summary=f"Rendered {chart.chart_type} chart: {chart.title}",
                    details=chart.title,
                    metadata={"chart": chart.model_dump()},
                )

            if tool == "generate_artifact":
                from aegis.services.artifact_generator import generate_execution_artifacts  # noqa: PLC0415

                artifacts = generate_execution_artifacts(
                    execution_id=str(payload.get("execution_id", "manual")),
                    title=str(payload.get("title", "AegisAI Deliverable")),
                    markdown=str(payload.get("markdown", "")),
                    requested_formats=payload.get("formats"),
                )
                return AgentOutput(
                    success=bool(artifacts),
                    summary=f"Generated {len(artifacts)} deliverable artifact(s)",
                    details=to_json(artifacts),
                    metadata={"artifacts": artifacts},
                )

            if tool == "mcp_call":
                mcp_tool = str(payload.get("tool_name", ""))
                mcp_args = payload.get("arguments") or {}
                res = self._mcp.call_tool(mcp_tool, mcp_args)
                ok = res.get("status") == "ok"
                return AgentOutput(
                    success=ok,
                    summary=f"MCP tool '{mcp_tool}' status={res.get('status')}",
                    details=to_json(res),
                    errors=[] if ok else [str(res.get("message") or res.get("error") or "MCP unavailable")],
                )

            if tool == "read_file":
                path = str(payload.get("path", ""))
                if not path:
                    return AgentOutput(success=False, errors=["Missing 'path' in payload"])
                content = read_file(path)
                return AgentOutput(
                    success=True,
                    summary=f"Read file {path}",
                    details=content[:4000],
                )

            if tool == "to_json":
                raw = str(payload.get("value", ""))
                data = to_json(raw)
                return AgentOutput(
                    success=True,
                    summary="Converted value to JSON",
                    details=to_json(data) if isinstance(data, (dict, list)) else str(data),
                )

            return AgentOutput(
                success=False,
                errors=[f"Unknown tool '{tool}'. Supported: {', '.join(ALLOWLISTED_TOOLS)}"],
            )
        except Exception as exc:
            log.exception("ToolAgent.run failed")
            return AgentOutput(success=False, errors=[str(exc)])
