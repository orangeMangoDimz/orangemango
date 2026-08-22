"""Access to the job-scraping MCP server."""

from __future__ import annotations

from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config.const.chatbot import (
    JOB_SUBAGENT_MCP_NAME,
    JOB_SUBAGENT_MCP_TRANSPORT,
    JOB_SUBAGENT_MCP_URL,
    SCRAPE_JOBS_TOOL_NAME,
)


class JobScraperRepository:
    """Resolve and cache the ``scrape_jobs`` MCP tool."""

    def __init__(self, *, mcp_url: str = JOB_SUBAGENT_MCP_URL) -> None:
        self._client: MultiServerMCPClient = MultiServerMCPClient(
            {
                JOB_SUBAGENT_MCP_NAME: {
                    "transport": JOB_SUBAGENT_MCP_TRANSPORT,
                    "url": mcp_url,
                }
            }
        )
        self._tool: Any | None = None

    async def scrape_jobs_tool(self) -> Any:
        if self._tool is None:
            tools: list[Any] = await self._client.get_tools()
            tools_by_name: dict[str, Any] = {tool.name: tool for tool in tools}
            if SCRAPE_JOBS_TOOL_NAME not in tools_by_name:
                raise RuntimeError(
                    f"The MCP server did not expose {SCRAPE_JOBS_TOOL_NAME}. "
                    f"Available tools: {sorted(tools_by_name)}"
                )
            self._tool = tools_by_name[SCRAPE_JOBS_TOOL_NAME]
        return self._tool


JOB_SCRAPER_REPOSITORY: JobScraperRepository = JobScraperRepository()
