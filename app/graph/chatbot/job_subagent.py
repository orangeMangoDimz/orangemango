"""Job subagent graph: search and extraction prerequisites plus matching."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.config.const.chatbot import (
    BRANCH_END,
    GRAPH_JOB_SUBAGENT,
    NODE_EXTRACT_PASTED_JOB,
    NODE_MATCH_JOBS,
    NODE_SCRAPE_JOBS,
    ROUTE_EXTRACT_JOB,
    ROUTE_MATCH_JOBS,
    ROUTE_SEARCH_JOBS,
)
from app.graph.chatbot.container import ChatbotContainer
from app.models.chatbot.state import ConversationState


class JobSubagentGraphBuilder:
    """Build the job workflow, including search/extract prerequisites."""

    def __init__(self, *, container: ChatbotContainer) -> None:
        self._container = container

    async def _search_node(self, state: ConversationState) -> dict[str, Any]:
        return self._container.execution.record_completed_action(
            state,
            ROUTE_SEARCH_JOBS,
            await self._container.job_workflow.scrape_jobs_with_mcp(state),
            emit_result=True,
        )

    async def _extract_node(self, state: ConversationState) -> dict[str, Any]:
        return self._container.execution.record_completed_action(
            state,
            ROUTE_EXTRACT_JOB,
            await self._container.job_workflow.extract_pasted_job(state),
            emit_result=True,
        )

    async def _match_node(self, state: ConversationState) -> dict[str, Any]:
        return self._container.execution.record_completed_action(
            state,
            ROUTE_MATCH_JOBS,
            await self._container.job_workflow.calculate_job_matches(state),
            emit_result=True,
        )

    def build(self) -> Any:
        router = self._container.job_router
        builder: StateGraph = StateGraph(ConversationState)
        builder.add_node(NODE_SCRAPE_JOBS, self._search_node)
        builder.add_node(NODE_EXTRACT_PASTED_JOB, self._extract_node)
        builder.add_node(NODE_MATCH_JOBS, self._match_node)

        builder.set_conditional_entry_point(
            router.route_into_job_subagent,
            {
                NODE_EXTRACT_PASTED_JOB: NODE_EXTRACT_PASTED_JOB,
                NODE_SCRAPE_JOBS: NODE_SCRAPE_JOBS,
                NODE_MATCH_JOBS: NODE_MATCH_JOBS,
                BRANCH_END: END,
            },
        )
        builder.add_conditional_edges(
            NODE_SCRAPE_JOBS,
            router.route_after_search_or_extract,
            {
                NODE_MATCH_JOBS: NODE_MATCH_JOBS,
                BRANCH_END: END,
            },
        )
        builder.add_conditional_edges(
            NODE_EXTRACT_PASTED_JOB,
            router.route_after_search_or_extract,
            {
                NODE_MATCH_JOBS: NODE_MATCH_JOBS,
                BRANCH_END: END,
            },
        )
        builder.add_edge(NODE_MATCH_JOBS, END)
        return builder.compile(name=GRAPH_JOB_SUBAGENT)
