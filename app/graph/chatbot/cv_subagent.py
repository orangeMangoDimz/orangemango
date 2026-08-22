"""CV subagent graph: extraction plus the review and comparison actions."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.config.const.chatbot import (
    BRANCH_END,
    GRAPH_CV_SUBAGENT,
    NODE_COMPARE_CVS,
    NODE_EXTRACT_CV,
    NODE_MISSING_CV,
    NODE_REVIEW_CV,
    ROUTE_COMPARE_CVS,
    ROUTE_EXTRACT_CV,
    ROUTE_REVIEW_CV,
)
from app.graph.chatbot.container import ChatbotContainer
from app.models.chatbot.state import ConversationState


class CvSubagentGraphBuilder:
    """Build the CV workflow, including extraction prerequisites."""

    def __init__(self, *, container: ChatbotContainer) -> None:
        self._container = container

    async def _extract_node(self, state: ConversationState) -> dict[str, Any]:
        return self._container.execution.record_completed_action(
            state,
            ROUTE_EXTRACT_CV,
            await self._container.cv_workflow.run_cv_subagent(state),
        )

    async def _review_node(self, state: ConversationState) -> dict[str, Any]:
        return self._container.execution.record_completed_action(
            state,
            ROUTE_REVIEW_CV,
            await self._container.cv_workflow.run_cv_review(state),
            emit_result=True,
        )

    async def _compare_node(self, state: ConversationState) -> dict[str, Any]:
        return self._container.execution.record_completed_action(
            state,
            ROUTE_COMPARE_CVS,
            await self._container.cv_workflow.run_cv_comparison(state),
            emit_result=True,
        )

    async def _missing_cv_node(self, state: ConversationState) -> dict[str, Any]:
        return await self._container.cv_workflow.handle_missing_cv(state)

    def build(self) -> Any:
        router = self._container.cv_router
        builder: StateGraph = StateGraph(ConversationState)
        builder.add_node(NODE_EXTRACT_CV, self._extract_node)
        builder.add_node(NODE_REVIEW_CV, self._review_node)
        builder.add_node(NODE_COMPARE_CVS, self._compare_node)
        builder.add_node(NODE_MISSING_CV, self._missing_cv_node)
        builder.set_conditional_entry_point(
            router.route_into_cv_subagent,
            {
                NODE_EXTRACT_CV: NODE_EXTRACT_CV,
                NODE_REVIEW_CV: NODE_REVIEW_CV,
                NODE_COMPARE_CVS: NODE_COMPARE_CVS,
                NODE_MISSING_CV: NODE_MISSING_CV,
            },
        )
        builder.add_conditional_edges(
            NODE_EXTRACT_CV,
            router.route_after_cv_extraction,
            {
                NODE_REVIEW_CV: NODE_REVIEW_CV,
                NODE_COMPARE_CVS: NODE_COMPARE_CVS,
                BRANCH_END: END,
            },
        )
        builder.add_edge(NODE_REVIEW_CV, END)
        builder.add_edge(NODE_COMPARE_CVS, END)
        builder.add_edge(NODE_MISSING_CV, END)
        return builder.compile(name=GRAPH_CV_SUBAGENT)
