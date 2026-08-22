from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.config.const.chatbot import (
    NODE_CV_SUBAGENT,
    NODE_INGEST_INPUT,
    NODE_JOB_SUBAGENT,
    NODE_REQUEST_ROUTER,
    NODE_RESPOND,
    NODE_SUMMARIZE_CONVERSATION,
    NODE_VALIDATE_PLAN,
    NODE_WORKFLOW_PLANNER,
    ROUTE_COMPARE_CVS,
    ROUTE_EXTRACT_CV,
    ROUTE_EXTRACT_JOB,
    ROUTE_MATCH_JOBS,
    ROUTE_RESPOND,
    ROUTE_REVIEW_CV,
    ROUTE_SEARCH_JOBS,
)
from app.graph.chatbot.container import ChatbotContainer
from app.graph.chatbot.cv_subagent import CvSubagentGraphBuilder
from app.graph.chatbot.job_subagent import JobSubagentGraphBuilder
from app.models.chat_model import ChatModel
from app.models.chatbot.state import ConversationState, StudioInput


class ChatbotGraphBuilder:
    """Assemble the top-level chatbot graph from container-provided nodes."""

    def __init__(self, *, container: ChatbotContainer) -> None:
        self._container = container

    async def _ingest_input(self, state: ConversationState) -> dict[str, Any]:
        return self._container.ingest.ingest_input(state)

    async def _summarize_conversation(
        self,
        state: ConversationState,
    ) -> dict[str, Any]:
        return await self._container.memory.summarize_conversation_node(state)

    async def _request_router(self, state: ConversationState) -> dict[str, Any]:
        return await self._container.routing.request_router_node(state)

    async def _workflow_planner(self, state: ConversationState) -> dict[str, Any]:
        return await self._container.routing.workflow_planner_node(state)

    async def _validate_plan(self, state: ConversationState) -> dict[str, Any]:
        return await self._container.plan.validate_plan_node(state)

    async def _respond(
        self,
        state: ConversationState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        return await self._container.response.respond_node(state, config=config)

    def build(self, *, checkpointer: Any | None = None) -> Any:
        router = self._container.router
        cv_subagent: Any = CvSubagentGraphBuilder(container=self._container).build()
        job_subagent: Any = JobSubagentGraphBuilder(container=self._container).build()

        builder: StateGraph = StateGraph(
            ConversationState,
            input_schema=StudioInput,
        )
        builder.add_node(NODE_INGEST_INPUT, self._ingest_input)
        builder.add_node(NODE_SUMMARIZE_CONVERSATION, self._summarize_conversation)
        builder.add_node(NODE_REQUEST_ROUTER, self._request_router)
        builder.add_node(NODE_WORKFLOW_PLANNER, self._workflow_planner)
        builder.add_node(NODE_VALIDATE_PLAN, self._validate_plan)
        builder.add_node(NODE_CV_SUBAGENT, cv_subagent)
        builder.add_node(NODE_JOB_SUBAGENT, job_subagent)
        builder.add_node(NODE_RESPOND, self._respond)

        builder.add_edge(START, NODE_INGEST_INPUT)
        builder.add_edge(NODE_INGEST_INPUT, NODE_SUMMARIZE_CONVERSATION)
        builder.add_edge(NODE_SUMMARIZE_CONVERSATION, NODE_REQUEST_ROUTER)
        builder.add_edge(NODE_REQUEST_ROUTER, NODE_WORKFLOW_PLANNER)
        builder.add_edge(NODE_WORKFLOW_PLANNER, NODE_VALIDATE_PLAN)
        builder.add_conditional_edges(
            NODE_VALIDATE_PLAN,
            router.route_after_plan_validation,
            {
                ROUTE_EXTRACT_CV: NODE_CV_SUBAGENT,
                ROUTE_REVIEW_CV: NODE_CV_SUBAGENT,
                ROUTE_COMPARE_CVS: NODE_CV_SUBAGENT,
                ROUTE_EXTRACT_JOB: NODE_JOB_SUBAGENT,
                ROUTE_SEARCH_JOBS: NODE_JOB_SUBAGENT,
                ROUTE_MATCH_JOBS: NODE_JOB_SUBAGENT,
                ROUTE_RESPOND: NODE_RESPOND,
            },
        )
        builder.add_conditional_edges(
            NODE_CV_SUBAGENT,
            router.route_after_cv_subagent,
            {
                NODE_WORKFLOW_PLANNER: NODE_WORKFLOW_PLANNER,
                ROUTE_RESPOND: NODE_RESPOND,
            },
        )
        builder.add_conditional_edges(
            NODE_JOB_SUBAGENT,
            router.route_after_job_subagent,
            {
                NODE_WORKFLOW_PLANNER: NODE_WORKFLOW_PLANNER,
                ROUTE_RESPOND: NODE_RESPOND,
            },
        )
        builder.add_edge(NODE_RESPOND, END)

        return builder.compile(checkpointer=checkpointer)


def build_graph(
    *,
    checkpointer: Any | None = None,
    chat_model: ChatModel | None = None,
) -> Any:
    container: ChatbotContainer = ChatbotContainer(chat_model=chat_model)
    return ChatbotGraphBuilder(container=container).build(checkpointer=checkpointer)


graph: Any = build_graph()
