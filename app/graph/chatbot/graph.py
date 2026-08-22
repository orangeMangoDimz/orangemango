"""Hierarchical planner graph with cumulative domain args and strict boundaries."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.config.const.chatbot import (
    MAX_AGENT_ACTIONS,
    NODE_CV_SUBAGENT,
    NODE_JOB_SUBAGENT,
    NODE_PARENT_PLANNER,
    NODE_RESPOND,
)
from app.graph.chatbot.container import ChatbotContainer
from app.graph.chatbot.cv_subagent import CvSubagentGraphBuilder
from app.graph.chatbot.job_subagent import JobSubagentGraphBuilder
from app.models.chat_model import ChatModel
from app.models.chatbot.state import (
    ConversationState,
    CvSubagentState,
    FinalResponseState,
    JobSubagentState,
    ParentPlannerInputState,
    PlanStepState,
    StudioInput,
)


class ChatbotGraphBuilder:
    """Execute parent and child plan arrays from timeline progress."""

    PARENT_NODES: frozenset[str] = frozenset(
        {NODE_CV_SUBAGENT, NODE_JOB_SUBAGENT, NODE_RESPOND}
    )

    def __init__(self, *, container: ChatbotContainer) -> None:
        self._container = container

    def _validate_parent_plan(self, state: ConversationState) -> ConversationState:
        plan: list[PlanStepState] = list(state.get("plan") or [])
        intent: dict[str, Any] = dict(state.get("intent") or {})
        args: dict[str, Any] = dict(state.get("args") or {})
        validation = dict(
            state.get("validation") or self._container.events.empty_validation()
        )
        nodes: list[str] = [str(step.get("node") or "") for step in plan]
        passed = []
        errors = []
        if plan and all(node in self.PARENT_NODES for node in nodes):
            passed.append(
                self._container.events.validation_entry(
                    "PARENT_PLAN_NODE_CHECK", "Parent plan nodes are valid."
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "PARENT_PLAN_NODE_CHECK", "Parent plan contains an invalid node."
                )
            )
        if len(nodes) == len(set(nodes)):
            passed.append(
                self._container.events.validation_entry(
                    "PARENT_PLAN_DUPLICATE_CHECK",
                    "Parent plan has no duplicate nodes.",
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "PARENT_PLAN_DUPLICATE_CHECK",
                    "Parent plan contains duplicate nodes.",
                )
            )
        if nodes and nodes[-1] == NODE_RESPOND:
            passed.append(
                self._container.events.validation_entry(
                    "PARENT_PLAN_RESPONSE_CHECK",
                    "Parent plan ends at the response node.",
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "PARENT_PLAN_RESPONSE_CHECK",
                    "Parent plan must end at the response node.",
                )
            )
        if len(nodes) <= MAX_AGENT_ACTIONS:
            passed.append(
                self._container.events.validation_entry(
                    "PARENT_PLAN_LIMIT_CHECK", "Parent plan is within the action limit."
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "PARENT_PLAN_LIMIT_CHECK", "Parent plan exceeds the action limit."
                )
            )
        known_cv_ids: set[str] = {
            str(item.get("id") or "")
            for item in (args.get("cv") or [])
            if isinstance(item, dict)
        }
        known_job_ids: set[str] = {
            str(item.get("id") or "")
            for item in (args.get("job") or [])
            if isinstance(item, dict)
        }
        cv_ids: set[str] = {str(value) for value in intent.get("cv_ids") or []}
        job_ids: set[str] = {str(value) for value in intent.get("job_ids") or []}
        if cv_ids.issubset(known_cv_ids) and job_ids.issubset(known_job_ids):
            passed.append(
                self._container.events.validation_entry(
                    "PARENT_INTENT_TARGET_CHECK", "Parent intent targets are valid."
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "PARENT_INTENT_TARGET_CHECK",
                    "Parent intent contains an unknown CV or job target.",
                )
            )
        validation = self._container.events.update_validation(
            validation,
            passed=passed,
            errors=errors,
        )
        return {**state, "validation": validation}

    async def _parent_planner(
        self,
        state: ParentPlannerInputState,
    ) -> dict[str, Any]:
        prepared: dict[str, Any] = self._container.events.prepare_parent_turn(
            state  # type: ignore[arg-type]
        )
        planning_state: ConversationState = {**state, **prepared}
        try:
            update: dict[
                str, Any
            ] = await self._container.hierarchical.parent_plan_node(planning_state)
            planned: ConversationState = self._validate_parent_plan(
                {**planning_state, **update}
            )
            failed: bool = bool((planned.get("validation") or {}).get("errors"))
            query: str = str((planned.get("intent") or {}).get("query") or "")
            return {
                **prepared,
                **update,
                "validation": dict(planned.get("validation") or {}),
                "timeline": self._container.events.append_event(
                    [],
                    node=NODE_PARENT_PLANNER,
                    status="failed" if failed else "success",
                    summary=(
                        "Parent planning failed validation."
                        if failed
                        else query or "Parent execution plan created."
                    ),
                ),
            }
        except Exception as exc:
            return {
                **prepared,
                "intent": {},
                "plan": [],
                "timeline": self._container.events.append_event(
                    [],
                    node=NODE_PARENT_PLANNER,
                    status="failed",
                    summary=f"Parent planning failed: {type(exc).__name__}.",
                ),
            }

    async def _respond(
        self,
        state: FinalResponseState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        update: dict[str, Any] = await self._container.response.respond_node(
            state,
            config=config,
        )
        failure: str = str(update.pop("failure", "") or "")
        return {
            **update,
            "response_projection": dict(state),
            "timeline": self._container.events.append_event(
                list(state.get("timeline") or []),
                node=NODE_RESPOND,
                status="failed" if update.get("response") is None else "success",
                summary=failure or "Response generated.",
            ),
        }

    def _respond_send(self, state: ConversationState) -> Send:
        return Send(NODE_RESPOND, self._container.events.final_projection(state))

    @staticmethod
    def _completed_nodes(state: ConversationState) -> set[str]:
        return {
            str(event.get("node") or "")
            for event in (state.get("timeline") or [])
            if isinstance(event, dict) and event.get("status") == "success"
        }

    def build(self, *, checkpointer: Any | None = None) -> Any:
        cv_graph: Any = CvSubagentGraphBuilder(container=self._container).build()
        job_graph: Any = JobSubagentGraphBuilder(container=self._container).build()

        async def run_cv_subagent(state: CvSubagentState) -> dict[str, Any]:
            child_output = await cv_graph.ainvoke(state)
            output: dict[str, Any] = self._container.events.strip_subagent_state(
                child_output
            )
            status: str = self._container.events.latest_status(child_output)
            output["timeline"] = self._container.events.append_event(
                list(output.get("timeline") or []),
                node=NODE_CV_SUBAGENT,
                status=status,
                summary=str(
                    (child_output.get("timeline") or [{}])[-1].get("summary")
                    or "CV subagent completed."
                ),
            )
            output["args"] = {"cv": (output.get("args") or {}).get("cv") or []}
            return output

        async def run_job_subagent(state: JobSubagentState) -> dict[str, Any]:
            child_output = await job_graph.ainvoke(state)
            output: dict[str, Any] = self._container.events.strip_subagent_state(
                child_output
            )
            status: str = self._container.events.latest_status(child_output)
            output["timeline"] = self._container.events.append_event(
                list(output.get("timeline") or []),
                node=NODE_JOB_SUBAGENT,
                status=status,
                summary=str(
                    (child_output.get("timeline") or [{}])[-1].get("summary")
                    or "Job subagent completed."
                ),
            )
            output["args"] = {
                "cv": (output.get("args") or {}).get("cv") or [],
                "job": (output.get("args") or {}).get("job") or [],
            }
            return output

        def next_parent_step(state: ConversationState) -> str | Send:
            timeline: list[dict[str, Any]] = [
                event
                for event in (state.get("timeline") or [])
                if isinstance(event, dict)
            ]
            if timeline and timeline[-1].get("status") == "failed":
                return self._respond_send(state)
            if timeline and timeline[-1].get("status") == "pending":
                return NODE_PARENT_PLANNER
            completed: set[str] = self._completed_nodes(state)
            for step in state.get("plan") or []:
                node: str = str(step.get("node") or "")
                if node in completed:
                    continue
                if node == NODE_CV_SUBAGENT:
                    return Send(
                        NODE_CV_SUBAGENT, self._container.events.cv_input(state, step)
                    )
                if node == NODE_JOB_SUBAGENT:
                    return Send(
                        NODE_JOB_SUBAGENT,
                        self._container.events.job_input(state, step),
                    )
                if node == NODE_RESPOND:
                    return self._respond_send(state)
            return self._respond_send(state)

        builder: StateGraph = StateGraph(ConversationState, input_schema=StudioInput)
        builder.add_node(
            NODE_PARENT_PLANNER,
            self._parent_planner,
            input_schema=ParentPlannerInputState,
        )
        builder.add_node(
            NODE_CV_SUBAGENT,
            run_cv_subagent,
            input_schema=CvSubagentState,
        )
        builder.add_node(
            NODE_JOB_SUBAGENT,
            run_job_subagent,
            input_schema=JobSubagentState,
        )
        builder.add_node(NODE_RESPOND, self._respond, input_schema=FinalResponseState)
        destinations: dict[str, str] = {
            NODE_PARENT_PLANNER: NODE_PARENT_PLANNER,
            NODE_CV_SUBAGENT: NODE_CV_SUBAGENT,
            NODE_JOB_SUBAGENT: NODE_JOB_SUBAGENT,
        }
        builder.add_edge(START, NODE_PARENT_PLANNER)
        builder.add_conditional_edges(
            NODE_PARENT_PLANNER, next_parent_step, destinations
        )
        builder.add_conditional_edges(NODE_CV_SUBAGENT, next_parent_step, destinations)
        builder.add_conditional_edges(NODE_JOB_SUBAGENT, next_parent_step, destinations)
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
