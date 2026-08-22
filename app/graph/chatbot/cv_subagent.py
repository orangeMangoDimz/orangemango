"""Isolated CV subagent with planner-only plans and strict action inputs."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.config.const.chatbot import (
    BRANCH_END,
    NODE_COMPARE_CVS,
    NODE_CV_PLANNER,
    NODE_EXTRACT_CV,
    NODE_REVIEW_CV,
)
from app.graph.chatbot.container import ChatbotContainer
from app.models.chatbot.state import (
    CvActionState,
    CvSubagentState,
    PlanStepState,
    TimelineEventState,
)


class CvSubagentGraphBuilder:
    """Plan and execute the CV nodes selected from a parent dependency."""

    ALLOWED_NODES: frozenset[str] = frozenset(
        {NODE_EXTRACT_CV, NODE_REVIEW_CV, NODE_COMPARE_CVS}
    )

    def __init__(self, *, container: ChatbotContainer) -> None:
        self._container = container

    def _validate_plan(self, state: CvSubagentState) -> CvSubagentState:
        plan: list[PlanStepState] = list(state.get("plan") or [])
        args: dict[str, Any] = dict(state.get("args") or {})
        cvs: list[dict[str, Any]] = [
            item for item in (args.get("cv") or []) if isinstance(item, dict)
        ]
        nodes: list[str] = [str(step.get("node") or "") for step in plan]
        validation = dict(
            state.get("validation") or self._container.events.empty_validation()
        )
        passed = []
        errors = []
        if plan and all(node in self.ALLOWED_NODES for node in nodes):
            passed.append(
                self._container.events.validation_entry(
                    "CV_PLAN_NODE_CHECK", "CV plan nodes are valid."
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "CV_PLAN_NODE_CHECK", "CV plan contains an invalid node."
                )
            )
        if len(nodes) == len(set(nodes)):
            passed.append(
                self._container.events.validation_entry(
                    "CV_PLAN_DUPLICATE_CHECK", "CV plan has no duplicate nodes."
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "CV_PLAN_DUPLICATE_CHECK", "CV plan contains duplicate nodes."
                )
            )
        known_ids: set[str] = {str(item.get("id") or "") for item in cvs}
        cv_ids: list[str] = [
            str(value) for value in (args.get("cv_ids") or []) if str(value)
        ]
        if not cv_ids:
            cv_ids = [value for value in known_ids if value]
            args["cv_ids"] = cv_ids
        if cv_ids and set(cv_ids).issubset(known_ids):
            passed.append(
                self._container.events.validation_entry(
                    "CV_TARGET_CHECK", "CV targets are valid."
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "CV_TARGET_CHECK", "CV targets are missing or invalid."
                )
            )
        selected: list[dict[str, Any]] = [
            item for item in cvs if str(item.get("id") or "") in cv_ids
        ]
        target_valid: bool = True
        target_message: str = "CV action target counts are valid."
        if NODE_REVIEW_CV in nodes and len(selected) != 1:
            target_valid = False
            target_message = "CV review requires exactly one target CV."
        if NODE_COMPARE_CVS in nodes and len(selected) < 2:
            target_valid = False
            target_message = "CV comparison requires at least two target CVs."
        if any(not str(item.get("content") or "").strip() for item in selected):
            target_valid = False
            target_message = "Every selected CV requires content."
        entry = self._container.events.validation_entry(
            "CV_ACTION_TARGET_CHECK", target_message
        )
        (passed if target_valid else errors).append(entry)
        need_to_extract: int = sum(
            1
            for item in selected
            if str(item.get("content") or "").strip() and not item.get("features")
        )
        args["need_to_extract"] = need_to_extract
        requires_features: bool = bool(
            {NODE_REVIEW_CV, NODE_COMPARE_CVS}.intersection(nodes)
        )
        if not requires_features or not need_to_extract or NODE_EXTRACT_CV in nodes:
            passed.append(
                self._container.events.validation_entry(
                    "CV_EXTRACTION_PLAN_CHECK",
                    "CV extraction prerequisites are represented in the plan.",
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "CV_EXTRACTION_PLAN_CHECK",
                    "CV extraction is required before the requested CV action.",
                )
            )
        validation = self._container.events.update_validation(
            validation,
            passed=passed,
            errors=errors,
        )
        return {**state, "args": args, "validation": validation}

    async def _plan(self, state: CvSubagentState) -> dict[str, Any]:
        timeline = list(state.get("timeline") or [])
        try:
            update: dict[str, Any] = await self._container.hierarchical.cv_plan_node(
                state
            )
            planned: CvSubagentState = self._validate_plan({**state, **update})
            failed: bool = bool((planned.get("validation") or {}).get("errors"))
            return {
                **update,
                "args": dict(planned.get("args") or {}),
                "validation": dict(planned.get("validation") or {}),
                "timeline": self._container.events.append_event(
                    timeline,
                    node=NODE_CV_PLANNER,
                    status="failed" if failed else "success",
                    summary=(
                        "CV planning failed validation."
                        if failed
                        else "CV execution plan created."
                    ),
                ),
            }
        except Exception as exc:
            return {
                "plan": [],
                "timeline": self._container.events.append_event(
                    timeline,
                    node=NODE_CV_PLANNER,
                    status="failed",
                    summary=f"CV planning failed: {type(exc).__name__}.",
                ),
            }

    def _conversation_state(self, state: CvActionState) -> dict[str, Any]:
        args: dict[str, Any] = dict(state.get("args") or {})
        cv_ids: list[str] = list(args.get("cv_ids") or [])
        return {
            "cv": {
                "documents": [
                    self._container.events.cv_document(item)
                    for item in (args.get("cv") or [])
                    if isinstance(item, dict)
                ],
                "needs_extraction": bool(args.get("need_to_extract")),
            },
            "routing": {
                "request": {
                    "review": {
                        "target_role": args.get("target_role"),
                        "mode": args.get("review_mode") or "general",
                        "focus": args.get("review_focus"),
                    }
                },
                "targets": {
                    "cv": {
                        "scope": "one" if len(cv_ids) == 1 else "all",
                        "ids": cv_ids,
                    }
                },
            },
            "errors": [],
        }

    def _updated_args(
        self,
        state: CvActionState,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        args: dict[str, Any] = dict(state.get("args") or {})
        cv_update: dict[str, Any] = (
            dict(update.get("cv")) if isinstance(update.get("cv"), dict) else {}
        )
        documents: list[dict[str, Any]] = [
            self._container.events.cv_item(item)
            for item in (cv_update.get("documents") or [])
            if isinstance(item, dict)
        ]
        if documents:
            args["cv"] = documents
        args["need_to_extract"] = sum(
            1
            for item in (args.get("cv") or [])
            if isinstance(item, dict)
            and str(item.get("content") or "").strip()
            and not item.get("features")
        )
        return args

    async def _extract(self, state: CvActionState) -> dict[str, Any]:
        update: dict[str, Any] = await self._container.cv_workflow.run_cv_subagent(
            self._conversation_state(state)  # type: ignore[arg-type]
        )
        failures: list[str] = list(update.get("errors") or [])
        return {
            "args": self._updated_args(state, update),
            "timeline": self._container.events.append_event(
                list(state.get("timeline") or []),
                node=NODE_EXTRACT_CV,
                status="failed" if failures else "success",
                summary=failures[-1] if failures else "CV extraction completed.",
            ),
        }

    async def _review(self, state: CvActionState) -> dict[str, Any]:
        update: dict[str, Any] = await self._container.cv_workflow.run_cv_review(
            self._conversation_state(state)  # type: ignore[arg-type]
        )
        failures: list[str] = list(update.get("errors") or [])
        return {
            "args": self._updated_args(state, update),
            "timeline": self._container.events.append_event(
                list(state.get("timeline") or []),
                node=NODE_REVIEW_CV,
                status="failed" if failures else "success",
                summary=failures[-1] if failures else "CV review completed.",
            ),
        }

    async def _compare(self, state: CvActionState) -> dict[str, Any]:
        update: dict[str, Any] = await self._container.cv_workflow.run_cv_comparison(
            self._conversation_state(state)  # type: ignore[arg-type]
        )
        failures: list[str] = list(update.get("errors") or [])
        comparison: Any = (update.get("cv") or {}).get("comparison")
        args: dict[str, Any] | None = None
        if isinstance(comparison, dict):
            args = {
                "cv_ids": list((state.get("args") or {}).get("cv_ids") or []),
                "comparison": comparison,
            }
        return {
            "timeline": self._container.events.append_event(
                list(state.get("timeline") or []),
                node=NODE_COMPARE_CVS,
                status=(
                    "failed"
                    if failures or not isinstance(comparison, dict)
                    else "success"
                ),
                summary=failures[-1] if failures else "CV comparison completed.",
                args=args,
            )
        }

    @staticmethod
    def _next_node(state: CvSubagentState) -> str:
        if (state.get("validation") or {}).get("errors"):
            return BRANCH_END
        timeline: list[TimelineEventState] = list(state.get("timeline") or [])
        completed: set[str] = {
            str(event.get("node") or "")
            for event in timeline
            if event.get("status") == "success"
        }
        if timeline and timeline[-1].get("status") == "failed":
            return BRANCH_END
        for step in state.get("plan") or []:
            node: str = str(step.get("node") or "")
            if node not in completed:
                return node
        return BRANCH_END

    def build(self) -> Any:
        builder: StateGraph = StateGraph(CvSubagentState)
        builder.add_node(NODE_CV_PLANNER, self._plan)
        builder.add_node(NODE_EXTRACT_CV, self._extract, input_schema=CvActionState)
        builder.add_node(NODE_REVIEW_CV, self._review, input_schema=CvActionState)
        builder.add_node(NODE_COMPARE_CVS, self._compare, input_schema=CvActionState)
        builder.add_edge(START, NODE_CV_PLANNER)
        destinations: dict[str, str] = {
            NODE_EXTRACT_CV: NODE_EXTRACT_CV,
            NODE_REVIEW_CV: NODE_REVIEW_CV,
            NODE_COMPARE_CVS: NODE_COMPARE_CVS,
            BRANCH_END: END,
        }
        builder.add_conditional_edges(NODE_CV_PLANNER, self._next_node, destinations)
        builder.add_conditional_edges(NODE_EXTRACT_CV, self._next_node, destinations)
        builder.add_conditional_edges(NODE_REVIEW_CV, self._next_node, destinations)
        builder.add_conditional_edges(NODE_COMPARE_CVS, self._next_node, destinations)
        return builder.compile(name="cv_subagent")
