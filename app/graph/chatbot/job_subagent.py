"""Isolated job subagent with planner-only plans and strict action inputs."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.config.const.chatbot import (
    BRANCH_END,
    NODE_EXTRACT_PASTED_JOB,
    NODE_JOB_PLANNER,
    NODE_MATCH_JOBS,
    NODE_SCRAPE_JOBS,
)
from app.graph.chatbot.container import ChatbotContainer
from app.models.chatbot.state import (
    JobActionState,
    JobSubagentState,
    PlanStepState,
    TimelineEventState,
)


class JobSubagentGraphBuilder:
    """Plan and execute job nodes selected from a parent dependency."""

    ALLOWED_NODES: frozenset[str] = frozenset(
        {NODE_SCRAPE_JOBS, NODE_EXTRACT_PASTED_JOB, NODE_MATCH_JOBS}
    )

    def __init__(self, *, container: ChatbotContainer) -> None:
        self._container = container

    def _validate_plan(self, state: JobSubagentState) -> JobSubagentState:
        plan: list[PlanStepState] = list(state.get("plan") or [])
        args: dict[str, Any] = dict(state.get("args") or {})
        nodes: list[str] = [str(step.get("node") or "") for step in plan]
        validation = dict(
            state.get("validation") or self._container.events.empty_validation()
        )
        passed = []
        errors = []
        if plan and all(node in self.ALLOWED_NODES for node in nodes):
            passed.append(
                self._container.events.validation_entry(
                    "JOB_PLAN_NODE_CHECK", "Job plan nodes are valid."
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "JOB_PLAN_NODE_CHECK", "Job plan contains an invalid node."
                )
            )
        if len(nodes) == len(set(nodes)):
            passed.append(
                self._container.events.validation_entry(
                    "JOB_PLAN_DUPLICATE_CHECK", "Job plan has no duplicate nodes."
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "JOB_PLAN_DUPLICATE_CHECK", "Job plan contains duplicate nodes."
                )
            )
        cv_ids: list[str] = [
            str(value) for value in (args.get("cv_ids") or []) if str(value)
        ]
        job_ids: list[str] = [
            str(value) for value in (args.get("job_ids") or []) if str(value)
        ]
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
        if not cv_ids:
            cv_ids = [value for value in known_cv_ids if value]
            args["cv_ids"] = cv_ids
        if not job_ids:
            job_ids = [value for value in known_job_ids if value]
            args["job_ids"] = job_ids
        if set(cv_ids).issubset(known_cv_ids) and set(job_ids).issubset(known_job_ids):
            passed.append(
                self._container.events.validation_entry(
                    "JOB_TARGET_CHECK", "Job planner targets are valid."
                )
            )
        else:
            errors.append(
                self._container.events.validation_entry(
                    "JOB_TARGET_CHECK", "Job planner targets are invalid."
                )
            )
        if NODE_SCRAPE_JOBS in nodes:
            search: dict[str, Any] = dict(args.get("search") or {})
            keywords: list[str] = [
                str(value).strip()
                for value in (search.get("keywords") or [])
                if str(value).strip()
            ]
            if not keywords:
                for cv in args.get("cv") or []:
                    if not isinstance(cv, dict) or not isinstance(
                        cv.get("features"), dict
                    ):
                        continue
                    features: dict[str, Any] = cv["features"]
                    keywords = [
                        str(value).strip()
                        for value in (
                            features.get("role_tags")
                            or features.get("skill_names")
                            or []
                        )
                        if str(value).strip()
                    ][:5]
                    if keywords:
                        break
            search["keywords"] = keywords
            args["search"] = search
            if keywords:
                passed.append(
                    self._container.events.validation_entry(
                        "JOB_SEARCH_INPUT_CHECK", "Job search input is valid."
                    )
                )
            else:
                errors.append(
                    self._container.events.validation_entry(
                        "JOB_SEARCH_INPUT_CHECK", "Job search keywords are missing."
                    )
                )
        if NODE_EXTRACT_PASTED_JOB in nodes:
            if str(args.get("pasted_content") or "").strip():
                passed.append(
                    self._container.events.validation_entry(
                        "JOB_CONTENT_CHECK", "Pasted job content is valid."
                    )
                )
            else:
                errors.append(
                    self._container.events.validation_entry(
                        "JOB_CONTENT_CHECK", "Pasted job content is missing."
                    )
                )
        validation = self._container.events.update_validation(
            validation,
            passed=passed,
            errors=errors,
        )
        return {**state, "args": args, "validation": validation}

    async def _plan(self, state: JobSubagentState) -> dict[str, Any]:
        timeline = list(state.get("timeline") or [])
        try:
            update: dict[str, Any] = await self._container.hierarchical.job_plan_node(
                state
            )
            planned: JobSubagentState = self._validate_plan({**state, **update})
            validation_errors: bool = bool(
                (planned.get("validation") or {}).get("errors")
            )
            needs_cv: bool = any(
                step.get("node") == NODE_MATCH_JOBS
                for step in planned.get("plan") or []
            ) and not any(
                isinstance(item, dict) and item.get("features")
                for item in (planned.get("args") or {}).get("cv") or []
            )
            return {
                **update,
                "args": dict(planned.get("args") or {}),
                "validation": dict(planned.get("validation") or {}),
                "timeline": self._container.events.append_event(
                    timeline,
                    node=NODE_JOB_PLANNER,
                    status=(
                        "failed"
                        if validation_errors
                        else "pending"
                        if needs_cv
                        else "success"
                    ),
                    summary=(
                        "Job planning failed validation."
                        if validation_errors
                        else (
                            "CV preparation is required before job matching."
                            if needs_cv
                            else "Job execution plan created."
                        )
                    ),
                ),
            }
        except Exception as exc:
            return {
                "plan": [],
                "timeline": self._container.events.append_event(
                    timeline,
                    node=NODE_JOB_PLANNER,
                    status="failed",
                    summary=f"Job planning failed: {type(exc).__name__}.",
                ),
            }

    def _conversation_state(self, state: JobActionState) -> dict[str, Any]:
        args: dict[str, Any] = dict(state.get("args") or {})
        cv_ids: list[str] = list(args.get("cv_ids") or [])
        job_ids: list[str] = list(args.get("job_ids") or [])
        return {
            "cv": {
                "documents": [
                    self._container.events.cv_document(item)
                    for item in (args.get("cv") or [])
                    if isinstance(item, dict)
                ]
            },
            "jobs": {
                "results": [
                    self._container.events.job_result(item)
                    for item in (args.get("job") or [])
                    if isinstance(item, dict)
                ],
                "scrape_request": dict(args.get("search") or {}),
                "scrape_total": int(args.get("scrape_total") or 0),
                "scrape_truncated": bool(args.get("scrape_truncated")),
                "active_job_keys": list(args.get("active_job_keys") or []),
                "matches": list(args.get("matches") or []),
            },
            "routing": {
                "request": {
                    "job": {
                        "task": args.get("source") or "none",
                        "response": args.get("response") or "none",
                        "source": args.get("source") or "none",
                        "input": args.get("pasted_content"),
                        "refresh": bool(args.get("refresh")),
                        "scrape": dict(args.get("search") or {}),
                    },
                    "score": {
                        "requested": bool(args.get("show_score")),
                        "visible": bool(args.get("show_score")),
                    },
                },
                "targets": {
                    "cv": {
                        "scope": "one" if len(cv_ids) == 1 else "all",
                        "ids": cv_ids,
                    },
                    "job": {
                        "scope": "one" if len(job_ids) == 1 else "all",
                        "keys": job_ids,
                    },
                },
            },
            "errors": [],
        }

    def _updated_args(
        self,
        state: JobActionState,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        args: dict[str, Any] = dict(state.get("args") or {})
        jobs_update: dict[str, Any] = (
            dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
        )
        current_results: list[dict[str, Any]] = [
            self._container.events.job_result(item)
            for item in (args.get("job") or [])
            if isinstance(item, dict)
        ]
        results: list[dict[str, Any]] = [
            item
            for item in (jobs_update.get("results") or current_results)
            if isinstance(item, dict)
        ]
        args["job"] = [
            self._container.events.job_item(item, index)
            for index, item in enumerate(results)
        ]
        for key in (
            "scrape_total",
            "scrape_truncated",
            "active_job_keys",
            "matches",
        ):
            if key in jobs_update:
                args[key] = jobs_update[key]
        matches: list[dict[str, Any]] = [
            item for item in (args.get("matches") or []) if isinstance(item, dict)
        ]
        score_by_job: dict[str, Any] = {
            str(item.get("job_key") or ""): item.get("score") for item in matches
        }
        for item in args.get("job") or []:
            if isinstance(item, dict) and str(item.get("id") or "") in score_by_job:
                item["match"] = score_by_job[str(item.get("id") or "")]
        args["need_to_search"] = not bool(args.get("job"))
        args["need_to_extract"] = sum(
            1
            for item in (args.get("job") or [])
            if isinstance(item, dict)
            and (item.get("validation_status") != "valid" or not item.get("features"))
        )
        return args

    async def _search(self, state: JobActionState) -> dict[str, Any]:
        update: dict[
            str, Any
        ] = await self._container.job_workflow.scrape_jobs_with_mcp(
            self._conversation_state(state)  # type: ignore[arg-type]
        )
        failures: list[str] = list(update.get("errors") or [])
        args: dict[str, Any] = self._updated_args(state, update)
        job_ids: list[str] = [
            str(item.get("id") or "")
            for item in (args.get("job") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        return {
            "args": args,
            "timeline": self._container.events.append_event(
                list(state.get("timeline") or []),
                node=NODE_SCRAPE_JOBS,
                status="failed" if failures and not job_ids else "success",
                summary=(
                    failures[-1]
                    if failures and not job_ids
                    else f"{len(job_ids)} job(s) are currently available."
                ),
                args={
                    "search": dict(args.get("search") or {}),
                    "job_ids": job_ids,
                    "result_count": len(job_ids),
                },
            ),
        }

    async def _extract(self, state: JobActionState) -> dict[str, Any]:
        update: dict[str, Any] = await self._container.job_workflow.extract_pasted_job(
            self._conversation_state(state)  # type: ignore[arg-type]
        )
        failures: list[str] = list(update.get("errors") or [])
        return {
            "args": self._updated_args(state, update),
            "timeline": self._container.events.append_event(
                list(state.get("timeline") or []),
                node=NODE_EXTRACT_PASTED_JOB,
                status="failed" if failures else "success",
                summary=failures[-1]
                if failures
                else "Pasted job extraction completed.",
            ),
        }

    async def _match(self, state: JobActionState) -> dict[str, Any]:
        update: dict[
            str, Any
        ] = await self._container.job_workflow.calculate_job_matches(
            self._conversation_state(state)  # type: ignore[arg-type]
        )
        failures: list[str] = list(update.get("errors") or [])
        args: dict[str, Any] = self._updated_args(state, update)
        matches: list[dict[str, Any]] = [
            item for item in (args.get("matches") or []) if isinstance(item, dict)
        ]
        verdicts: list[str] = [
            str((item.get("score") or {}).get("fit_verdict") or "uncertain")
            for item in matches
        ]
        fit_count: int = sum(value == "yes" for value in verdicts)
        not_fit_count: int = sum(value == "no" for value in verdicts)
        uncertain_count: int = len(verdicts) - fit_count - not_fit_count
        return {
            "args": args,
            "timeline": self._container.events.append_event(
                list(state.get("timeline") or []),
                node=NODE_MATCH_JOBS,
                status="failed" if failures and not matches else "success",
                summary=(
                    failures[-1]
                    if failures and not matches
                    else (
                        f"{len(matches)} job match(es) evaluated: "
                        f"{fit_count} fit, {not_fit_count} not fit, "
                        f"{uncertain_count} uncertain."
                    )
                ),
                args={
                    "cv_ids": list(args.get("cv_ids") or []),
                    "job_ids": list(args.get("job_ids") or []),
                    "matches": matches,
                },
            ),
        }

    @staticmethod
    def _next_node(state: JobSubagentState) -> str:
        if (state.get("validation") or {}).get("errors"):
            return BRANCH_END
        timeline: list[TimelineEventState] = list(state.get("timeline") or [])
        if timeline and timeline[-1].get("status") in {"failed", "pending"}:
            return BRANCH_END
        completed: set[str] = {
            str(event.get("node") or "")
            for event in timeline
            if event.get("status") == "success"
        }
        for step in state.get("plan") or []:
            node: str = str(step.get("node") or "")
            if node not in completed:
                return node
        return BRANCH_END

    def build(self) -> Any:
        builder: StateGraph = StateGraph(JobSubagentState)
        builder.add_node(NODE_JOB_PLANNER, self._plan)
        builder.add_node(NODE_SCRAPE_JOBS, self._search, input_schema=JobActionState)
        builder.add_node(
            NODE_EXTRACT_PASTED_JOB,
            self._extract,
            input_schema=JobActionState,
        )
        builder.add_node(NODE_MATCH_JOBS, self._match, input_schema=JobActionState)
        builder.add_edge(START, NODE_JOB_PLANNER)
        destinations: dict[str, str] = {
            NODE_SCRAPE_JOBS: NODE_SCRAPE_JOBS,
            NODE_EXTRACT_PASTED_JOB: NODE_EXTRACT_PASTED_JOB,
            NODE_MATCH_JOBS: NODE_MATCH_JOBS,
            BRANCH_END: END,
        }
        builder.add_conditional_edges(NODE_JOB_PLANNER, self._next_node, destinations)
        builder.add_conditional_edges(NODE_SCRAPE_JOBS, self._next_node, destinations)
        builder.add_conditional_edges(
            NODE_EXTRACT_PASTED_JOB,
            self._next_node,
            destinations,
        )
        builder.add_conditional_edges(NODE_MATCH_JOBS, self._next_node, destinations)
        return builder.compile(name="job_subagent")
