"""Record each executed agent action as a structured execution step."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AnyMessage

from app.config.const.chatbot import (
    EXECUTION_DESTINATIONS,
    EXECUTION_NODE_USES,
    GRAPH_CHATBOT,
    GRAPH_CV_SUBAGENT,
    GRAPH_JOB_SUBAGENT,
    NODE_EXTRACT_PASTED_JOB,
    NODE_SCRAPE_JOBS,
    NODE_VALIDATE_PLAN,
    ROUTE_COMPARE_CVS,
    ROUTE_EXTRACT_CV,
    ROUTE_EXTRACT_JOB,
    ROUTE_MATCH_JOBS,
    ROUTE_RESPOND,
    ROUTE_REVIEW_CV,
    ROUTE_SEARCH_JOBS,
    USER_FACING_ACTIONS,
)
from app.config.const.chatbot_errors import (
    ERROR_JOB_SCRAPING_FAILED,
    REASON_ACTION_COMPLETED,
    REASON_ACTION_FAILED,
    REASON_CV_FEATURES_READY,
    REASON_JOB_EXTRACTION_READY,
    REASON_PLAN_VALIDATED,
    REASON_REUSED_EXISTING_RESULT,
    REASON_SEARCH_RESULTS_READY,
    STATUS_UNAVAILABLE,
)
from app.models.chatbot.literals import AgentAction, ExecutionStatus
from app.models.chatbot.state import (
    ConversationState,
    ExecutionContextState,
    ExecutionFromNodeState,
    ExecutionStepState,
    ExecutionToNodeState,
    merge_routing_maps,
)
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.repositories.chatbot.job_state_repository import JobStateRepository
from app.services.chatbot.action_reuse_service import ActionReuseService
from app.services.chatbot.job_key_utils import JobKeyUtils
from app.services.chatbot.result_projection_service import ResultProjectionService


class ExecutionService:
    """Build execution trace steps and attach them to node updates."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        cvs: CvStateRepository,
        jobs: JobStateRepository,
        reuse: ActionReuseService,
        projection: ResultProjectionService,
    ) -> None:
        self._state = state
        self._cvs = cvs
        self._jobs = jobs
        self._reuse = reuse
        self._projection = projection

    def _new_action_errors(
        self,
        state: ConversationState,
        update: dict[str, Any],
    ) -> list[str]:
        previous: set[str] = {
            str(item) for item in (state.get("errors") or []) if str(item)
        }
        return [
            str(item)
            for item in (update.get("errors") or [])
            if str(item) and str(item) not in previous
        ]

    def _execution_source_node(
        self,
        state: ConversationState,
        action: AgentAction,
    ) -> tuple[str, str, str]:
        actions: set[str] = set(self._state.completed_actions(state))
        if (
            action in {ROUTE_REVIEW_CV, ROUTE_COMPARE_CVS}
            and ROUTE_EXTRACT_CV in actions
        ):
            return GRAPH_CV_SUBAGENT, ROUTE_EXTRACT_CV, REASON_CV_FEATURES_READY
        if action == ROUTE_MATCH_JOBS:
            if ROUTE_SEARCH_JOBS in actions:
                return GRAPH_JOB_SUBAGENT, NODE_SCRAPE_JOBS, REASON_SEARCH_RESULTS_READY
            if ROUTE_EXTRACT_JOB in actions:
                return (
                    GRAPH_JOB_SUBAGENT,
                    NODE_EXTRACT_PASTED_JOB,
                    REASON_JOB_EXTRACTION_READY,
                )
        plan: dict[str, Any] = self._state.plan_bucket(state)
        return (
            GRAPH_CHATBOT,
            NODE_VALIDATE_PLAN,
            str(plan.get("reason") or REASON_PLAN_VALIDATED),
        )

    def _execution_args(
        self,
        action: AgentAction,
        state: ConversationState,
    ) -> dict[str, Any]:
        request: dict[str, Any] = self._state.request_values_from_state(state)
        if action == ROUTE_EXTRACT_CV:
            documents: list[dict[str, Any]] = self._cvs.state_cv_documents(state)
            ids: list[str] = [
                str(document.get("id"))
                for document in documents
                if str(document.get("id") or "").strip()
            ]
            if ids:
                return {"cv_ids": ids}
            return {
                "cv_filenames": [
                    str(document.get("filename") or "cv.pdf") for document in documents
                ]
            }
        if action == ROUTE_REVIEW_CV:
            target: dict[str, Any] | None = self._cvs.resolve_selected_cv(state)
            args: dict[str, Any] = {
                "cv_id": target.get("id") if target else None,
                "mode": request.get("review_mode") or "general",
            }
            if request.get("review_focus"):
                args["focus"] = request["review_focus"]
            if request.get("review_target_role"):
                args["target_role"] = request["review_target_role"]
            return {
                key: value for key, value in args.items() if value not in (None, "")
            }
        if action == ROUTE_COMPARE_CVS:
            return {
                "cv_ids": [
                    str(document.get("id"))
                    for document in self._cvs.resolve_selected_cvs(state)
                    if str(document.get("id") or "").strip()
                ]
            }
        if action == ROUTE_EXTRACT_JOB:
            return {"source": "pasted"}
        if action == ROUTE_SEARCH_JOBS:
            raw: Any = self._state.jobs_bucket(state).get("scrape_request")
            if not isinstance(raw, dict):
                raw = request.get("scrape_request")
            raw_request: dict[str, Any] = raw if isinstance(raw, dict) else {}
            return {
                key: value
                for key, value in raw_request.items()
                if value not in (None, "", [], {})
            }
        if action == ROUTE_MATCH_JOBS:
            jobs: list[dict[str, Any]] = self._jobs.resolve_selected_jobs(state)
            args = {
                "cv_ids": [
                    str(document.get("id"))
                    for document in self._cvs.resolve_selected_cvs(state)
                    if str(document.get("id") or "").strip()
                ],
                "job_keys": [
                    JobKeyUtils.job_selection_key(item, index)
                    for index, item in enumerate(jobs)
                ],
            }
            if request.get("job_source"):
                args["source"] = request["job_source"]
            return args
        return {}

    def _execution_context(
        self,
        action: AgentAction,
        state: ConversationState,
    ) -> ExecutionContextState:
        if action != ROUTE_SEARCH_JOBS:
            return {}
        request: dict[str, Any] = self._state.request_values_from_state(state)
        source: Any = request.get("role_source") or "none"
        if source == "none":
            return {}
        context: ExecutionContextState = {"args_source": source}
        if source == "cv_inferred":
            document: dict[str, Any] | None = self._cvs.resolve_selected_cv(state)
            if document:
                context["source_reference"] = {
                    "cv_id": document.get("id"),
                    "field": "cv_features.role_tags",
                }
        elif source == "active_goal":
            goal: dict[str, Any] | None = self._jobs.active_job_goal(state)
            if goal:
                context["source_reference"] = {
                    "goal_id": goal.get("id"),
                    "field": "jobs.active_job_goal.role_constraints",
                }
        else:
            context["source_reference"] = {"field": "routing.request.role.evidence"}
        return context

    def _execution_result(
        self,
        action: AgentAction,
        state: ConversationState,
        update: dict[str, Any],
        merged_state: ConversationState,
    ) -> tuple[ExecutionStatus, dict[str, Any], str | None]:
        errors: list[str] = self._new_action_errors(state, update)
        jobs_update: dict[str, Any] = (
            dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
        )
        if action == ROUTE_EXTRACT_CV:
            documents = self._cvs.extracted_cv_documents(merged_state)
            result = {
                "cv_count": len(documents),
                "cv_ids": [str(item.get("id")) for item in documents],
                "errors": errors,
            }
            failed = not any(item.get("cv_features") for item in documents)
        elif action == ROUTE_REVIEW_CV:
            review: Any = (update.get("cv") or {}).get("review")
            result = {
                "status": review.get("status") if isinstance(review, dict) else None,
                "feedback_count": len(review.get("feedback") or [])
                if isinstance(review, dict)
                else 0,
                "errors": errors,
            }
            failed = (
                not isinstance(review, dict)
                or review.get("status") == STATUS_UNAVAILABLE
            )
        elif action == ROUTE_COMPARE_CVS:
            comparison: Any = (update.get("cv") or {}).get("comparison")
            result = {
                "candidate_count": len(comparison.get("candidates") or [])
                if isinstance(comparison, dict)
                else 0,
                "errors": errors,
            }
            failed = not isinstance(comparison, dict)
        elif action == ROUTE_EXTRACT_JOB:
            payload: dict[str, Any] = self._projection.slim_extract_job_result(
                jobs_update, state
            )
            result = {
                "job_count": payload.get("job_count", 0),
                "validation_status": payload.get("validation_status"),
                "errors": errors,
            }
            failed = payload.get("validation_status") != "valid"
        elif action == ROUTE_SEARCH_JOBS:
            payload = self._projection.slim_search_result(jobs_update, state)
            result = {
                "job_count": payload.get("job_count", 0),
                "scrape_total": payload.get("scrape_total", 0),
                "truncated": bool(payload.get("scrape_truncated")),
                "errors": errors,
            }
            failed = any(item.startswith(ERROR_JOB_SCRAPING_FAILED) for item in errors)
        else:
            matches: list[Any] = [
                item
                for item in (jobs_update.get("matches") or [])
                if isinstance(item, dict)
            ]
            result = {"match_count": len(matches), "errors": errors}
            failed = bool(errors)
        status: ExecutionStatus = "failed" if failed else "completed"
        error: str | None = errors[0] if failed and errors else None
        return status, result, error

    def build_execution_step(
        self,
        state: ConversationState,
        action: AgentAction,
        update: dict[str, Any],
        merged_state: ConversationState,
    ) -> ExecutionStepState:
        source_graph, source_node, source_reason = self._execution_source_node(
            state, action
        )
        destination_graph, destination_node = EXECUTION_DESTINATIONS[action]
        status, result, error = self._execution_result(
            action, state, update, merged_state
        )
        route: Any = self._state.router_bucket(state).get(
            "route"
        ) or self._state.plan_bucket(state).get("action")
        from_node: ExecutionFromNodeState = {
            "graph": source_graph,
            "node": source_node,
            "uses": list(EXECUTION_NODE_USES.get((source_graph, source_node), [])),
            "note": {
                "summary": "Selected the next executable action.",
                "reason": source_reason,
            },
            "output": {"route": route or ROUTE_RESPOND, "next_action": action},
        }
        to_node: ExecutionToNodeState = {
            "graph": destination_graph,
            "node": destination_node,
            "uses": list(
                EXECUTION_NODE_USES.get((destination_graph, destination_node), [])
            ),
            "note": {
                "summary": f"Executed {action}.",
                "reason": REASON_ACTION_FAILED
                if status == "failed"
                else REASON_ACTION_COMPLETED,
            },
            "args": self._execution_args(action, state),
            "result": result,
        }
        return {
            "index": len(self._state.execution_steps(state)) + 1,
            "action": action,
            "status": status,
            "from": from_node,
            "to": to_node,
            "context": self._execution_context(action, state),
            "error": error,
        }

    def build_skipped_execution_step(
        self,
        state: ConversationState,
        action: AgentAction,
    ) -> ExecutionStepState:
        destination_graph, destination_node = EXECUTION_DESTINATIONS[action]
        if action == ROUTE_SEARCH_JOBS:
            result = {
                "reused": True,
                "job_count": len(self._jobs.job_results_for_display(state)),
            }
        else:
            result = {
                "reused": True,
                "match_count": len(self._state.jobs_bucket(state).get("matches") or []),
            }
        return {
            "index": len(self._state.execution_steps(state)) + 1,
            "action": action,
            "status": "skipped",
            "from": {
                "graph": GRAPH_CHATBOT,
                "node": NODE_VALIDATE_PLAN,
                "uses": list(EXECUTION_NODE_USES[(GRAPH_CHATBOT, NODE_VALIDATE_PLAN)]),
                "note": {
                    "summary": "Selected the next executable action.",
                    "reason": str(
                        self._state.plan_bucket(state).get("reason")
                        or REASON_PLAN_VALIDATED
                    ),
                },
                "output": {
                    "route": self._state.router_bucket(state).get("route")
                    or ROUTE_RESPOND,
                    "next_action": action,
                },
            },
            "to": {
                "graph": destination_graph,
                "node": destination_node,
                "uses": list(
                    EXECUTION_NODE_USES[(destination_graph, destination_node)]
                ),
                "note": {
                    "summary": f"Skipped {action}.",
                    "reason": REASON_REUSED_EXISTING_RESULT,
                },
                "args": self._execution_args(action, state),
                "result": result,
            },
            "context": self._execution_context(action, state),
            "error": None,
        }

    def record_completed_action(
        self,
        state: ConversationState,
        action: AgentAction,
        update: dict[str, Any],
        *,
        emit_result: bool = False,
    ) -> dict[str, Any]:
        """Attach one executed action to a node update without duplicate entries."""
        actions: list[str] = self._state.completed_actions(state)
        if action not in actions:
            actions.append(action)
        routing_update: Any = update.get("routing")
        plan_update: Any = (
            routing_update.get("plan")
            if isinstance(routing_update, dict)
            else update.get("plan") or update.get("router")
        )
        nested: dict[str, Any] = (
            dict(plan_update) if isinstance(plan_update, dict) else {}
        )
        rest: dict[str, Any] = {
            key: value
            for key, value in update.items()
            if key not in {"router", "plan", "routing"}
        }
        if emit_result and action in USER_FACING_ACTIONS:
            payload: dict[str, Any] | None = self._projection.slim_action_result(
                action, update, state
            )
            if payload is not None:
                result_messages: list[AnyMessage] = (
                    self._projection.build_action_result_messages(action, payload)
                )
                existing_messages: Any = rest.get("messages")
                if isinstance(existing_messages, list) and existing_messages:
                    rest["messages"] = [*existing_messages, *result_messages]
                else:
                    rest["messages"] = result_messages
        recorded_routing: dict[str, Any] = (
            dict(routing_update) if isinstance(routing_update, dict) else {}
        )
        recorded_routing["plan"] = {**nested, "completed_actions": actions}
        recorded: dict[str, Any] = {**rest, "routing": recorded_routing}
        merged_state: ConversationState = {
            **state,
            **{key: value for key, value in rest.items() if key != "messages"},
        }
        if isinstance(update.get("cv"), dict):
            merged_state["cv"] = {**self._state.cv_bucket(state), **update["cv"]}
        if isinstance(update.get("jobs"), dict):
            merged_state["jobs"] = {**self._state.jobs_bucket(state), **update["jobs"]}
        if isinstance(update.get("routing"), dict):
            merged_state["routing"] = merge_routing_maps(
                self._state.routing_bucket(state),
                update["routing"],
            )
        if isinstance(update.get("targets"), dict):
            merged_state["targets"] = {
                **self._state.targets_bucket(state),
                **update["targets"],
            }
        if isinstance(update.get("selection"), dict):
            merged_state["targets"] = {
                **self._state.targets_bucket(state),
                **update["selection"],
            }
        step: ExecutionStepState = self.build_execution_step(
            state,
            action,
            update,
            merged_state,
        )
        recorded["execution"] = {
            "steps": [*self._state.execution_steps(state), step],
        }
        snapshot: dict[str, Any] | None = self._reuse.reusable_action_snapshot(
            action, update, state
        )
        if snapshot is None:
            return recorded
        fingerprint: str | None = self._reuse.action_fingerprint(action, merged_state)
        if not fingerprint:
            return recorded
        return {
            **recorded,
            "action_results": {
                action: {
                    "fingerprint": fingerprint,
                    "executed_at": datetime.now(timezone.utc).isoformat(),
                    "snapshot": snapshot,
                }
            },
        }
