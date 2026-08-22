"""Mechanically validate the planner's selected action.

Validation may reject missing prerequisites, invalid targets, duplicate actions,
or action-limit violations. It never reclassifies the planner's chosen action.
"""

from __future__ import annotations

from typing import Any

from app.config.const.chatbot import (
    AGENT_ACTIONS,
    MAX_AGENT_ACTIONS,
    NODE_SCRAPE_JOBS,
    ROUTE_COMPARE_CVS,
    ROUTE_EXTRACT_CV,
    ROUTE_EXTRACT_JOB,
    ROUTE_MATCH_JOBS,
    ROUTE_RESPOND,
    ROUTE_REVIEW_CV,
    ROUTE_SEARCH_JOBS,
)
from app.config.const.chatbot_errors import (
    ERROR_ACTION_LIMIT_REACHED,
    ERROR_CV_ALREADY_EXTRACTED,
    ERROR_CV_COMPARISON_TARGET_COUNT_INVALID,
    ERROR_CV_EXTRACTION_REQUIRED,
    ERROR_CV_REVIEW_TARGET_COUNT_INVALID,
    ERROR_CV_TARGET_MISSING,
    ERROR_CV_UPLOAD_REQUIRED,
    ERROR_DUPLICATE_ACTION,
    ERROR_EXISTING_JOB_TARGETS_MISSING,
    ERROR_JOB_DATA_REQUIRED_BEFORE_MATCHING,
    ERROR_JOB_TARGET_MISSING,
    ERROR_PASTED_JOB_REQUIRED,
    ERROR_PLAN_VALIDATION_FAILED,
    ERROR_PLAN_VALIDATION_FAILED_PREFIX,
    ERROR_TARGET_AMBIGUOUS,
    ERROR_UNKNOWN_WORKFLOW_ACTION,
    GOAL_INVALIDATION_CANCELLED,
    GOAL_SOURCE_CV_DERIVED,
    GOAL_SOURCE_EXPLICIT_SEARCH,
    REASON_PLANNER_SELECTED_ACTION,
    REASON_WORKFLOW_ACTION_SELECTED,
    VALIDATION_ACCEPTED,
    VALIDATION_REJECTED,
)
from app.models.chatbot.literals import (
    JobResponse,
    JobSource,
    JobTask,
    RoleSource,
    RouteName,
)
from app.models.chatbot.schemas import RouteDecision, ScrapeRequest
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.catalog_repository import RoutingCatalogRepository
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.repositories.chatbot.job_state_repository import JobStateRepository
from app.services.chatbot.action_reuse_service import ActionReuseService
from app.services.chatbot.execution_service import ExecutionService
from app.services.chatbot.text_utils import TextUtils


class PlanValidationService:
    """Stage-4 mechanical validation and job-state persistence."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        cvs: CvStateRepository,
        jobs: JobStateRepository,
        catalogs: RoutingCatalogRepository,
        reuse: ActionReuseService,
        execution: ExecutionService,
    ) -> None:
        self._state = state
        self._cvs = cvs
        self._jobs = jobs
        self._catalogs = catalogs
        self._reuse = reuse
        self._execution = execution

    def planned_job_stages(self,
        state: ConversationState,
        *,
        route: RouteName,
        selection: dict[str, Any],
        jobs_update: dict[str, Any],
    ) -> list[str]:
        gated: ConversationState = {
            **state,
            "routing": {
                "request": self._state.request_state_fields(selection),
                "targets": self._state.target_state_fields(selection),
            },
            "jobs": {**self._state.jobs_bucket(state), **jobs_update},
        }
        refresh: bool = bool(
            selection.get("job_refresh") or selection.get("refresh_requested")
        )
        if route == ROUTE_SEARCH_JOBS:
            if (
                not refresh
                and self._reuse.current_search_is_reusable(gated)
                and self._jobs.resolve_selected_jobs(gated)
            ):
                return []
            return [NODE_SCRAPE_JOBS]
        if route != ROUTE_MATCH_JOBS:
            return []
        stages: list[str] = []
        if selection.get("job_source") == "search":
            if refresh or not (
                self._reuse.current_search_is_reusable(gated) and self._jobs.resolve_selected_jobs(gated)
            ):
                stages.append(NODE_SCRAPE_JOBS)
        elif selection.get("job_source") == "pasted" and not self._state.jobs_bucket(gated).get(
            "results"
        ):
            stages.append(ROUTE_EXTRACT_JOB)
        if refresh or not self._reuse.action_result_is_reusable(
            gated,
            ROUTE_MATCH_JOBS,
            self._reuse.action_fingerprint(ROUTE_MATCH_JOBS, gated),
        ):
            stages.append(ROUTE_MATCH_JOBS)
        return stages


    def persist_planned_job_state(self,
        state: ConversationState,
        *,
        decision: RouteDecision,
        route: RouteName,
        selection: dict[str, Any],
        jobs_update: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Persist planner-selected job state without reclassifying the request."""
        current_goal: dict[str, Any] | None = self._jobs.active_job_goal(state)
        role_constraints: list[str] = TextUtils.normalize_role_constraints(
            selection.get("role_constraints")
        )
        role_evidence: str | None = selection.get("role_evidence")
        role_source: RoleSource = selection.get("role_source") or "none"
        if not role_constraints and current_goal and not current_goal.get("invalidated"):
            role_constraints = TextUtils.normalize_role_constraints(
                current_goal.get("role_constraints")
            )
            role_source = "active_goal"

        selected_state: ConversationState = {
            **state,
            "routing": {
                "request": self._state.request_state_fields(selection),
                "targets": self._state.target_state_fields(selection),
            },
        }
        document: dict[str, Any] | None = self._cvs.unambiguous_extracted_cv(selected_state)

        if route == ROUTE_SEARCH_JOBS and role_constraints:
            refresh_requested: bool = bool(
                selection.get("job_refresh")
                or selection.get("refresh_requested")
            )
            reuse_goal: bool = (
                refresh_requested
                and current_goal is not None
                and not current_goal.get("invalidated")
            )
            goal: dict[str, Any] = (
                current_goal
                if reuse_goal
                else self._jobs.build_active_job_goal(
                    source=(
                        GOAL_SOURCE_CV_DERIVED
                        if role_source == "cv_inferred"
                        else GOAL_SOURCE_EXPLICIT_SEARCH
                    ),
                    role_constraints=role_constraints,
                    cv_id=str(document.get("id")) if document else None,
                    cv_version=self._cvs.cv_version(document) if document else None,
                    originating_turn=self._state.last_user_text(state),
                )
            )
            jobs_update["active_job_goal"] = goal
            scrape_request: dict[str, Any] = dict(
                jobs_update.get("scrape_request") or {}
            )
            scrape_request["keywords"] = TextUtils.display_role_constraints(role_constraints)
            jobs_update["scrape_request"] = scrape_request
            selection = {
                **selection,
                "role_constraints": role_constraints,
                "role_evidence": role_evidence,
                "role_source": role_source,
                "job_source": "search",
            }
        elif route == ROUTE_MATCH_JOBS:
            if current_goal is not None and not current_goal.get("invalidated"):
                jobs_update["active_job_goal"] = current_goal
            if (
                selection.get("job_source") == "none"
                and self._state.jobs_bucket(state).get("results")
            ):
                selection = {**selection, "job_source": "existing"}
            if decision.assessment_requested:
                selection = {**selection, "assessment_requested": True}

        if selection.get("job_task") == "cancel":
            if current_goal is not None:
                jobs_update["active_job_goal"] = {
                    **current_goal,
                    "invalidated": True,
                    "invalidation_reason": GOAL_INVALIDATION_CANCELLED,
                }
            jobs_update["pending_match"] = None

        policy: dict[str, Any] = {
            "planned_stages": [],
            "policy_reason": REASON_PLANNER_SELECTED_ACTION,
            "active_goal_id": (
                jobs_update.get("active_job_goal") or {}
            ).get("id"),
        }
        if route in {ROUTE_SEARCH_JOBS, ROUTE_MATCH_JOBS}:
            policy["planned_stages"] = self.planned_job_stages(
                state,
                route=route,
                selection=selection,
                jobs_update=jobs_update,
            )
        return selection, jobs_update, policy

    def legacy_decision_from_stages(self, state: ConversationState) -> RouteDecision:
        request: dict[str, Any] = self._state.request_values_from_state(state)
        targets: dict[str, Any] = self._state.selection_bucket(state)
        plan: dict[str, Any] = self._state.plan_bucket(state)
        router: dict[str, Any] = self._state.router_bucket(state)
        goal: str = str(request.get("goal") or "general_question")
        job_task: JobTask = request.get("job_task") or "none"
        job_response: JobResponse = request.get("job_response") or "none"
        job_refresh: bool = bool(request.get("job_refresh"))
        job_source: JobSource = request.get("job_source") or "none"
        selected_ids: list[str] = [
            str(item).strip()
            for item in (targets.get("selected_cv_ids") or [])
            if str(item).strip()
        ]
        return RouteDecision(
            route=(
                plan.get("action")
                or router.get("planned_action")
                or ROUTE_RESPOND
            ),
            reason=str(
                plan.get("reason")
                or request.get("goal_reason")
                or REASON_WORKFLOW_ACTION_SELECTED
            ),
            job_task=job_task,
            job_response=job_response,
            job_refresh=job_refresh,
            job_source=job_source,
            score_requested=bool(request.get("score_requested")),
            assessment_requested=bool(request.get("assessment_requested")),
            role_constraints=list(request.get("role_constraints") or []),
            role_evidence=request.get("role_evidence"),
            role_source=request.get("role_source") or "none",
            role_candidates=list(request.get("role_candidates") or []),
            job_target_scope=targets.get("job_target_scope") or "none",
            decision_confidence=float(
                request.get("decision_confidence")
                if request.get("decision_confidence") is not None
                else 1.0
            ),
            review_target_role=request.get("review_target_role"),
            review_mode=request.get("review_mode") or "general",
            review_focus=request.get("review_focus"),
            review_mode_reason=request.get("review_mode_reason"),
            needs_cv_text=bool(request.get("needs_cv_text")),
            needs_cv_features=bool(request.get("needs_cv_features"))
            or goal in {ROUTE_REVIEW_CV, ROUTE_COMPARE_CVS, ROUTE_EXTRACT_CV}
            or job_task == "match",
            is_follow_up=bool(request.get("is_follow_up")),
            selected_cv_id=selected_ids[0] if len(selected_ids) == 1 else None,
            selected_job_keys=targets.get("selected_job_keys"),
            scrape_request=ScrapeRequest(
                **dict(request.get("scrape_request") or {})
            ),
        )


    def planned_action_validation_error(self,
        state: ConversationState,
        decision: RouteDecision,
    ) -> str | None:
        action: str = decision.route
        if action not in {ROUTE_RESPOND, *AGENT_ACTIONS}:
            return f"{ERROR_UNKNOWN_WORKFLOW_ACTION}{action}"
        if action == ROUTE_RESPOND:
            return None
        if len(self._state.completed_actions(state)) >= MAX_AGENT_ACTIONS:
            return ERROR_ACTION_LIMIT_REACHED

        selection: dict[str, Any] = self._state.selection_bucket(state)
        catalogs: dict[str, Any] = self._catalogs.routing_catalogs(state)
        selected_ids: list[str] = [
            str(item).strip()
            for item in (selection.get("selected_cv_ids") or [])
            if str(item).strip()
        ]
        if any(item not in catalogs["cv_ids"] for item in selected_ids):
            return ERROR_CV_TARGET_MISSING
        selected_keys: list[str] = [
            str(item).strip()
            for item in (selection.get("selected_job_keys") or [])
            if str(item).strip()
        ]
        valid_job_keys: set[str] = catalogs["job_keys"] | catalogs["match_keys"]
        if any(item not in valid_job_keys for item in selected_keys):
            return ERROR_JOB_TARGET_MISSING
        if selection.get("targets_ambiguous"):
            return ERROR_TARGET_AMBIGUOUS
        if action == ROUTE_EXTRACT_CV:
            if not self._cvs.state_cv_documents(state):
                return ERROR_CV_UPLOAD_REQUIRED
            if not self._cvs.cvs_need_extraction(state):
                return ERROR_CV_ALREADY_EXTRACTED
        if action in {ROUTE_REVIEW_CV, ROUTE_COMPARE_CVS, ROUTE_MATCH_JOBS}:
            if not self._cvs.state_cv_documents(state):
                return ERROR_CV_UPLOAD_REQUIRED
            extracted_ids: set[str] = {
                str(item.get("id") or "") for item in self._cvs.extracted_cv_documents(state)
            }
            target_ids: set[str] = set(selected_ids) if selected_ids else extracted_ids
            if not target_ids.issubset(extracted_ids):
                return ERROR_CV_EXTRACTION_REQUIRED
            if action == ROUTE_REVIEW_CV and len(target_ids) != 1:
                return ERROR_CV_REVIEW_TARGET_COUNT_INVALID
            if action == ROUTE_COMPARE_CVS and len(target_ids) < 2:
                return ERROR_CV_COMPARISON_TARGET_COUNT_INVALID
        if action == ROUTE_EXTRACT_JOB and not (
            selection.get("job_input_text") or ""
        ).strip():
            return ERROR_PASTED_JOB_REQUIRED
        if action == ROUTE_MATCH_JOBS:
            if selection.get("job_source") == "existing" and not catalogs["job_keys"]:
                return ERROR_EXISTING_JOB_TARGETS_MISSING
            if selection.get("job_source") in {"search", "pasted"}:
                if not catalogs["job_keys"]:
                    return ERROR_JOB_DATA_REQUIRED_BEFORE_MATCHING
        if action in self._state.completed_actions(state) and not selection.get("refresh_requested"):
            if action in {ROUTE_SEARCH_JOBS, ROUTE_MATCH_JOBS}:
                if not self._reuse.action_result_is_reusable(
                    state,
                    action,
                    self._reuse.action_fingerprint(action, state),
                ):
                    return f"{ERROR_DUPLICATE_ACTION}{action}"
            else:
                return f"{ERROR_DUPLICATE_ACTION}{action}"
        return None


    async def validate_plan_node(self, state: ConversationState) -> dict[str, Any]:
        decision: RouteDecision = self.legacy_decision_from_stages(state)
        base_error: str | None = self.planned_action_validation_error(state, decision)
        if base_error:
            return {
                "routing": {
                    "plan": {
                        "action": ROUTE_RESPOND,
                        "reason": base_error,
                        "validation": VALIDATION_REJECTED,
                        "validation_error": base_error,
                        "planned_stages": [],
                    },
                },
                "errors": self._state.state_errors(state, [base_error]),
            }

        selection: dict[str, Any] = self._state.selection_bucket(state)
        scrape_request: dict[str, Any] = decision.scrape_request.model_dump(
            exclude_none=True
        )
        if not scrape_request:
            scrape_request = dict(self._state.jobs_bucket(state).get("scrape_request") or {})
        jobs_update: dict[str, Any] = {
            "scrape_request": scrape_request,
        }
        try:
            route: RouteName = decision.route
            route_reason: str = decision.reason
            selection, jobs_update, policy = self.persist_planned_job_state(
                state,
                decision=decision,
                route=route,
                selection=selection,
                jobs_update=jobs_update,
            )
            request_state: dict[str, Any] = self._state.request_state_fields(selection)
            request_state["cv"] = {
                **dict(request_state.get("cv") or {}),
                "text_needed": bool(decision.needs_cv_text),
                "features_needed": bool(decision.needs_cv_features),
            }
            result: dict[str, Any] = {
                "routing": {
                    "request": request_state,
                    "targets": self._state.target_state_fields(selection),
                    "plan": {
                        "action": route,
                        "reason": route_reason,
                        "validation": VALIDATION_ACCEPTED,
                        "validation_error": None,
                        "planned_stages": policy.get("planned_stages") or [],
                        "policy_reason": policy.get("policy_reason") or "",
                        "active_goal_id": policy.get("active_goal_id"),
                    },
                },
                "jobs": jobs_update,
            }
            if route in {ROUTE_SEARCH_JOBS, ROUTE_MATCH_JOBS} and not (
                policy.get("planned_stages") or []
            ):
                result["execution"] = {
                    "steps": [
                        *self._state.execution_steps(state),
                        self._execution.build_skipped_execution_step(state, route),
                    ]
                }
            return result
        except Exception as exc:
            reason: str = ERROR_PLAN_VALIDATION_FAILED
            return {
                "routing": {
                    "plan": {
                        "action": ROUTE_RESPOND,
                        "reason": reason,
                        "validation": VALIDATION_REJECTED,
                        "validation_error": reason,
                        "planned_stages": [],
                    },
                },
                "errors": self._state.state_errors(
                    state,
                    [f"{ERROR_PLAN_VALIDATION_FAILED_PREFIX}{type(exc).__name__}"],
                ),
            }
