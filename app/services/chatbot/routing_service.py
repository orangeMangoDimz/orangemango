"""Classify the user's request and select the next workflow action."""

from __future__ import annotations

import json
from typing import Any

from app.config.const.chatbot import (
    MAX_AGENT_ACTIONS,
    MAX_ROUTER_CHARS,
    ROUTE_MATCH_JOBS,
    ROUTE_RESPOND,
    ROUTE_SEARCH_JOBS,
)
from app.config.const.chatbot_errors import (
    ERROR_REQUEST_ROUTER_FAILED_PREFIX,
    ERROR_WORKFLOW_PLANNER_FAILED_PREFIX,
    REASON_NO_USER_MESSAGE,
    REASON_REQUEST_ROUTER_FAILED,
    REASON_SEARCH_COMPLETE_ASSESSMENT,
    REASON_SEARCH_COMPLETE_PRESENT_RESULTS,
    REASON_WORKFLOW_PLANNER_FAILED,
    VALIDATION_PENDING,
)
from app.config.const.chatbot_prompts import (
    REQUEST_ROUTER_DATA_HEADER,
    REQUEST_ROUTER_PROMPT,
    WORKFLOW_PLANNER_DATA_HEADER,
    WORKFLOW_PLANNER_PROMPT,
)
from app.models.chat_model import ChatModel
from app.models.chatbot.literals import JobResponse, JobSource, JobTask
from app.models.chatbot.schemas import RequestDecision, RoleCandidate, WorkflowPlan
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.catalog_repository import RoutingCatalogRepository
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.repositories.chatbot.job_state_repository import JobStateRepository
from app.services.chatbot.action_reuse_service import ActionReuseService
from app.services.chatbot.conversation_service import ConversationService
from app.services.chatbot.text_utils import TextUtils


class RequestRoutingService:
    """Stage 2 request classification and stage 3 workflow planning."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        cvs: CvStateRepository,
        jobs: JobStateRepository,
        catalogs: RoutingCatalogRepository,
        reuse: ActionReuseService,
        conversation: ConversationService,
        chat_model: ChatModel,
    ) -> None:
        self._state = state
        self._cvs = cvs
        self._jobs = jobs
        self._catalogs = catalogs
        self._reuse = reuse
        self._conversation = conversation
        self._chat_model = chat_model

    def normalize_job_role_fields(
        self,
        latest: str,
        *,
        job_task: str,
        role_constraints: Any,
        role_evidence: Any,
        role_source: Any,
        scrape_keywords: Any,
    ) -> tuple[list[str], str | None, str]:
        """Recover explicit role fields when the structured router omits them."""
        constraints: list[str] = TextUtils.normalize_role_constraints(role_constraints)
        evidence: str = str(role_evidence or "").strip()
        source: str = str(role_source or "none")
        if job_task != "search":
            return constraints, evidence or None, source

        keywords: list[str] = TextUtils.normalize_role_constraints(scrape_keywords)
        if not constraints:
            constraints = keywords
        if not evidence:
            evidence = (
                TextUtils.first_contiguous_phrase(latest, keywords or constraints) or ""
            )
        if source == "none" and evidence:
            source = "explicit"
        return constraints, evidence or None, source

    def role_constraints_from_cv(self, document: dict[str, Any] | None) -> list[str]:
        if not isinstance(document, dict):
            return []
        features: Any = document.get("cv_features")
        if not isinstance(features, dict):
            return []
        return TextUtils.normalize_role_constraints(features.get("role_tags") or [])

    def normalize_role_candidates(self, values: Any) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in values or []:
            if isinstance(item, RoleCandidate):
                raw_role: Any = item.role
                raw_confidence: Any = item.confidence
                raw_evidence: Any = item.evidence
            elif isinstance(item, dict):
                raw_role = item.get("role")
                raw_confidence = item.get("confidence", 1.0)
                raw_evidence = item.get("evidence")
            else:
                continue
            role: str = TextUtils.normalize_fingerprint_text(raw_role)
            evidence: str = TextUtils.short_text(raw_evidence or "", 300)
            if not role or role in seen or not evidence:
                continue
            try:
                confidence: float = max(0.0, min(1.0, float(raw_confidence)))
            except (TypeError, ValueError):
                confidence = 0.0
            seen.add(role)
            candidates.append(
                {
                    "role": role,
                    "confidence": confidence,
                    "evidence": evidence,
                }
            )
        return sorted(candidates, key=lambda item: item["confidence"], reverse=True)[:5]

    def request_router_context(self, state: ConversationState) -> dict[str, Any]:
        catalogs: dict[str, Any] = self._catalogs.routing_catalogs(state)
        return {
            "latest_user_message": self._state.last_user_text(state)[:MAX_ROUTER_CHARS],
            "recent_conversation": self._conversation.router_recent_conversation(state),
            "conversation_memory": self._state.conversation_memory(state),
            "active_job_goal": self._jobs.active_job_goal(state),
            "cv_profiles": self._catalogs.routing_cv_profiles(state),
            "cvs": catalogs["cvs"],
            "jobs": catalogs["jobs"],
            "matches": catalogs["matches"],
        }

    def mapped_job_request(
        self,
        decision: RequestDecision,
    ) -> tuple[JobTask, JobResponse, JobSource, bool]:
        """Normalize the router's mapped job contract."""
        task: JobTask = decision.job.task
        response: JobResponse = decision.job.response
        source: JobSource = decision.job.source
        refresh: bool = bool(decision.job.refresh)

        if task == "extract" and source == "none":
            source = "pasted"
        if task == "search" and source == "none":
            source = "search"
        if task == "match" and source == "none":
            source = "existing"
        return task, response, source, refresh

    def planner_context(self, state: ConversationState) -> dict[str, Any]:
        catalogs: dict[str, Any] = self._catalogs.routing_catalogs(state)
        documents: list[dict[str, Any]] = self._cvs.state_cv_documents(state)
        extracted_ids: list[str] = [
            str(document.get("id") or "")
            for document in self._cvs.extracted_cv_documents(state)
            if str(document.get("id") or "").strip()
        ]
        targets: dict[str, Any] = self._state.selection_bucket(state)
        request: dict[str, Any] = self._state.request_values_from_state(state)
        return {
            "goal": {
                key: request.get(key)
                for key in (
                    "goal",
                    "goal_reason",
                    "job_task",
                    "job_response",
                    "job_source",
                    "assessment_requested",
                    "score_requested",
                    "role_constraints",
                    "review_mode",
                    "review_focus",
                    "needs_cv_features",
                    "is_follow_up",
                    "role_source",
                    "role_candidates",
                )
                if request.get(key) is not None
            },
            "targets": {
                key: targets.get(key)
                for key in (
                    "cv_target_scope",
                    "selected_cv_ids",
                    "selected_job_keys",
                    "job_target_scope",
                    "unresolved_references",
                    "targets_ambiguous",
                )
                if targets.get(key) is not None
            },
            "state_facts": {
                "cv_count": len(documents),
                "cv_ids": sorted(catalogs["cv_ids"]),
                "extracted_cv_ids": extracted_ids,
                "cv_needs_extraction": self._cvs.cvs_need_extraction(state),
                "valid_job_keys": sorted(catalogs["job_keys"]),
                "valid_match_keys": sorted(catalogs["match_keys"]),
                "job_count": len(catalogs["jobs"]),
                "match_count": len(catalogs["matches"]),
                "search_reusable": self._reuse.current_search_is_reusable(state),
                "match_reusable": self._reuse.action_result_is_reusable(
                    state,
                    ROUTE_MATCH_JOBS,
                    self._reuse.action_fingerprint(ROUTE_MATCH_JOBS, state),
                ),
                "cv_profiles": self._catalogs.routing_cv_profiles(state),
                "active_job_goal": self._jobs.active_job_goal(state),
                "pending_match": self._jobs.pending_match_request(state),
                "completed_actions": self._state.completed_actions(state),
                "remaining_action_budget": max(
                    0,
                    MAX_AGENT_ACTIONS - len(self._state.completed_actions(state)),
                ),
                "refresh_requested": bool(request.get("refresh_requested")),
                "pasted_job_available": bool(
                    (request.get("job_input_text") or "").strip()
                ),
            },
        }

    async def request_router_node(self, state: ConversationState) -> dict[str, Any]:
        latest: str = self._state.last_user_text(state)
        if not latest:
            decision = RequestDecision(
                goal="general_question",
                reason=REASON_NO_USER_MESSAGE,
            )
        else:
            try:
                classifier: Any = self._chat_model.structured(RequestDecision)
                decision = await classifier.ainvoke(
                    [
                        {"role": "system", "content": REQUEST_ROUTER_PROMPT},
                        {
                            "role": "user",
                            "content": REQUEST_ROUTER_DATA_HEADER
                            + json.dumps(
                                self.request_router_context(state),
                                ensure_ascii=False,
                            ),
                        },
                    ]
                )
            except Exception as exc:
                decision = RequestDecision(
                    goal="general_question",
                    reason=REASON_REQUEST_ROUTER_FAILED,
                )
                return {
                    "routing": {
                        "request": self._state.request_state_fields(
                            {"goal": decision.goal, "reason": decision.reason}
                        ),
                        "targets": self._state.default_target_fields(),
                    },
                    "errors": self._state.state_errors(
                        state,
                        [f"{ERROR_REQUEST_ROUTER_FAILED_PREFIX}{type(exc).__name__}"],
                    ),
                }

        (
            mapped_job_task,
            mapped_job_response,
            mapped_job_source,
            mapped_job_refresh,
        ) = self.mapped_job_request(decision)
        mapped_job_refresh = bool(
            mapped_job_refresh and self._jobs.active_job_goal(state)
        )
        catalogs: dict[str, Any] = self._catalogs.routing_catalogs(state)
        selected_cv_ids: list[str] = []
        invalid_cv_ids: list[str] = []
        for item in decision.selected_cv_ids:
            value: str = str(item).strip()
            if not value:
                continue
            if value in catalogs["cv_ids"] and value not in selected_cv_ids:
                selected_cv_ids.append(value)
            elif value not in catalogs["cv_ids"]:
                invalid_cv_ids.append(value)

        selected_job_keys: list[str] = []
        invalid_job_keys: list[str] = []
        valid_job_keys: set[str] = catalogs["job_keys"] | catalogs["match_keys"]
        for item in decision.selected_job_keys or []:
            value = str(item).strip()
            if not value:
                continue
            if value in valid_job_keys and value not in selected_job_keys:
                selected_job_keys.append(value)
            elif value not in valid_job_keys:
                invalid_job_keys.append(value)

        unresolved: list[str] = [
            *decision.unresolved_references,
            *(f"unknown CV: {item}" for item in invalid_cv_ids),
            *(f"unknown job: {item}" for item in invalid_job_keys),
        ]
        targets_ambiguous: bool = bool(decision.targets_ambiguous or unresolved)
        if decision.job_target_scope == "all":
            selected_job_keys = []

        request_input: dict[str, Any] = {
            **decision.model_dump(),
            "goal_reason": decision.reason,
            "job_task": mapped_job_task,
            "job_response": mapped_job_response,
            "job_refresh": mapped_job_refresh,
            "job_source": (
                "pasted" if decision.job.task == "extract" else mapped_job_source
            ),
            "job_input_text": (latest if mapped_job_source == "pasted" else None),
            "scrape_request": decision.job.scrape.model_dump(exclude_none=True),
            "show_score": bool(decision.score_requested),
            "refresh_requested": bool(mapped_job_refresh),
            "role_candidates": [item.model_dump() for item in decision.role_candidates],
        }
        request: dict[str, Any] = self._state.request_state_fields(request_input)
        request_values_view: dict[str, Any] = self._state.request_values(request)
        role_constraints, role_evidence, role_source = self.normalize_job_role_fields(
            latest,
            job_task=mapped_job_task,
            role_constraints=request_values_view.get("role_constraints"),
            role_evidence=request_values_view.get("role_evidence"),
            role_source=request_values_view.get("role_source"),
            scrape_keywords=request_values_view.get("scrape_request", {}).get(
                "keywords"
            )
            if isinstance(request_values_view.get("scrape_request"), dict)
            else [],
        )
        request["role"] = {
            **dict(request.get("role") or {}),
            "constraints": role_constraints,
            "evidence": role_evidence,
            "source": role_source,
        }
        targets: dict[str, Any] = self._state.target_state_fields(
            {
                "cv_target_scope": decision.cv_target_scope,
                "selected_cv_ids": selected_cv_ids,
                "job_target_scope": decision.job_target_scope,
                "selected_job_keys": selected_job_keys,
                "unresolved_references": unresolved,
                "targets_ambiguous": targets_ambiguous,
            }
        )
        return {"routing": {"request": request, "targets": targets}}

    async def workflow_planner_node(self, state: ConversationState) -> dict[str, Any]:
        try:
            planner: Any = self._chat_model.structured(WorkflowPlan)
            plan: WorkflowPlan = await planner.ainvoke(
                [
                    {"role": "system", "content": WORKFLOW_PLANNER_PROMPT},
                    {
                        "role": "user",
                        "content": WORKFLOW_PLANNER_DATA_HEADER
                        + json.dumps(self.planner_context(state), ensure_ascii=False),
                    },
                ]
            )
        except Exception as exc:
            plan = WorkflowPlan(
                action=ROUTE_RESPOND,
                reason=REASON_WORKFLOW_PLANNER_FAILED,
            )
            return {
                "routing": {
                    "plan": {
                        "action": plan.action,
                        "reason": plan.reason,
                        "validation": VALIDATION_PENDING,
                    },
                },
                "errors": self._state.state_errors(
                    state,
                    [f"{ERROR_WORKFLOW_PLANNER_FAILED_PREFIX}{type(exc).__name__}"],
                ),
            }

        request: dict[str, Any] = self._state.request_values_from_state(state)
        if (
            plan.action == ROUTE_SEARCH_JOBS
            and ROUTE_SEARCH_JOBS in self._state.completed_actions(state)
            and not request.get("refresh_requested")
        ):
            if request.get("assessment_requested") and self._state.jobs_bucket(
                state
            ).get("results"):
                plan = WorkflowPlan(
                    action=ROUTE_MATCH_JOBS,
                    reason=REASON_SEARCH_COMPLETE_ASSESSMENT,
                )
            else:
                plan = WorkflowPlan(
                    action=ROUTE_RESPOND,
                    reason=REASON_SEARCH_COMPLETE_PRESENT_RESULTS,
                )

        return {
            "routing": {
                "plan": {
                    "action": plan.action,
                    "reason": plan.reason,
                    "validation": VALIDATION_PENDING,
                },
            }
        }
