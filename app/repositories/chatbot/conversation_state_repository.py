"""Read and project the chatbot's conversation state.

Every accessor is a pure read over ``ConversationState``. Tolerating
pre-migration shapes (top-level ``targets`` / ``plan`` / ``selection`` buckets)
happens here so services only ever see the nested contract.
"""

from __future__ import annotations

from typing import Any, cast

from app.config.const.chatbot import AGENT_ACTIONS, REQUEST_VALUE_KEYS
from app.models.chatbot.literals import JobResponse, JobTask
from app.models.chatbot.schemas import ConversationMemory
from app.models.chatbot.state import ConversationState, ExecutionStepState
from app.services.chatbot.message_reader import MessageReader


class ConversationStateRepository:
    """Bucket accessors, request projection, and default state fragments."""

    def __init__(self, *, messages: MessageReader) -> None:
        self._messages = messages

    def cv_bucket(self, state: ConversationState) -> dict[str, Any]:
        value: Any = state.get("cv")
        return dict(value) if isinstance(value, dict) else {}

    def routing_bucket(self, state: ConversationState) -> dict[str, Any]:
        value: Any = state.get("routing")
        return dict(value) if isinstance(value, dict) else {}

    def request_bucket(self, state: ConversationState) -> dict[str, Any]:
        value: Any = self.routing_bucket(state).get("request")
        return dict(value) if isinstance(value, dict) else {}

    def request_values(self, request: dict[str, Any]) -> dict[str, Any]:
        """Project nested request state into a private runtime view."""
        goal: dict[str, Any] = request.get("goal") or {}
        job: dict[str, Any] = request.get("job") or {}
        role: dict[str, Any] = request.get("role") or {}
        assessment: dict[str, Any] = request.get("assessment") or {}
        score: dict[str, Any] = request.get("score") or {}
        review: dict[str, Any] = request.get("review") or {}
        cv: dict[str, Any] = request.get("cv") or {}
        context: dict[str, Any] = request.get("context") or {}
        return {
            "goal": goal.get("name") or "general_question",
            "goal_reason": goal.get("reason") or "",
            "decision_confidence": goal.get("confidence", 1.0),
            "job_source": job.get("source") or "none",
            "job_task": job.get("task") or "none",
            "job_response": job.get("response") or "none",
            "job_refresh": bool(job.get("refresh")),
            "job_input_text": job.get("input"),
            "score_requested": bool(score.get("requested")),
            "assessment_requested": bool(assessment.get("requested")),
            "show_score": bool(score.get("visible")),
            "refresh_requested": bool(job.get("refresh")),
            "match_detail_level": assessment.get("detail_level") or "summary",
            "role_constraints": list(role.get("constraints") or []),
            "role_evidence": role.get("evidence"),
            "role_source": role.get("source") or "none",
            "role_candidates": list(role.get("candidates") or []),
            "review_target_role": review.get("target_role"),
            "review_mode": review.get("mode") or "general",
            "review_focus": review.get("focus"),
            "review_mode_reason": review.get("reason"),
            "needs_cv_text": bool(cv.get("text_needed")),
            "needs_cv_features": bool(cv.get("features_needed")),
            "is_follow_up": bool(context.get("follow_up")),
            "scrape_request": dict(job.get("scrape") or {}),
        }

    def request_values_from_state(self, state: ConversationState) -> dict[str, Any]:
        return self.request_values(self.request_bucket(state))

    def request_job_task(self, state: ConversationState) -> JobTask:
        return cast(
            JobTask, self.request_values_from_state(state).get("job_task") or "none"
        )

    def request_job_response(self, state: ConversationState) -> JobResponse:
        return cast(
            JobResponse,
            self.request_values_from_state(state).get("job_response") or "none",
        )

    def request_show_score(self, state: ConversationState) -> bool:
        values: dict[str, Any] = self.request_values_from_state(state)
        return bool(values.get("show_score") or values.get("score_requested"))

    def non_request_fields(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return {
            key: item for key, item in value.items() if key not in REQUEST_VALUE_KEYS
        }

    def normalize_targets_state(self, value: dict[str, Any]) -> dict[str, Any]:
        """Read the nested target contract while tolerating pre-migration state."""
        nested_cv: Any = value.get("cv")
        nested_job: Any = value.get("job")
        if isinstance(nested_cv, dict) or isinstance(nested_job, dict):
            return {
                "cv": dict(nested_cv) if isinstance(nested_cv, dict) else {},
                "job": dict(nested_job) if isinstance(nested_job, dict) else {},
                "unresolved_references": list(value.get("unresolved_references") or []),
                "ambiguous": bool(
                    value.get("ambiguous", value.get("targets_ambiguous"))
                ),
            }
        selected_cv_ids: list[str] = [
            str(item).strip()
            for item in (value.get("selected_cv_ids") or [])
            if str(item).strip()
        ]
        selected_job_keys: list[str] = [
            str(item).strip()
            for item in (value.get("selected_job_keys") or [])
            if str(item).strip()
        ]
        return {
            "cv": {
                "scope": value.get("cv_target_scope") or "none",
                "ids": selected_cv_ids,
            },
            "job": {
                "scope": value.get("job_target_scope") or "none",
                "keys": selected_job_keys,
            },
            "unresolved_references": list(value.get("unresolved_references") or []),
            "ambiguous": bool(value.get("targets_ambiguous")),
        }

    def targets_bucket(self, state: ConversationState) -> dict[str, Any]:
        value: Any = self.routing_bucket(state).get("targets")
        if isinstance(value, dict):
            return self.normalize_targets_state(value)
        value = state.get("targets")
        if isinstance(value, dict):
            return self.normalize_targets_state(value)
        value = state.get("selection")
        return self.normalize_targets_state(value) if isinstance(value, dict) else {}

    def plan_bucket(self, state: ConversationState) -> dict[str, Any]:
        value: Any = self.routing_bucket(state).get("plan")
        if isinstance(value, dict):
            return dict(value)
        value = state.get("plan")
        if isinstance(value, dict):
            return dict(value)
        value = state.get("router")
        return dict(value) if isinstance(value, dict) else {}

    def router_bucket(self, state: ConversationState) -> dict[str, Any]:
        merged: dict[str, Any] = self.non_request_fields(state.get("router"))
        merged.update(self.request_values_from_state(state))
        merged.update(self.plan_bucket(state))
        plan: dict[str, Any] = self.plan_bucket(state)
        if plan.get("action") is not None:
            merged["route"] = plan.get("action")
        if plan.get("reason") is not None:
            merged["route_reason"] = plan.get("reason")
        if plan.get("validation") is not None:
            merged["plan_validation"] = plan.get("validation")
        return merged

    def selection_bucket(self, state: ConversationState) -> dict[str, Any]:
        merged: dict[str, Any] = self.non_request_fields(state.get("selection"))
        merged.update(self.request_values_from_state(state))
        targets: dict[str, Any] = self.targets_bucket(state)
        cv_targets: dict[str, Any] = targets.get("cv") or {}
        job_targets: dict[str, Any] = targets.get("job") or {}
        merged.update(
            {
                "cv_target_scope": cv_targets.get("scope") or "none",
                "selected_cv_ids": list(cv_targets.get("ids") or []),
                "selected_cv_id": (
                    cv_targets.get("ids")[0]
                    if len(cv_targets.get("ids") or []) == 1
                    else None
                ),
                "job_target_scope": job_targets.get("scope") or "none",
                "selected_job_keys": list(job_targets.get("keys") or []) or None,
                "unresolved_references": list(
                    targets.get("unresolved_references") or []
                ),
                "targets_ambiguous": bool(targets.get("ambiguous")),
            }
        )
        return merged

    def jobs_bucket(self, state: ConversationState) -> dict[str, Any]:
        value: Any = state.get("jobs")
        return dict(value) if isinstance(value, dict) else {}

    def action_results_bucket(self, state: ConversationState) -> dict[str, Any]:
        value: Any = state.get("action_results")
        return dict(value) if isinstance(value, dict) else {}

    def completed_actions(self, state: ConversationState) -> list[str]:
        """Return the valid agent actions completed during this user message."""
        return [
            action
            for action in self.plan_bucket(state).get("completed_actions") or []
            if isinstance(action, str) and action in AGENT_ACTIONS
        ]

    def execution_bucket(self, state: ConversationState) -> dict[str, Any]:
        value: Any = state.get("execution")
        return dict(value) if isinstance(value, dict) else {}

    def execution_steps(self, state: ConversationState) -> list[ExecutionStepState]:
        value: Any = self.execution_bucket(state).get("steps")
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    def default_request_fields(self) -> dict[str, Any]:
        return {
            "goal": {"name": "general_question", "reason": "", "confidence": 1.0},
            "job": {
                "task": "none",
                "response": "none",
                "source": "none",
                "input": None,
                "refresh": False,
                "scrape": {},
            },
            "role": {
                "constraints": [],
                "evidence": None,
                "source": "none",
                "candidates": [],
            },
            "assessment": {"requested": False, "detail_level": "summary"},
            "score": {"requested": False, "visible": False},
            "review": {
                "target_role": None,
                "mode": "general",
                "focus": None,
                "reason": None,
            },
            "cv": {"text_needed": False, "features_needed": False},
            "context": {"follow_up": False},
        }

    def default_target_fields(self) -> dict[str, Any]:
        return {
            "cv": {"scope": "none", "ids": []},
            "job": {"scope": "none", "keys": []},
            "unresolved_references": [],
            "ambiguous": False,
        }

    def default_selection_fields(self) -> dict[str, Any]:
        """Compatibility view for legacy helpers; active state uses request/targets."""
        return {**self.default_request_fields(), **self.default_target_fields()}

    def conversation_memory(self, state: ConversationState) -> dict[str, Any]:
        raw: Any = state.get("conversation_memory")
        if not isinstance(raw, dict):
            return {}
        try:
            return ConversationMemory.model_validate(raw).model_dump()
        except Exception:
            return {}

    def conversation_memory_cursor(self, state: ConversationState) -> int:
        raw: Any = state.get("conversation_memory_cursor", 0)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    def last_user_text(self, state: ConversationState) -> str:
        for message in reversed(state.get("messages") or []):
            if self._messages.message_role(message) in {"human", "user"}:
                return self._messages.message_text(message)
        return ""

    def state_errors(
        self, state: ConversationState, extra: list[str] | None = None
    ) -> list[str]:
        return list(state.get("errors") or []) + list(extra or [])

    def request_state_fields(self, value: dict[str, Any]) -> dict[str, Any]:
        return {
            "goal": {
                "name": value.get("goal") or "general_question",
                "reason": value.get("goal_reason") or value.get("reason") or "",
                "confidence": value.get("decision_confidence", 1.0),
            },
            "job": {
                "task": value.get("job_task") or value.get("task") or "none",
                "response": value.get("job_response")
                or value.get("response")
                or "none",
                "source": value.get("job_source") or "none",
                "input": value.get("job_input_text"),
                "refresh": bool(
                    value.get("job_refresh") or value.get("refresh_requested")
                ),
                "scrape": dict(value.get("scrape_request") or {}),
            },
            "role": {
                "constraints": list(value.get("role_constraints") or []),
                "evidence": value.get("role_evidence"),
                "source": value.get("role_source") or "none",
                "candidates": list(value.get("role_candidates") or []),
            },
            "assessment": {
                "requested": bool(value.get("assessment_requested")),
                "detail_level": value.get("match_detail_level") or "summary",
            },
            "score": {
                "requested": bool(value.get("score_requested")),
                "visible": bool(value.get("show_score")),
            },
            "review": {
                "target_role": value.get("review_target_role"),
                "mode": value.get("review_mode") or "general",
                "focus": value.get("review_focus"),
                "reason": value.get("review_mode_reason"),
            },
            "cv": {
                "text_needed": bool(value.get("needs_cv_text")),
                "features_needed": bool(value.get("needs_cv_features")),
            },
            "context": {"follow_up": bool(value.get("is_follow_up"))},
        }

    def target_state_fields(self, value: dict[str, Any]) -> dict[str, Any]:
        selected_cv_ids: list[str] = [
            str(item).strip()
            for item in (value.get("selected_cv_ids") or [])
            if str(item).strip()
        ]
        selected_job_keys: list[str] = [
            str(item).strip()
            for item in (value.get("selected_job_keys") or [])
            if str(item).strip()
        ]
        return {
            "cv": {
                "scope": value.get("cv_target_scope") or "none",
                "ids": selected_cv_ids,
            },
            "job": {
                "scope": value.get("job_target_scope") or "none",
                "keys": selected_job_keys,
            },
            "unresolved_references": list(value.get("unresolved_references") or []),
            "ambiguous": bool(value.get("targets_ambiguous")),
        }
