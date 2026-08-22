"""Read and resolve job results, targets, and the active job goal."""

from __future__ import annotations

from typing import Any

from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.services.chatbot.job_key_utils import JobKeyUtils
from app.services.chatbot.text_utils import TextUtils


class JobStateRepository:
    """Job result selection plus active-goal construction and validation."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        cvs: CvStateRepository,
    ) -> None:
        self._state = state
        self._cvs = cvs

    def resolve_selected_jobs(self, state: ConversationState) -> list[dict[str, Any]]:
        job_results: Any = self._state.jobs_bucket(state).get("results") or []
        results: list[dict[str, Any]] = [
            item
            for item in job_results
            if isinstance(item, dict) and item.get("validation_status") == "valid"
        ]
        raw_keys: Any = self._state.selection_bucket(state).get("selected_job_keys")
        if isinstance(raw_keys, list) and raw_keys:
            wanted: set[str] = {
                str(key).strip() for key in raw_keys if str(key).strip()
            }
            if wanted:
                selected: list[dict[str, Any]] = []
                for index, item in enumerate(job_results):
                    if (
                        not isinstance(item, dict)
                        or item.get("validation_status") != "valid"
                    ):
                        continue
                    if JobKeyUtils.job_selection_key(item, index) in wanted:
                        selected.append(item)
                return selected or results
        active_keys: Any = self._state.jobs_bucket(state).get("active_job_keys")
        if isinstance(active_keys, list) and active_keys:
            wanted = {str(key).strip() for key in active_keys if str(key).strip()}
            if wanted:
                selected = []
                for index, item in enumerate(job_results):
                    if (
                        not isinstance(item, dict)
                        or item.get("validation_status") != "valid"
                    ):
                        continue
                    if JobKeyUtils.job_selection_key(item, index) in wanted:
                        selected.append(item)
                return selected
        return results

    def build_active_job_goal(
        self,
        *,
        source: str,
        role_constraints: list[str],
        cv_id: str | None,
        cv_version: str | None,
        originating_turn: str,
        invalidated: bool = False,
        invalidation_reason: str | None = None,
    ) -> dict[str, Any]:
        constraints: list[str] = TextUtils.normalize_role_constraints(role_constraints)
        identity: dict[str, Any] = {
            "source": source,
            "role_constraints": constraints,
            "cv_id": cv_id or None,
            "cv_version": cv_version or None,
        }
        return {
            "id": TextUtils.canonical_json_hash(identity),
            "source": source,
            "role_constraints": constraints,
            "cv_id": cv_id or None,
            "cv_version": cv_version or None,
            "originating_turn": originating_turn,
            "invalidated": invalidated,
            "invalidation_reason": invalidation_reason,
        }

    def active_job_goal(self, state: ConversationState) -> dict[str, Any] | None:
        value: Any = self._state.jobs_bucket(state).get("active_job_goal")
        return dict(value) if isinstance(value, dict) else None

    def pending_match_request(self, state: ConversationState) -> dict[str, Any] | None:
        value: Any = self._state.jobs_bucket(state).get("pending_match")
        return dict(value) if isinstance(value, dict) else None

    def active_job_goal_is_usable(self, state: ConversationState) -> bool:
        goal: dict[str, Any] | None = self.active_job_goal(state)
        if goal is None or goal.get("invalidated") or not goal.get("role_constraints"):
            return False
        document: dict[str, Any] | None = self._cvs.unambiguous_extracted_cv(state)
        if document is None:
            return False
        goal_cv_id: str = str(goal.get("cv_id") or "").strip()
        if goal_cv_id and goal_cv_id != str(document.get("id") or ""):
            return False
        goal_version: str = str(goal.get("cv_version") or "")
        if goal_version and goal_version != self._cvs.cv_version(document):
            return False
        return True

    def job_results_for_display(self, state: ConversationState) -> list[dict[str, Any]]:
        jobs_state: dict[str, Any] = self._state.jobs_bucket(state)
        active_keys: Any = jobs_state.get("active_job_keys")
        selected_keys: Any = self._state.selection_bucket(state).get(
            "selected_job_keys"
        )
        if (isinstance(active_keys, list) and active_keys) or (
            isinstance(selected_keys, list) and selected_keys
        ):
            return self.resolve_selected_jobs(state)
        return [
            item for item in (jobs_state.get("results") or []) if isinstance(item, dict)
        ]
