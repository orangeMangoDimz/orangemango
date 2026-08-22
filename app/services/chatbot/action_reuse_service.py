"""Fingerprint executed actions so identical work is reused, not repeated."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.config.const.chatbot import SEARCH_RESULT_TTL_SECONDS
from app.models.chatbot.literals import AgentAction
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.repositories.chatbot.job_state_repository import JobStateRepository
from app.services.chatbot.job_key_utils import JobKeyUtils
from app.services.chatbot.text_utils import TextUtils


class ActionReuseService:
    """Decide whether a stored action result still satisfies the current request."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        cvs: CvStateRepository,
        jobs: JobStateRepository,
    ) -> None:
        self._state = state
        self._cvs = cvs
        self._jobs = jobs

    def normalize_scrape_request(
        self, request: dict[str, Any] | None
    ) -> dict[str, Any]:
        raw: dict[str, Any] = request if isinstance(request, dict) else {}
        keywords: list[str] = sorted(
            {
                TextUtils.normalize_fingerprint_text(item)
                for item in (raw.get("keywords") or [])
                if TextUtils.normalize_fingerprint_text(item)
            }
        )
        sites: list[str] = sorted(
            {
                TextUtils.normalize_fingerprint_text(item)
                for item in (raw.get("sites") or [])
                if TextUtils.normalize_fingerprint_text(item)
            }
        )
        return {
            "keywords": keywords,
            "sites": sites,
            "max_age_hours": raw.get("max_age_hours"),
        }

    def search_result_is_fresh(
        self,
        entry: dict[str, Any] | None,
        *,
        now: datetime | None = None,
    ) -> bool:
        if not isinstance(entry, dict):
            return False
        executed: datetime | None = TextUtils.parse_executed_at(
            entry.get("executed_at")
        )
        if executed is None:
            return False
        current: datetime = now or datetime.now(timezone.utc)
        return (current - executed).total_seconds() <= SEARCH_RESULT_TTL_SECONDS

    def stored_action_result(
        self,
        state: ConversationState,
        action: str,
    ) -> dict[str, Any] | None:
        entry: Any = self._state.action_results_bucket(state).get(action)
        return dict(entry) if isinstance(entry, dict) else None

    def action_fingerprint(self, action: str, state: ConversationState) -> str | None:
        selection: dict[str, Any] = self._state.selection_bucket(state)
        if action == "review_cv":
            target: dict[str, Any] | None = self._cvs.resolve_selected_cv(state)
            if target is None:
                documents: list[dict[str, Any]] = self._cvs.extracted_cv_documents(
                    state
                ) or self._cvs.state_cv_documents(state)
                target = documents[0] if documents else None
            if target is None:
                return None
            return TextUtils.canonical_json_hash(
                {
                    "action": "review_cv",
                    "cv_id": str(target.get("id") or ""),
                    "cv_version": self._cvs.cv_version(target),
                    "mode": selection.get("review_mode") or "general",
                    "focus": TextUtils.normalize_fingerprint_text(
                        selection.get("review_focus")
                    ),
                    "target_role": TextUtils.normalize_fingerprint_text(
                        selection.get("review_target_role")
                    ),
                }
            )
        if action == "compare_cvs":
            documents = self._cvs.extracted_cv_documents(state)
            if len(documents) < 2:
                return None
            return TextUtils.canonical_json_hash(
                {
                    "action": "compare_cvs",
                    "cvs": [
                        {
                            "id": str(doc.get("id") or ""),
                            "version": self._cvs.cv_version(doc),
                        }
                        for doc in documents
                    ],
                }
            )
        if action == "extract_job":
            text: str = TextUtils.normalize_fingerprint_text(
                selection.get("job_input_text") or self._state.last_user_text(state)
            )
            if not text:
                return None
            return TextUtils.canonical_json_hash(
                {"action": "extract_job", "text": text}
            )
        if action == "search_jobs":
            request: dict[str, Any] = self.normalize_scrape_request(
                self._state.jobs_bucket(state).get("scrape_request")
            )
            return TextUtils.canonical_json_hash({"action": "search_jobs", **request})
        if action == "match_jobs":
            cvs: list[dict[str, Any]] = self._cvs.resolve_selected_cvs(state)
            jobs: list[dict[str, Any]] = self._jobs.resolve_selected_jobs(state)
            if not cvs or not jobs:
                return None
            payload: dict[str, Any] = {
                "action": "match_jobs",
                "cvs": [
                    {
                        "id": str(doc.get("id") or ""),
                        "version": self._cvs.cv_version(doc),
                    }
                    for doc in cvs
                ],
                "jobs": [
                    {
                        "key": JobKeyUtils.job_selection_key(item, index),
                        "version": JobKeyUtils.job_content_version(item),
                    }
                    for index, item in enumerate(jobs)
                ],
            }
            if selection.get("job_source") == "search":
                payload["search"] = self.normalize_scrape_request(
                    self._state.jobs_bucket(state).get("scrape_request")
                )
            return TextUtils.canonical_json_hash(payload)
        return None

    def action_result_is_reusable(
        self,
        state: ConversationState,
        action: str,
        fingerprint: str | None,
        *,
        now: datetime | None = None,
    ) -> bool:
        if not fingerprint:
            return False
        entry: dict[str, Any] | None = self.stored_action_result(state, action)
        if entry is None or entry.get("fingerprint") != fingerprint:
            return False
        snapshot: Any = entry.get("snapshot")
        if not isinstance(snapshot, dict) or not snapshot:
            return False
        if action == "search_jobs" and not self.search_result_is_fresh(entry, now=now):
            return False
        return True

    def current_search_is_reusable(self, state: ConversationState) -> bool:
        return self.action_result_is_reusable(
            state,
            "search_jobs",
            self.action_fingerprint("search_jobs", state),
        )

    def reusable_action_snapshot(
        self,
        action: AgentAction,
        update: dict[str, Any],
        state: ConversationState,
    ) -> dict[str, Any] | None:
        if action == "review_cv":
            review: Any = (update.get("cv") or {}).get("review")
            if not isinstance(review, dict):
                return None
            if review.get("status") == "unavailable":
                return None
            if not (review.get("feedback") or review.get("overall_score") is not None):
                return None
            return {"cv": {"review": review}}
        if action == "compare_cvs":
            comparison: Any = (update.get("cv") or {}).get("comparison")
            if not isinstance(comparison, dict):
                return None
            if not comparison.get("overview") or not comparison.get("candidates"):
                return None
            return {"cv": {"comparison": comparison}}
        if action == "extract_job":
            jobs_update: dict[str, Any] = (
                dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
            )
            results: list[dict[str, Any]] = [
                item
                for item in (jobs_update.get("results") or [])
                if isinstance(item, dict)
            ]
            latest: dict[str, Any] | None = results[-1] if results else None
            if latest is None or latest.get("validation_status") != "valid":
                return None
            return {
                "jobs": {
                    "results": results,
                    "active_job_keys": [
                        JobKeyUtils.job_selection_key(latest, max(len(results) - 1, 0))
                    ],
                }
            }
        if action == "search_jobs":
            jobs_update = (
                dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
            )
            results = [
                item
                for item in (jobs_update.get("results") or [])
                if isinstance(item, dict)
            ]
            if not results:
                return None
            keys: Any = jobs_update.get("active_job_keys")
            if not isinstance(keys, list) or not keys:
                keys = [
                    JobKeyUtils.job_selection_key(item, index)
                    for index, item in enumerate(results)
                ]
            return {
                "jobs": {
                    "results": results,
                    "active_job_keys": keys,
                    "scrape_total": jobs_update.get("scrape_total", len(results)),
                    "scrape_truncated": bool(jobs_update.get("scrape_truncated")),
                }
            }
        if action == "match_jobs":
            jobs_update = (
                dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
            )
            matches: list[dict[str, Any]] = [
                item
                for item in (jobs_update.get("matches") or [])
                if isinstance(item, dict)
            ]
            if not matches:
                return None
            return {"jobs": {"matches": matches}}
        return None
