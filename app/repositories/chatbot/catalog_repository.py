"""Build the CV, job, and match catalogs the routing prompts select against."""

from __future__ import annotations

from typing import Any

from app.config.const.chatbot import MAX_CV_DOCUMENTS
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.repositories.chatbot.job_state_repository import JobStateRepository
from app.services.chatbot.job_key_utils import JobKeyUtils


class RoutingCatalogRepository:
    """Catalog projections of the CVs, jobs, and matches available to routing."""

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

    def routing_catalogs(self, state: ConversationState) -> dict[str, Any]:
        documents: list[dict[str, Any]] = self._cvs.state_cv_documents(state)
        cvs: list[dict[str, Any]] = [
            {
                "id": str(document.get("id") or ""),
                "filename": str(document.get("filename") or "cv.pdf"),
            }
            for document in documents[:MAX_CV_DOCUMENTS]
            if str(document.get("id") or "").strip()
        ]
        jobs: list[dict[str, Any]] = []
        job_keys: set[str] = set()
        for index, item in enumerate(
            self._state.jobs_bucket(state).get("results") or []
        ):
            if not isinstance(item, dict):
                continue
            key: str = JobKeyUtils.job_selection_key(item, index)
            job_keys.add(key)
            entry: dict[str, Any] = {"key": key, "row": index + 1}
            card: Any = item.get("job_card")
            if isinstance(card, dict):
                for field in ("title", "company"):
                    if card.get(field):
                        entry[field] = card[field]
            jobs.append(entry)

        matches: list[dict[str, Any]] = []
        match_keys: set[str] = set()
        for item in self._state.jobs_bucket(state).get("matches") or []:
            if not isinstance(item, dict):
                continue
            key: str = str(item.get("job_key") or "").strip()
            if not key or key in match_keys:
                continue
            match_keys.add(key)
            entry = {"key": key, "row": len(matches) + 1}
            card = item.get("job_card")
            if isinstance(card, dict):
                for field in ("title", "company"):
                    if card.get(field):
                        entry[field] = card[field]
            matches.append(entry)

        return {
            "cvs": cvs,
            "jobs": jobs,
            "matches": matches,
            "cv_ids": {item["id"] for item in cvs},
            "job_keys": job_keys,
            "match_keys": match_keys,
        }

    def routing_cv_profiles(self, state: ConversationState) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for document in self._cvs.extracted_cv_documents(state)[:MAX_CV_DOCUMENTS]:
            features: Any = document.get("cv_features")
            if not isinstance(features, dict):
                continue
            summary: dict[str, Any] = self._cvs.cv_feature_summary(features)
            profiles.append(
                {
                    "id": str(document.get("id") or ""),
                    "filename": str(document.get("filename") or "cv.pdf"),
                    **summary,
                }
            )
        return profiles
