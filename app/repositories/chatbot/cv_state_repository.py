"""Read and resolve CV documents held in conversation state."""

from __future__ import annotations

import hashlib
from typing import Any

from app.config.const.chatbot import CV_FEATURE_INTENTS, PDF_UPLOAD_MARKER
from app.models.chatbot.literals import RouteName
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.services.chatbot.text_utils import TextUtils


class CvStateRepository:
    """CV document lookup, extraction status, and target resolution."""

    def __init__(self, *, state: ConversationStateRepository) -> None:
        self._state = state

    def state_cv_documents(self, state: ConversationState) -> list[dict[str, Any]]:
        documents: Any = self._state.cv_bucket(state).get("documents")
        if isinstance(documents, list) and documents:
            return [dict(item) for item in documents if isinstance(item, dict)]
        return []

    def cv_needs_extraction_update(
        self, documents: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "cv": {
                "needs_extraction": any(
                    (doc.get("cv_text") or "").strip() and not doc.get("cv_features")
                    for doc in documents
                )
            }
        }

    def cvs_need_extraction(self, state: ConversationState) -> bool:
        documents: list[dict[str, Any]] = self.state_cv_documents(state)
        return bool(
            self._state.cv_bucket(state).get("needs_extraction")
            or any(
                (doc.get("cv_text") or "").strip() and not doc.get("cv_features")
                for doc in documents
            )
        )

    def intent_requires_cv_features(self, state: ConversationState) -> bool:
        router: dict[str, Any] = self._state.router_bucket(state)
        route: RouteName = router.get("route") or "respond"
        if route in CV_FEATURE_INTENTS:
            return True
        return route == "respond" and bool(router.get("needs_cv_features"))

    def cv_feature_summary(self, features: dict[str, Any] | None) -> dict[str, Any]:
        features: dict[str, Any] = features or {}
        return {
            key: features.get(key)
            for key in (
                "role_tags",
                "skill_names",
                "seniority",
                "years_of_experience",
                "current_location",
            )
            if features.get(key) not in (None, [], "")
        }

    def extracted_cv_documents(self, state: ConversationState) -> list[dict[str, Any]]:
        """Return uploaded CVs that have usable extracted features."""
        return [
            document
            for document in self.state_cv_documents(state)
            if (document.get("cv_text") or "").strip() and document.get("cv_features")
        ]

    def resolve_selected_cv(self, state: ConversationState) -> dict[str, Any] | None:
        documents: list[dict[str, Any]] = self.resolve_selected_cvs(state)
        return documents[0] if documents else None

    def resolve_selected_cvs(self, state: ConversationState) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = self.extracted_cv_documents(state)
        if not documents:
            return []
        selected_ids: Any = self._state.selection_bucket(state).get("selected_cv_ids")
        if isinstance(selected_ids, list) and selected_ids:
            wanted: set[str] = {
                str(item).strip() for item in selected_ids if str(item).strip()
            }
            selected: list[dict[str, Any]] = [
                document
                for document in documents
                if str(document.get("id") or "") in wanted
            ]
            return selected or documents
        selected_id: str = str(
            self._state.selection_bucket(state).get("selected_cv_id") or ""
        ).strip()
        if not selected_id:
            return documents
        selected: list[dict[str, Any]] = [
            document
            for document in documents
            if str(document.get("id") or "") == selected_id
        ]
        return selected or documents

    def cv_version(self, document: dict[str, Any] | None) -> str:
        if not isinstance(document, dict):
            return ""
        text: str = (document.get("cv_text") or "").strip()
        if text:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
        features: Any = document.get("cv_features")
        if isinstance(features, dict) and features:
            return TextUtils.canonical_json_hash(features)
        return str(document.get("id") or "")

    def is_cv_upload_turn(self, state: ConversationState) -> bool:
        return PDF_UPLOAD_MARKER in self._state.last_user_text(state)

    def unambiguous_extracted_cv(
        self, state: ConversationState
    ) -> dict[str, Any] | None:
        documents: list[dict[str, Any]] = self.extracted_cv_documents(state)
        selected_id: str = str(
            self._state.selection_bucket(state).get("selected_cv_id") or ""
        ).strip()
        if selected_id:
            matched: list[dict[str, Any]] = [
                document
                for document in documents
                if str(document.get("id") or "") == selected_id
            ]
            return matched[0] if len(matched) == 1 else None
        if len(documents) == 1:
            return documents[0]
        return None

    def existing_cv_review(self, state: ConversationState) -> dict[str, Any] | None:
        review: Any = self._state.cv_bucket(state).get("review")
        if isinstance(review, dict):
            return review
        for doc in self.state_cv_documents(state):
            candidate: Any = doc.get("cv_review")
            if isinstance(candidate, dict):
                return candidate
        return None
