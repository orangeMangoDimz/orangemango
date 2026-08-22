"""Ingest uploaded CVs and reset per-turn routing state."""

from __future__ import annotations

from typing import Any

from app.config.const.chatbot import MAX_CV_DOCUMENTS, ROUTE_RESPOND
from app.config.const.chatbot_errors import (
    ERROR_CV_DOCUMENT_LIMIT_REACHED,
    ERROR_CV_UPLOAD_FAILED,
    ERROR_CV_UPLOAD_NO_READABLE_DOCUMENTS,
    ERROR_CV_UPLOAD_PAYLOAD_MISSING,
    REASON_AWAITING_REQUEST_ROUTING,
    VALIDATION_PENDING,
)
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.services.chatbot.upload_parser import UploadParser


class IngestService:
    """First node: normalize messages, decode uploads, reset routing buckets."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        cvs: CvStateRepository,
        uploads: UploadParser,
    ) -> None:
        self._state = state
        self._cvs = cvs
        self._uploads = uploads

    def ingest_input(self, state: ConversationState) -> dict[str, Any]:
        updates: dict[str, Any] = {
            "pending_cv_upload": None,
            "pending_cv_uploads": None,
            "input_error": False,
            "cv": {
                "needs_extraction": False,
            },
            "routing": {
                "request": self._state.default_request_fields(),
                "targets": self._state.default_target_fields(),
                "plan": {
                    "action": ROUTE_RESPOND,
                    "reason": REASON_AWAITING_REQUEST_ROUTING,
                    "validation": VALIDATION_PENDING,
                    "completed_actions": [],
                },
            },
            "execution": {
                "steps": [],
            },
            "response": None,
            "errors": [],
        }

        messages: list[Any] = list(state.get("messages") or [])
        message_updates: list[Any] = self._uploads.sanitize_file_messages(messages)
        cleared_uploads: list[Any] = self._uploads.clear_stashed_uploads(messages)
        if message_updates or cleared_uploads:
            merged_by_id: dict[str, Any] = {}
            ordered: list[Any] = []
            for message in [*message_updates, *cleared_uploads]:
                message_id: str = (
                    str(message.get("id"))
                    if isinstance(message, dict) and message.get("id") is not None
                    else str(getattr(message, "id", "") or "")
                )
                if message_id and message_id in merged_by_id:
                    continue
                if message_id:
                    merged_by_id[message_id] = message
                ordered.append(message)
            updates["messages"] = ordered

        pending_uploads: Any = state.get("pending_cv_uploads")
        if not isinstance(pending_uploads, list) or not pending_uploads:
            single: Any = state.get("pending_cv_upload")
            if isinstance(single, dict):
                pending_uploads = [single]
            else:
                pending_uploads = self._uploads.pending_uploads_from_messages(messages)
        if not pending_uploads:
            return updates

        if any(
            isinstance(item, dict) and item.get("missing_bytes")
            for item in pending_uploads
        ):
            return {
                **updates,
                "input_error": True,
                "errors": [ERROR_CV_UPLOAD_PAYLOAD_MISSING],
            }

        existing_documents: list[dict[str, Any]] = self._cvs.state_cv_documents(state)
        remaining_slots: int = max(0, MAX_CV_DOCUMENTS - len(existing_documents))
        if remaining_slots == 0:
            return {
                **updates,
                "input_error": True,
                "errors": [f"{ERROR_CV_DOCUMENT_LIMIT_REACHED}{MAX_CV_DOCUMENTS}"],
            }

        new_documents: list[dict[str, Any]] = []
        errors: list[str] = []
        for upload in pending_uploads[:remaining_slots]:
            try:
                new_documents.append(self._uploads.cv_document_from_upload(upload))
            except Exception as exc:
                filename: str = (
                    str(upload.get("filename") or "cv.pdf")
                    if isinstance(upload, dict)
                    else "cv.pdf"
                )
                errors.append(
                    f"{ERROR_CV_UPLOAD_FAILED}{filename}:{type(exc).__name__}"
                )

        if not new_documents:
            return {
                **updates,
                "input_error": True,
                "errors": errors or [ERROR_CV_UPLOAD_NO_READABLE_DOCUMENTS],
            }

        documents: list[dict[str, Any]] = [*existing_documents, *new_documents]
        updates.update(
            {
                "cv": {
                    "documents": documents,
                    "comparison": None,
                    **self._cvs.cv_needs_extraction_update(documents)["cv"],
                },
                "errors": errors,
            }
        )
        return updates
