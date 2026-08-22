"""Run the CV extraction, review, and comparison workflows."""

from __future__ import annotations

import json
from typing import Any

from app.config.const.chatbot import MAX_CV_DOCUMENTS
from app.config.const.chatbot_errors import (
    ERROR_CV_COMPARISON_FAILED_PREFIX,
    ERROR_CV_COMPARISON_REQUIRES_TWO_DOCUMENTS,
    ERROR_CV_EXTRACTION_FAILED,
    ERROR_CV_EXTRACTION_INVALID,
    ERROR_CV_EXTRACTION_REQUIRED_FOR_REVIEW,
    ERROR_CV_REVIEW_FAILED_PREFIX,
    ERROR_CV_UPLOAD_REQUIRED,
    STATUS_UNAVAILABLE,
)
from app.config.const.chatbot_prompts import COMPARE_CVS_PROMPT, CV_PROFILES_DATA_HEADER
from app.models.chat_model import ChatModel
from app.models.chatbot.schemas import CvComparisonResult
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.repositories.chatbot.subgraph_repository import SubgraphRepository
from app.services.chatbot.result_projection_service import ResultProjectionService


class CvWorkflowService:
    """CV extraction plus the review and comparison actions built on it."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        cvs: CvStateRepository,
        subgraphs: SubgraphRepository,
        projection: ResultProjectionService,
        chat_model: ChatModel,
    ) -> None:
        self._state = state
        self._cvs = cvs
        self._subgraphs = subgraphs
        self._projection = projection
        self._chat_model = chat_model
        self._review_graph: Any = subgraphs.build_cv_review_graph(
            chat_model=chat_model,
        )

    def missing_cv_update(self, state: ConversationState) -> dict[str, Any]:
        return {
            "cv": {"needs_extraction": False},
            "errors": self._state.state_errors(
                state,
                [ERROR_CV_UPLOAD_REQUIRED],
            ),
        }


    async def run_cv_subagent(self, state: ConversationState) -> dict[str, Any]:
        documents: list[dict[str, Any]] = self._cvs.state_cv_documents(state)
        if not documents:
            return self.missing_cv_update(state)

        updated_documents: list[dict[str, Any]] = []
        errors: list[str] = []
        for document in documents:
            cv_text: str = (document.get("cv_text") or "").strip()
            if not cv_text:
                updated_documents.append(dict(document))
                continue
            if document.get("cv_features"):
                updated_documents.append(dict(document))
                continue
            filename: str = str(document.get("filename") or "cv.pdf")
            try:
                result: dict[str, Any] = await self._subgraphs.cv_extraction.ainvoke(
                    {"cv_text": cv_text}
                )
                compact: dict[str, Any] = self._projection.compact_cv_result(result)
                if compact.get("validation_status") != "valid":
                    updated_documents.append(
                        {
                            **document,
                            "cv_result": compact,
                            "cv_features": None,
                        }
                    )
                    errors.append(
                        f"{ERROR_CV_EXTRACTION_INVALID}{filename}"
                    )
                    continue
                updated_documents.append(
                    {
                        **document,
                        "cv_result": compact,
                        "cv_features": compact.get("matching_features"),
                    }
                )
            except Exception as exc:
                updated_documents.append(
                    {
                        **document,
                        "cv_result": None,
                        "cv_features": None,
                    }
                )
                errors.append(f"{ERROR_CV_EXTRACTION_FAILED}{filename}:{type(exc).__name__}")

        update: dict[str, Any] = {
            "cv": {
                "documents": updated_documents,
                **self._cvs.cv_needs_extraction_update(updated_documents)["cv"],
            }
        }
        if errors:
            update["errors"] = self._state.state_errors(state, errors)
        return update


    async def handle_missing_cv(self, state: ConversationState) -> dict[str, Any]:
        return self.missing_cv_update(state)


    async def run_cv_review(self, state: ConversationState) -> dict[str, Any]:
        documents: list[dict[str, Any]] = self._cvs.state_cv_documents(state)
        if not documents:
            return self.missing_cv_update(state)

        selection: dict[str, Any] = self._state.selection_bucket(state)
        target: dict[str, Any] | None = self._cvs.resolve_selected_cv(state)
        if target is None:
            return {
                "cv": {"review": None},
                "errors": self._state.state_errors(
                    state,
                    [ERROR_CV_EXTRACTION_REQUIRED_FOR_REVIEW],
                ),
            }

        try:
            result: dict[str, Any] = await self._review_graph.ainvoke(
                {
                    "cv_text": (target.get("cv_text") or "").strip(),
                    "cv_features": target.get("cv_features"),
                    "target_role": selection.get("review_target_role"),
                    "review_mode": selection.get("review_mode") or "general",
                    "review_focus": selection.get("review_focus"),
                }
            )
            review: Any = result.get("cv_review")
            if not isinstance(review, dict):
                raise ValueError("CV review graph returned no review result")
            updated_documents: list[dict[str, Any]] = [
                {**doc, "cv_review": review}
                if doc.get("id") == target.get("id")
                else dict(doc)
                for doc in documents
            ]
            return {
                "cv": {
                    "documents": updated_documents,
                    "review": review,
                    **self._cvs.cv_needs_extraction_update(updated_documents)["cv"],
                }
            }
        except Exception as exc:
            review: dict[str, Any] = {
                "status": STATUS_UNAVAILABLE,
                "mode": selection.get("review_mode") or "general",
                "focus": selection.get("review_focus"),
                "target_role": selection.get("review_target_role"),
                "overall_score": None,
                "applicable_weight": 0,
                "criteria": [],
                "feedback": [],
                "deterministic_signals": {},
                "validation_errors": [f"{ERROR_CV_REVIEW_FAILED_PREFIX}{type(exc).__name__}"],
            }
            updated_documents: list[dict[str, Any]] = [
                {**doc, "cv_review": review}
                if doc.get("id") == target.get("id")
                else dict(doc)
                for doc in documents
            ]
            return {
                "cv": {
                    "documents": updated_documents,
                    "review": review,
                    **self._cvs.cv_needs_extraction_update(updated_documents)["cv"],
                },
                "errors": self._state.state_errors(
                    state,
                    [f"{ERROR_CV_REVIEW_FAILED_PREFIX}{type(exc).__name__}"],
                ),
            }

    async def run_cv_comparison(self, state: ConversationState) -> dict[str, Any]:
        documents: list[dict[str, Any]] = self._cvs.extracted_cv_documents(state)
        if len(documents) < 2:
            return {
                "cv": {"comparison": None},
                "errors": self._state.state_errors(
                    state,
                    [ERROR_CV_COMPARISON_REQUIRES_TWO_DOCUMENTS],
                ),
            }

        profiles: list[dict[str, Any]] = [
            {
                "filename": str(doc.get("filename") or "cv.pdf"),
                "features": self._cvs.cv_feature_summary(
                    doc.get("cv_features")
                    if isinstance(doc.get("cv_features"), dict)
                    else None
                ),
            }
            for doc in documents[:MAX_CV_DOCUMENTS]
        ]
        comparer: Any = self._chat_model.structured(CvComparisonResult)
        try:
            comparison: CvComparisonResult = await comparer.ainvoke(
                [
                    {"role": "system", "content": COMPARE_CVS_PROMPT},
                    {
                        "role": "user",
                        "content": CV_PROFILES_DATA_HEADER
                        + json.dumps(profiles, ensure_ascii=False),
                    },
                ]
            )
            return {"cv": {"comparison": comparison.model_dump()}}
        except Exception as exc:
            return {
                "cv": {"comparison": None},
                "errors": self._state.state_errors(
                    state,
                    [f"{ERROR_CV_COMPARISON_FAILED_PREFIX}{type(exc).__name__}"],
                ),
            }
