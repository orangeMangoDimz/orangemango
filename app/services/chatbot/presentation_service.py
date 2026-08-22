"""Assemble the structured presentation payload handed to the response model."""

from __future__ import annotations

from typing import Any

from app.config.const.chatbot import (
    ROUTE_EXTRACT_JOB,
    ROUTE_MATCH_JOBS,
    ROUTE_RESPOND,
    ROUTE_SEARCH_JOBS,
)
from app.config.const.chatbot_errors import REVIEW_SCORE_SCALE, STATUS_UNAVAILABLE
from app.models.chatbot.literals import JobResponse
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.repositories.chatbot.job_state_repository import JobStateRepository
from app.services.chatbot.match_presentation_service import MatchPresentationService
from app.services.chatbot.result_projection_service import ResultProjectionService
from app.services.chatbot.text_utils import TextUtils


class PresentationService:
    """Public projections of review, comparison, job, and match state."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        cvs: CvStateRepository,
        jobs: JobStateRepository,
        projection: ResultProjectionService,
        matches: MatchPresentationService,
    ) -> None:
        self._state = state
        self._cvs = cvs
        self._jobs = jobs
        self._projection = projection
        self._matches = matches

    def public_presentation_intent(self, state: ConversationState) -> str:
        response: JobResponse = self._state.request_job_response(state)
        if response == "list":
            return "list"
        if response == "summary":
            return "summary"
        if response == "recommendation":
            return "recommendation"
        if response in {"explanation", "details"}:
            return response
        return "none"

    def match_identity(self, item: dict[str, Any], index: int) -> str:
        key: str = str(item.get("job_key") or "").strip()
        if key:
            return key
        card: dict[str, Any] = (
            item.get("job_card") if isinstance(item.get("job_card"), dict) else {}
        )
        return str(card.get("url") or f"match:{index}").strip()

    def search_role_label(self, state: ConversationState) -> str:
        goal: dict[str, Any] | None = self._jobs.active_job_goal(state)
        if goal is None:
            return ""
        return ", ".join(
            TextUtils.display_role_constraints(list(goal.get("role_constraints") or []))
        )

    def public_job_cards(self, state: ConversationState) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        for item in self._jobs.job_results_for_display(state):
            card: Any = item.get("job_card") if isinstance(item, dict) else None
            slim: dict[str, Any] = self._projection.slim_job_card(
                card if isinstance(card, dict) else None,
                include_description=False,
            )
            slim.pop("company", None)
            if slim:
                jobs.append(slim)
        return jobs

    def job_cards_for_current_request(
        self, state: ConversationState
    ) -> list[dict[str, Any]]:
        response: JobResponse = self._state.request_job_response(state)
        if response != "list":
            return []
        route: Any = self._state.router_bucket(state).get("route")
        if route in {ROUTE_SEARCH_JOBS, ROUTE_EXTRACT_JOB, ROUTE_RESPOND}:
            return self.public_job_cards(state)
        return []

    def public_assessment(
        self,
        state: ConversationState,
        *,
        show_score: bool,
    ) -> dict[str, Any] | None:
        response: JobResponse = self._state.request_job_response(state)
        if response not in {"summary", "recommendation", "explanation", "details"}:
            return None
        matches: list[dict[str, Any]] = [
            item
            for item in (self._state.jobs_bucket(state).get("matches") or [])
            if isinstance(item, dict)
        ]
        if response == "summary":
            if not matches and ROUTE_MATCH_JOBS not in self._state.completed_actions(
                state
            ):
                return None
            detail_level: Any = self._state.selection_bucket(state).get(
                "match_detail_level"
            )
            if detail_level not in {"summary", "full"}:
                detail_level = "summary"
            return self._matches.build_public_match_summary(
                matches,
                show_score=show_score,
                detail_level=detail_level,
            )

        detail_level: Any = self._state.selection_bucket(state).get(
            "match_detail_level"
        )
        if detail_level not in {"summary", "full"}:
            detail_level = "summary"

        if response == "recommendation":
            if not matches:
                return {"status": STATUS_UNAVAILABLE}
            return self._matches.build_public_match_recommendation(
                matches,
                show_score=show_score,
                detail_level=detail_level,
            )

        selected_keys: Any = self._state.selection_bucket(state).get(
            "selected_job_keys"
        )
        wanted: list[str] = [
            str(key).strip() for key in (selected_keys or []) if str(key).strip()
        ]
        if len(wanted) != 1:
            return None
        selected = [
            item
            for index, item in enumerate(matches)
            if self.match_identity(item, index) == wanted[0]
        ]
        if not selected:
            return None
        return self._matches.build_public_match_selected(
            selected,
            selected_key=wanted[0],
            show_score=show_score,
            detail_level=detail_level,
        )

    def public_review_payload(self, state: ConversationState) -> dict[str, Any] | None:
        review: Any = self._state.cv_bucket(state).get("review")
        if not isinstance(review, dict):
            selected: dict[str, Any] | None = self._cvs.resolve_selected_cv(state)
            if isinstance(selected, dict):
                review = selected.get("cv_review")
        slim: dict[str, Any] | None = self._projection.slim_review_result(
            review if isinstance(review, dict) else None
        )
        if slim is None or not slim.get("feedback"):
            return None
        payload: dict[str, Any] = {
            "mode": slim.get("mode"),
            "focus": slim.get("focus"),
            "target_role": slim.get("target_role"),
            "feedback": slim.get("feedback") or [],
        }
        if slim.get("overall_score") is not None:
            payload["overall_score"] = slim.get("overall_score")
            payload["score_scale"] = REVIEW_SCORE_SCALE
        return payload

    def public_comparison_payload(
        self, state: ConversationState
    ) -> dict[str, Any] | None:
        return self._projection.slim_comparison_result(
            self._state.cv_bucket(state).get("comparison")
            if isinstance(self._state.cv_bucket(state).get("comparison"), dict)
            else None
        )

    def public_extracted_job(self, state: ConversationState) -> dict[str, Any] | None:
        route: Any = self._state.router_bucket(state).get("route")
        if route not in {ROUTE_EXTRACT_JOB, ROUTE_RESPOND}:
            return None
        results: list[dict[str, Any]] = self._jobs.job_results_for_display(state)
        if not results:
            return None
        selected_keys: Any = self._state.selection_bucket(state).get(
            "selected_job_keys"
        )
        if route == ROUTE_RESPOND and len(results) > 1 and not selected_keys:
            return None
        latest: dict[str, Any] = results[-1]
        card: dict[str, Any] = self._projection.slim_job_card(
            latest.get("job_card") if isinstance(latest.get("job_card"), dict) else None
        )
        extract: Any = latest.get("extract")
        if not isinstance(extract, dict):
            return card or None
        payload: dict[str, Any] = dict(card)
        raw_content: str = TextUtils.short_text(extract.get("raw_content") or "", 6000)
        if raw_content:
            payload["description"] = raw_content
        responsibilities: list[str] = [
            str(value).strip()
            for value in (extract.get("responsibilities") or [])
            if str(value).strip()
        ][:12]
        if responsibilities:
            payload["responsibilities"] = responsibilities
        for source_key, public_key in (
            ("required_skills", "required_skills"),
            ("preferred_skills", "preferred_skills"),
        ):
            names: list[str] = []
            for value in extract.get(source_key) or []:
                if isinstance(value, dict):
                    name: str = str(
                        value.get("name")
                        or value.get("normalized_name")
                        or value.get("raw_name")
                        or ""
                    ).strip()
                else:
                    name = str(value or "").strip()
                if name and name not in names:
                    names.append(name)
            if names:
                payload[public_key] = names[:12]
        return payload or None

    def performed_actions_payload(
        self, state: ConversationState
    ) -> list[dict[str, Any]]:
        performed: list[dict[str, Any]] = []
        for step in self._state.execution_steps(state):
            destination: Any = step.get("to")
            if not isinstance(destination, dict):
                continue
            item: dict[str, Any] = {
                "action": step.get("action"),
                "status": step.get("status"),
                "args": dict(destination.get("args") or {}),
                "result": dict(destination.get("result") or {}),
            }
            context: Any = step.get("context")
            if isinstance(context, dict) and context.get("args_source"):
                item["args_source"] = context["args_source"]
            if step.get("error"):
                item["error"] = step["error"]
            performed.append(item)
        return performed

    def presentation_payload(self, state: ConversationState) -> dict[str, Any]:
        router: dict[str, Any] = self._state.router_bucket(state)
        selection: dict[str, Any] = self._state.selection_bucket(state)
        request: dict[str, Any] = self._state.request_values_from_state(state)
        show_score: bool = bool(
            selection.get("show_score") or selection.get("score_requested")
        )
        jobs: list[dict[str, Any]] = self.job_cards_for_current_request(state)
        payload: dict[str, Any] = {
            "intent": self.public_presentation_intent(state),
            "show_score": show_score,
            "assessment_requested": bool(request.get("assessment_requested")),
            "input_error": bool(state.get("input_error")),
            "conversation_memory": self._state.conversation_memory(state),
        }
        performed_actions: list[dict[str, Any]] = self.performed_actions_payload(state)
        if performed_actions:
            payload["performed_actions"] = performed_actions
        role: str = self.search_role_label(state)
        if not role:
            role = ", ".join(
                TextUtils.display_role_constraints(
                    list(request.get("role_constraints") or [])
                )
            )
        if role:
            payload["role"] = role
        payload["job_list"] = jobs
        payload["job_list_count"] = len(jobs)
        missing_prerequisites: list[str] = []
        if request.get("job_task") == "match":
            if not self._cvs.extracted_cv_documents(state):
                missing_prerequisites.append("cv")
            if not self._jobs.job_results_for_display(state):
                missing_prerequisites.append("jobs")
        if jobs or missing_prerequisites:
            payload["available_job_count"] = len(
                self._jobs.job_results_for_display(state)
            )
        if jobs and self._state.jobs_bucket(state).get("scrape_truncated"):
            payload["more_jobs_may_exist"] = True
        if missing_prerequisites:
            payload["missing_prerequisites"] = missing_prerequisites
        assessment: dict[str, Any] | None = self.public_assessment(
            state, show_score=show_score
        )
        if assessment is not None:
            payload["assessment"] = assessment
        review: dict[str, Any] | None = self.public_review_payload(state)
        if review is not None:
            payload["review"] = review
        comparison: dict[str, Any] | None = self.public_comparison_payload(state)
        if comparison is not None:
            payload["comparison"] = comparison
        extracted: dict[str, Any] | None = self.public_extracted_job(state)
        if extracted is not None:
            payload["extracted_job"] = extracted
        profiles: list[dict[str, Any]] = []
        for document in self._cvs.state_cv_documents(state):
            summary: dict[str, Any] = self._cvs.cv_feature_summary(
                document.get("cv_features")
                if isinstance(document.get("cv_features"), dict)
                else None
            )
            if not summary:
                continue
            profiles.append(
                {
                    "filename": str(document.get("filename") or "cv.pdf"),
                    **summary,
                }
            )
        if profiles:
            payload["cv_profiles"] = profiles

        needs_cv_text: bool = bool(
            selection.get("needs_cv_text") or router.get("needs_cv_text")
        )
        if needs_cv_text:
            documents_with_text: list[dict[str, Any]] = [
                document
                for document in self._cvs.state_cv_documents(state)
                if str(document.get("cv_text") or "").strip()
            ]
            selected_document: dict[str, Any] | None = self._cvs.resolve_selected_cv(
                state
            )
            selected_text: str = (
                str(selected_document.get("cv_text") or "").strip()
                if isinstance(selected_document, dict)
                else ""
            )
            if selected_text:
                payload["cv_text"] = TextUtils.short_text(selected_text, 4000)
            elif len(documents_with_text) == 1:
                payload["cv_text"] = TextUtils.short_text(
                    documents_with_text[0].get("cv_text") or "",
                    4000,
                )
            elif documents_with_text:
                payload["cv_texts"] = [
                    {
                        "filename": str(document.get("filename") or "cv.pdf"),
                        "text": TextUtils.short_text(
                            document.get("cv_text") or "", 3000
                        ),
                    }
                    for document in documents_with_text
                ]
        return payload
