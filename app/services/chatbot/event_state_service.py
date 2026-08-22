"""Planner state adapters, validation results, and response projection."""

from __future__ import annotations

from typing import Any

from app.config.const.chatbot import (
    MAX_CV_DOCUMENTS,
    NODE_COMPARE_CVS,
    NODE_MATCH_JOBS,
    NODE_PARENT_PLANNER,
    NODE_SCRAPE_JOBS,
)
from app.models.chatbot.state import (
    ConversationState,
    CvSubagentState,
    FinalResponseState,
    JobSubagentState,
    PlannerDependencyState,
    PlanStepState,
    TimelineEventState,
    ValidationEntryState,
    ValidationState,
    add_chat_messages,
)
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.services.chatbot.job_key_utils import JobKeyUtils
from app.services.chatbot.upload_parser import UploadParser


class EventStateService:
    """Keep planner, action, and final-response state boundaries explicit."""

    def __init__(self, *, state: ConversationStateRepository) -> None:
        self._state = state

    @staticmethod
    def empty_validation() -> ValidationState:
        return {"passed": [], "errors": []}

    @staticmethod
    def update_validation(
        validation: ValidationState | None,
        *,
        passed: list[ValidationEntryState] | None = None,
        errors: list[ValidationEntryState] | None = None,
    ) -> ValidationState:
        current: dict[str, ValidationEntryState] = {}
        buckets: dict[str, str] = {}
        for bucket in ("passed", "errors"):
            for entry in (validation or {}).get(bucket, []):
                code: str = str(entry.get("code") or "")
                if code:
                    current[code] = dict(entry)
                    buckets[code] = bucket
        for bucket, entries in (("passed", passed or []), ("errors", errors or [])):
            for entry in entries:
                code = str(entry.get("code") or "")
                if code:
                    current[code] = dict(entry)
                    buckets[code] = bucket
        return {
            "passed": [current[code] for code in current if buckets[code] == "passed"],
            "errors": [current[code] for code in current if buckets[code] == "errors"],
        }

    @staticmethod
    def validation_entry(code: str, message: str) -> ValidationEntryState:
        return {"code": code, "message": message}

    @staticmethod
    def append_event(
        timeline: list[TimelineEventState] | None,
        *,
        node: str,
        status: str,
        summary: str,
        args: dict[str, Any] | None = None,
    ) -> list[TimelineEventState]:
        event: TimelineEventState = {
            "order": len(timeline or []) + 1,
            "node": node,
            "status": status,  # type: ignore[typeddict-item]
            "summary": summary,
        }
        if args:
            event["args"] = args
        return [*(timeline or []), event]

    @staticmethod
    def latest_status(state: dict[str, Any]) -> str:
        timeline: Any = state.get("timeline")
        if not isinstance(timeline, list) or not timeline:
            return "failed"
        event: Any = timeline[-1]
        return (
            str(event.get("status") or "failed")
            if isinstance(event, dict)
            else "failed"
        )

    @staticmethod
    def cv_item(document: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": document.get("id"),
            "filename": document.get("filename"),
            "content": document.get("cv_text") or document.get("content") or "",
            "features": document.get("cv_features", document.get("features")),
            "extraction": document.get("cv_result", document.get("extraction")),
            "review": document.get("cv_review", document.get("review")),
        }

    @staticmethod
    def cv_document(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item.get("id"),
            "filename": item.get("filename") or "cv.pdf",
            "cv_text": item.get("content") or "",
            "cv_features": item.get("features"),
            "cv_result": item.get("extraction"),
            "cv_review": item.get("review"),
        }

    @staticmethod
    def job_item(result: dict[str, Any], index: int) -> dict[str, Any]:
        if "job_card" not in result:
            return dict(result)
        card: dict[str, Any] = (
            dict(result.get("job_card"))
            if isinstance(result.get("job_card"), dict)
            else {}
        )
        return {
            "id": JobKeyUtils.job_selection_key(result, index),
            "title": card.get("title"),
            "company": card.get("company"),
            "location": card.get("location"),
            "content": card.get("description") or "",
            "url": card.get("url"),
            "salary": card.get("salary"),
            "posted_date": card.get("posted_date"),
            "site": card.get("site"),
            "features": result.get("matching_features"),
            "extract": result.get("extract"),
            "validation_status": result.get("validation_status"),
            "validation_errors": list(result.get("validation_errors") or []),
            "warnings": list(result.get("warnings") or []),
            "match": result.get("match"),
        }

    @staticmethod
    def job_result(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_card": {
                "title": item.get("title") or "",
                "company": item.get("company") or "",
                "location": item.get("location") or "",
                "description": item.get("content") or "",
                "url": item.get("url") or "",
                "salary": item.get("salary") or "",
                "posted_date": item.get("posted_date") or "",
                "site": item.get("site") or "",
            },
            "matching_features": item.get("features"),
            "extract": item.get("extract"),
            "validation_status": item.get("validation_status"),
            "validation_errors": list(item.get("validation_errors") or []),
            "warnings": list(item.get("warnings") or []),
        }

    def prepare_parent_turn(self, state: ConversationState) -> dict[str, Any]:
        current_args: dict[str, Any] = dict(state.get("args") or {})
        cvs: list[dict[str, Any]] = [
            dict(item)
            for item in (current_args.get("cv") or [])
            if isinstance(item, dict)
        ]
        jobs: list[dict[str, Any]] = [
            dict(item)
            for item in (current_args.get("job") or [])
            if isinstance(item, dict)
        ]
        validation: ValidationState = self.empty_validation()
        messages: list[Any] = list(state.get("messages") or [])
        pending: list[dict[str, Any]] = UploadParser.pending_uploads_from_messages(
            messages
        )
        existing_ids: set[str] = {str(item.get("id") or "") for item in cvs}
        upload_errors: list[ValidationEntryState] = []
        appended: int = 0
        for upload in (pending or [])[: max(0, MAX_CV_DOCUMENTS - len(cvs))]:
            try:
                item: dict[str, Any] = self.cv_item(
                    UploadParser.cv_document_from_upload(upload)
                )
                cv_id: str = str(item.get("id") or "")
                if cv_id and cv_id not in existing_ids:
                    cvs.append(item)
                    existing_ids.add(cv_id)
                    appended += 1
            except Exception as exc:
                upload_errors.append(
                    self.validation_entry(
                        "CV_TYPE_CHECK",
                        f"CV type is invalid: {type(exc).__name__}.",
                    )
                )
        if upload_errors:
            validation = self.update_validation(validation, errors=upload_errors)
        else:
            validation = self.update_validation(
                validation,
                passed=[
                    self.validation_entry(
                        "CV_TYPE_CHECK",
                        (
                            f"{appended} new CV file(s) are valid."
                            if appended
                            else "Current CV data is valid."
                        ),
                    )
                ],
            )
        sanitized: list[Any] = UploadParser.sanitize_file_messages(messages)
        cleared: list[Any] = UploadParser.clear_stashed_uploads(messages)
        message_updates: list[Any] = [*sanitized, *cleared]
        return {
            "messages": add_chat_messages(messages, message_updates),
            "timeline": [],
            "args": {"cv": cvs, "job": jobs},
            "validation": validation,
            "intent": {},
            "plan": [],
        }

    @staticmethod
    def dependency(
        intent: dict[str, Any],
        step: PlanStepState,
    ) -> PlannerDependencyState:
        return {
            "source": NODE_PARENT_PLANNER,
            "query": str(intent.get("query") or ""),
            "goal": str(intent.get("goal") or ""),
            "expected": step["expected"],
            "reason": step["reason"],
            "cv_ids": list(intent.get("cv_ids") or []),
            "job_ids": list(intent.get("job_ids") or []),
        }

    def cv_input(
        self,
        state: ConversationState,
        step: PlanStepState,
    ) -> CvSubagentState:
        args: dict[str, Any] = dict(state.get("args") or {})
        cvs: list[dict[str, Any]] = [
            dict(item) for item in (args.get("cv") or []) if isinstance(item, dict)
        ]
        return {
            "timeline": list(state.get("timeline") or []),
            "dependency": self.dependency(dict(state.get("intent") or {}), step),
            "args": {
                "cv": cvs,
                "need_to_extract": sum(
                    1
                    for item in cvs
                    if str(item.get("content") or "").strip()
                    and not item.get("features")
                ),
            },
            "validation": dict(state.get("validation") or self.empty_validation()),
        }

    def job_input(
        self,
        state: ConversationState,
        step: PlanStepState,
    ) -> JobSubagentState:
        args: dict[str, Any] = dict(state.get("args") or {})
        cvs: list[dict[str, Any]] = [
            dict(item) for item in (args.get("cv") or []) if isinstance(item, dict)
        ]
        jobs: list[dict[str, Any]] = [
            dict(item) for item in (args.get("job") or []) if isinstance(item, dict)
        ]
        return {
            "timeline": list(state.get("timeline") or []),
            "dependency": self.dependency(dict(state.get("intent") or {}), step),
            "args": {
                "cv": cvs,
                "job": jobs,
                "need_to_search": not bool(jobs),
                "need_to_extract": sum(
                    1
                    for item in jobs
                    if item.get("validation_status") != "valid"
                    or not item.get("features")
                ),
            },
            "validation": dict(state.get("validation") or self.empty_validation()),
        }

    @staticmethod
    def strip_subagent_state(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "timeline": list(state.get("timeline") or []),
            "args": dict(state.get("args") or {}),
            "validation": dict(state.get("validation") or {}),
        }

    @staticmethod
    def _public_cv(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item.get(key)
            for key in ("id", "filename", "content", "features", "review")
            if item.get(key) not in (None, "", [], {})
        }

    @staticmethod
    def _public_job(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item.get(key)
            for key in (
                "id",
                "title",
                "company",
                "location",
                "content",
                "url",
                "salary",
                "posted_date",
                "features",
                "match",
            )
            if item.get(key) not in (None, "", [], {})
        }

    def final_projection(self, state: ConversationState) -> FinalResponseState:
        args: dict[str, Any] = dict(state.get("args") or {})
        intent: dict[str, Any] = dict(state.get("intent") or {})
        raw_timeline: list[dict[str, Any]] = [
            event for event in (state.get("timeline") or []) if isinstance(event, dict)
        ]
        timeline: list[TimelineEventState] = [
            {
                key: event[key]
                for key in ("order", "node", "status", "summary")
                if key in event
            }
            for event in raw_timeline
        ]
        nodes: set[str] = {str(event.get("node") or "") for event in timeline}
        wanted_cv: set[str] = {str(value) for value in intent.get("cv_ids") or []}
        wanted_job: set[str] = {str(value) for value in intent.get("job_ids") or []}
        visible_cv: list[dict[str, Any]] = []
        if "match_jobs" not in nodes and nodes & {
            "extract_cv",
            "review_cv",
            "compare_cvs",
        }:
            visible_cv = [
                self._public_cv(item)
                for item in (args.get("cv") or [])
                if isinstance(item, dict)
                and (not wanted_cv or str(item.get("id") or "") in wanted_cv)
            ]
        visible_job: list[dict[str, Any]] = []
        if "match_jobs" in nodes:
            visible_job = [
                self._public_job(item)
                for item in (args.get("job") or [])
                if isinstance(item, dict) and str(item.get("id") or "") in wanted_job
            ]
        elif nodes & {"search_jobs", "extract_job", "extract_pasted_job"}:
            visible_job = [
                self._public_job(item)
                for item in (args.get("job") or [])
                if isinstance(item, dict)
                and (not wanted_job or str(item.get("id") or "") in wanted_job)
            ]
        visible_cv_ids: set[str] = {
            str(item.get("id") or "") for item in visible_cv if item.get("id")
        }
        visible_job_ids: set[str] = {
            str(item.get("id") or "") for item in visible_job if item.get("id")
        }
        for event, raw in zip(timeline, raw_timeline, strict=True):
            event_args: Any = raw.get("args")
            if not isinstance(event_args, dict):
                continue
            if raw.get("node") == NODE_COMPARE_CVS and visible_cv_ids:
                event["args"] = {
                    "cv_ids": [
                        value
                        for value in (event_args.get("cv_ids") or [])
                        if str(value) in visible_cv_ids
                    ],
                    "comparison": event_args.get("comparison"),
                }
            elif raw.get("node") == NODE_SCRAPE_JOBS and visible_job_ids:
                event["args"] = {
                    "search": dict(event_args.get("search") or {}),
                    "job_ids": [
                        value
                        for value in (event_args.get("job_ids") or [])
                        if str(value) in visible_job_ids
                    ],
                    "result_count": len(visible_job_ids),
                }
            elif raw.get("node") == NODE_MATCH_JOBS and visible_job_ids:
                event["args"] = {
                    "job_ids": [
                        value
                        for value in (event_args.get("job_ids") or [])
                        if str(value) in visible_job_ids
                    ],
                    "matches": [
                        {
                            key: value
                            for key, value in match.items()
                            if key not in {"cv_id", "cv_filename"}
                        }
                        for match in (event_args.get("matches") or [])
                        if isinstance(match, dict)
                        and str(match.get("job_key") or "") in visible_job_ids
                    ],
                }
        return {
            "timeline": timeline,
            "args": {"cv": visible_cv, "job": visible_job},
            "validation": dict(state.get("validation") or self.empty_validation()),
        }
