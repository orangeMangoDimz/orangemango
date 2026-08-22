"""Conversation state contract and reducers for the chatbot graph.

The three reducers stay module-level functions: they are referenced inside
``Annotated[...]`` type expressions and cannot be bound methods.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.models.chatbot.literals import EventStatus
from app.services.chatbot.upload_parser import UploadParser


def add_chat_messages(
    left: list[Any] | None,
    right: list[Any] | None,
) -> list[Any]:
    sanitized_right: list[Any] = []
    for message in right or []:
        sanitized: Any | None = UploadParser.sanitize_file_message(
            message,
            stash_upload=True,
        )
        sanitized_right.append(message if sanitized is None else sanitized)
    return add_messages(left or [], sanitized_right)


def merge_maps(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    return {**(left or {}), **(right or {})}


class TimelineEventState(TypedDict, total=False):
    order: int
    node: str
    status: EventStatus
    summary: str
    args: dict[str, Any]


class PlanStepState(TypedDict):
    node: str
    expected: str
    reason: str


class ParentIntentState(TypedDict, total=False):
    query: str
    goal: str
    cv_ids: list[str]
    job_ids: list[str]


class PlannerDependencyState(ParentIntentState, total=False):
    source: str
    expected: str
    reason: str


class ValidationEntryState(TypedDict):
    code: str
    message: str


class ValidationState(TypedDict):
    passed: list[ValidationEntryState]
    errors: list[ValidationEntryState]


class ParentArgsState(TypedDict):
    cv: list[dict[str, Any]]
    job: list[dict[str, Any]]


class ParentPlannerInputState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_chat_messages]
    timeline: list[TimelineEventState]
    args: ParentArgsState
    validation: ValidationState


class CvSubagentArgsState(TypedDict, total=False):
    cv: list[dict[str, Any]]
    need_to_extract: int
    cv_ids: list[str]
    review_mode: str
    review_focus: str | None
    target_role: str | None


class CvSubagentState(TypedDict, total=False):
    timeline: list[TimelineEventState]
    dependency: PlannerDependencyState
    plan: list[PlanStepState]
    args: CvSubagentArgsState
    validation: ValidationState


class CvActionState(TypedDict, total=False):
    timeline: list[TimelineEventState]
    args: CvSubagentArgsState
    validation: ValidationState


class JobSubagentArgsState(TypedDict, total=False):
    cv: list[dict[str, Any]]
    job: list[dict[str, Any]]
    need_to_search: bool
    need_to_extract: int
    cv_ids: list[str]
    job_ids: list[str]
    source: str
    response: str
    refresh: bool
    pasted_content: str | None
    show_score: bool
    search: dict[str, Any]
    scrape_total: int
    scrape_truncated: bool
    active_job_keys: list[str]
    matches: list[dict[str, Any]]
    active_job_goal: dict[str, Any]
    pending_match: dict[str, Any] | None


class JobSubagentState(TypedDict, total=False):
    timeline: list[TimelineEventState]
    dependency: PlannerDependencyState
    plan: list[PlanStepState]
    args: JobSubagentArgsState
    validation: ValidationState


class JobActionState(TypedDict, total=False):
    timeline: list[TimelineEventState]
    args: JobSubagentArgsState
    validation: ValidationState


class FinalResponseArgsState(TypedDict):
    cv: list[dict[str, Any]]
    job: list[dict[str, Any]]


class FinalResponseState(TypedDict):
    timeline: list[TimelineEventState]
    args: FinalResponseArgsState
    validation: ValidationState


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_chat_messages]
    response: str | None
    job_list: list[dict[str, Any]]
    timeline: list[TimelineEventState]
    intent: ParentIntentState
    plan: list[PlanStepState]
    args: Annotated[ParentArgsState, merge_maps]
    validation: ValidationState
    response_projection: Annotated[FinalResponseState, merge_maps]


class StudioInput(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_chat_messages]
