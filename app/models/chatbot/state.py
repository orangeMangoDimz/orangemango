"""Conversation state contract and reducers for the chatbot graph.

The three reducers stay module-level functions: they are referenced inside
``Annotated[...]`` type expressions and cannot be bound methods.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.models.chatbot.literals import (
    AgentAction,
    CvTargetScope,
    ExecutionStatus,
    GoalName,
    JobResponse,
    JobSource,
    JobTargetScope,
    JobTask,
    ReviewMode,
    RoleSource,
    RouteName,
)
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


class RoutingGoalState(TypedDict, total=False):
    name: GoalName
    reason: str
    confidence: float


class RoutingJobState(TypedDict, total=False):
    task: JobTask
    response: JobResponse
    source: JobSource
    input: str | None
    refresh: bool
    scrape: dict[str, Any]


class RoutingRoleState(TypedDict, total=False):
    constraints: list[str]
    evidence: str | None
    source: RoleSource
    candidates: list[dict[str, Any]]


class RoutingAssessmentState(TypedDict, total=False):
    requested: bool
    detail_level: str


class RoutingScoreState(TypedDict, total=False):
    requested: bool
    visible: bool


class RoutingReviewState(TypedDict, total=False):
    target_role: str | None
    mode: ReviewMode
    focus: str | None
    reason: str | None


class RoutingCvState(TypedDict, total=False):
    text_needed: bool
    features_needed: bool


class RoutingContextState(TypedDict, total=False):
    follow_up: bool


class RoutingRequestState(TypedDict, total=False):
    goal: RoutingGoalState
    job: RoutingJobState
    role: RoutingRoleState
    assessment: RoutingAssessmentState
    score: RoutingScoreState
    review: RoutingReviewState
    cv: RoutingCvState
    context: RoutingContextState


class RoutingCvTargetsState(TypedDict, total=False):
    scope: CvTargetScope
    ids: list[str]


class RoutingJobTargetsState(TypedDict, total=False):
    scope: JobTargetScope
    keys: list[str]


class RoutingTargetsState(TypedDict, total=False):
    cv: RoutingCvTargetsState
    job: RoutingJobTargetsState
    unresolved_references: list[str]
    ambiguous: bool


class RoutingPlanState(TypedDict, total=False):
    action: RouteName
    reason: str
    validation: str
    validation_error: str | None
    planned_stages: list[str]
    policy_reason: str
    active_goal_id: str | None
    completed_actions: list[str]


class RoutingState(TypedDict, total=False):
    request: RoutingRequestState
    targets: RoutingTargetsState
    plan: RoutingPlanState


# ExecutionStatus is defined in app.models.chatbot.literals and imported above.


class ExecutionNoteState(TypedDict):
    summary: str
    reason: str


class ExecutionFromNodeState(TypedDict):
    graph: str
    node: str
    uses: list[str]
    note: ExecutionNoteState
    output: dict[str, Any]


class ExecutionToNodeState(TypedDict):
    graph: str
    node: str
    uses: list[str]
    note: ExecutionNoteState
    args: dict[str, Any]
    result: dict[str, Any]


class ExecutionContextState(TypedDict, total=False):
    args_source: RoleSource
    source_reference: dict[str, Any]


ExecutionStepState = TypedDict(
    "ExecutionStepState",
    {
        "index": int,
        "action": AgentAction,
        "status": ExecutionStatus,
        "from": ExecutionFromNodeState,
        "to": ExecutionToNodeState,
        "context": ExecutionContextState,
        "error": str | None,
    },
)


class ExecutionState(TypedDict):
    steps: list[ExecutionStepState]


def merge_routing_maps(
    left: RoutingState | None,
    right: RoutingState | None,
) -> RoutingState:
    merged: dict[str, Any] = dict(left or {})
    for section, update in (right or {}).items():
        if isinstance(update, dict) and isinstance(merged.get(section), dict):
            merged[section] = {**merged[section], **update}
        else:
            merged[section] = update
    return merged  # type: ignore[return-value]


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_chat_messages]
    pending_cv_upload: dict[str, Any] | None
    pending_cv_uploads: list[dict[str, Any]] | None
    input_error: bool
    cv: Annotated[dict[str, Any], merge_maps]
    routing: Annotated[RoutingState, merge_routing_maps]
    # Transitional top-level target/plan buckets are read-only fallbacks.
    targets: dict[str, Any]
    plan: dict[str, Any]
    # Legacy non-request routing buckets are retained for old helpers.
    router: Annotated[dict[str, Any], merge_maps]
    selection: Annotated[dict[str, Any], merge_maps]
    jobs: Annotated[dict[str, Any], merge_maps]
    action_results: Annotated[dict[str, Any], merge_maps]
    execution: Annotated[ExecutionState, merge_maps]
    conversation_memory: Annotated[dict[str, Any], merge_maps]
    conversation_memory_cursor: int
    response: str | None
    job_list: list[dict[str, Any]]
    errors: list[str]


class StudioInput(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_chat_messages]
    pending_cv_upload: dict[str, Any] | None
    pending_cv_uploads: list[dict[str, Any]] | None
