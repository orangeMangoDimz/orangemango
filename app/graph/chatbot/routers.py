"""Conditional-edge routing for the chatbot graph and its two subagents.

Method names intentionally match the original module-level functions: LangGraph
labels each conditional branch by the callable's ``__name__``, so renaming them
would change the rendered graph.
"""

from __future__ import annotations

from typing import Any

from app.config.const.chatbot import (
    AGENT_ACTIONS,
    BRANCH_END,
    MAX_AGENT_ACTIONS,
    NODE_EXTRACT_PASTED_JOB,
    NODE_MISSING_CV,
    NODE_SCRAPE_JOBS,
    NODE_WORKFLOW_PLANNER,
    ROUTE_COMPARE_CVS,
    ROUTE_EXTRACT_CV,
    ROUTE_EXTRACT_JOB,
    ROUTE_MATCH_JOBS,
    ROUTE_RESPOND,
    ROUTE_REVIEW_CV,
    ROUTE_SEARCH_JOBS,
)
from app.models.chatbot.literals import JobResponse, RouteName
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.repositories.chatbot.job_state_repository import JobStateRepository
from app.services.chatbot.action_reuse_service import ActionReuseService


class CvSubagentRouter:
    """Entry and post-extraction routing inside the CV subagent."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        cvs: CvStateRepository,
    ) -> None:
        self._state = state
        self._cvs = cvs

    def route_into_cv_subagent(self, state: ConversationState) -> str:
        documents: list[dict[str, Any]] = self._cvs.state_cv_documents(state)
        has_text: bool = any((doc.get("cv_text") or "").strip() for doc in documents)
        if not has_text:
            return NODE_MISSING_CV
        if self._cvs.cvs_need_extraction(state) or not any(
            doc.get("cv_features") for doc in documents
        ):
            return ROUTE_EXTRACT_CV
        route: RouteName | None = self._state.router_bucket(state).get("route")
        if route == ROUTE_COMPARE_CVS:
            return ROUTE_COMPARE_CVS
        if route == ROUTE_REVIEW_CV:
            return ROUTE_REVIEW_CV
        return ROUTE_EXTRACT_CV


    def route_after_cv_extraction(self, state: ConversationState) -> str:
        """Continue to the requested CV action after prerequisite extraction."""
        route: RouteName | None = self._state.router_bucket(state).get("route")
        extracted_count: int = len(self._cvs.extracted_cv_documents(state))

        if route == ROUTE_REVIEW_CV and extracted_count >= 1:
            return ROUTE_REVIEW_CV
        if route == ROUTE_COMPARE_CVS and extracted_count >= 2:
            return ROUTE_COMPARE_CVS
        return BRANCH_END

class JobSubagentRouter:
    """Entry and post-search routing inside the job subagent."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        jobs: JobStateRepository,
        reuse: ActionReuseService,
    ) -> None:
        self._state = state
        self._jobs = jobs
        self._reuse = reuse

    def route_into_job_subagent(self, state: ConversationState) -> str:
        router: dict[str, Any] = self._state.router_bucket(state)
        selection: dict[str, Any] = self._state.selection_bucket(state)
        route: RouteName | None = router.get("route")
        planned_stages: list[str] = self._state.plan_bucket(state).get("planned_stages") or []
        if not planned_stages:
            return BRANCH_END
        if route == ROUTE_EXTRACT_JOB:
            return NODE_EXTRACT_PASTED_JOB if ROUTE_EXTRACT_JOB in planned_stages else BRANCH_END
        if route == ROUTE_SEARCH_JOBS:
            return NODE_SCRAPE_JOBS if NODE_SCRAPE_JOBS in planned_stages else BRANCH_END
        if route == ROUTE_MATCH_JOBS:
            if NODE_SCRAPE_JOBS in planned_stages:
                if (
                    not selection.get("refresh_requested")
                    and self._reuse.current_search_is_reusable(state)
                    and self._jobs.resolve_selected_jobs(state)
                ):
                    planned_stages = [item for item in planned_stages if item != NODE_SCRAPE_JOBS]
                    if ROUTE_MATCH_JOBS not in planned_stages:
                        return BRANCH_END
                    if self._reuse.action_result_is_reusable(
                        state,
                        ROUTE_MATCH_JOBS,
                        self._reuse.action_fingerprint(ROUTE_MATCH_JOBS, state),
                    ):
                        return BRANCH_END
                return NODE_SCRAPE_JOBS
            if ROUTE_MATCH_JOBS in planned_stages:
                return ROUTE_MATCH_JOBS
            if (
                selection.get("job_source") == "pasted"
                and ROUTE_EXTRACT_JOB in planned_stages
                and not self._state.jobs_bucket(state).get("results")
            ):
                return NODE_EXTRACT_PASTED_JOB
        return BRANCH_END


    def route_after_search_or_extract(self, state: ConversationState) -> str:
        router: dict[str, Any] = self._state.router_bucket(state)
        route: RouteName | None = router.get("route")
        planned_stages: list[str] = self._state.plan_bucket(state).get("planned_stages") or []
        if route != ROUTE_MATCH_JOBS or ROUTE_MATCH_JOBS not in planned_stages:
            return BRANCH_END
        jobs_state: dict[str, Any] = self._state.jobs_bucket(state)
        active_keys: Any = jobs_state.get("active_job_keys")
        if isinstance(active_keys, list) and not active_keys:
            return BRANCH_END
        if not jobs_state.get("results"):
            return BRANCH_END
        return ROUTE_MATCH_JOBS

class ChatbotRouter:
    """Top-level routing between planning, subagents, and the response node."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        cvs: CvStateRepository,
    ) -> None:
        self._state = state
        self._cvs = cvs

    def route_after_router(self, state: ConversationState) -> str:
        actions: list[str] = self._state.completed_actions(state)
        if len(actions) >= MAX_AGENT_ACTIONS:
            return ROUTE_RESPOND
        router: dict[str, Any] = self._state.router_bucket(state)
        selection: dict[str, Any] = self._state.selection_bucket(state)
        route: RouteName = router.get("route") or ROUTE_RESPOND
        needs_extraction: bool = self._cvs.cvs_need_extraction(state)
        if needs_extraction and self._cvs.intent_requires_cv_features(state):
            if route in {ROUTE_REVIEW_CV, ROUTE_COMPARE_CVS} and ROUTE_EXTRACT_CV not in actions:
                return route
            return ROUTE_RESPOND if ROUTE_EXTRACT_CV in actions else ROUTE_EXTRACT_CV
        if route == ROUTE_EXTRACT_CV and not needs_extraction:
            return ROUTE_RESPOND
        if route in actions:
            return ROUTE_RESPOND
        if route == ROUTE_EXTRACT_CV:
            return ROUTE_EXTRACT_CV
        if route == ROUTE_REVIEW_CV:
            return ROUTE_REVIEW_CV
        if route == ROUTE_COMPARE_CVS:
            if len(self._cvs.extracted_cv_documents(state)) < 2:
                return ROUTE_RESPOND
            return ROUTE_COMPARE_CVS
        if route == ROUTE_EXTRACT_JOB:
            return ROUTE_EXTRACT_JOB
        if route == ROUTE_SEARCH_JOBS:
            return ROUTE_SEARCH_JOBS
        if route == ROUTE_MATCH_JOBS:
            if self._state.jobs_bucket(state).get("results"):
                return ROUTE_MATCH_JOBS
            if selection.get("job_source") in {"pasted", "search"}:
                return ROUTE_MATCH_JOBS
            return ROUTE_RESPOND
        return ROUTE_RESPOND


    def route_after_plan_validation(self, state: ConversationState) -> str:
        route: Any = self._state.router_bucket(state).get("route") or ROUTE_RESPOND
        plan: dict[str, Any] = self._state.plan_bucket(state)
        if route in {ROUTE_SEARCH_JOBS, ROUTE_MATCH_JOBS} and not (plan.get("planned_stages") or []):
            return ROUTE_RESPOND
        if route in AGENT_ACTIONS or route == ROUTE_RESPOND:
            return str(route)
        return ROUTE_RESPOND


    def route_after_agent_action(self, state: ConversationState) -> str:
        return (
            ROUTE_RESPOND
            if len(self._state.completed_actions(state)) >= MAX_AGENT_ACTIONS
            else NODE_WORKFLOW_PLANNER
        )


    def route_after_cv_subagent(self, state: ConversationState) -> str:
        documents: list[dict[str, Any]] = self._cvs.state_cv_documents(state)
        has_text: bool = any((doc.get("cv_text") or "").strip() for doc in documents)
        if not has_text:
            return ROUTE_RESPOND
        return self.route_after_agent_action(state)


    def route_after_job_subagent(self, state: ConversationState) -> str:
        router: dict[str, Any] = self._state.router_bucket(state)
        selection: dict[str, Any] = self._state.selection_bucket(state)
        route: Any = router.get("route")
        plan: dict[str, Any] = self._state.plan_bucket(state)
        staged: list[str] = plan.get("planned_stages") or []
        response_type: JobResponse = self._state.request_job_response(state)
        if route == ROUTE_MATCH_JOBS:
            if ROUTE_MATCH_JOBS not in staged or ROUTE_MATCH_JOBS in self._state.completed_actions(state):
                return ROUTE_RESPOND
        if route == ROUTE_SEARCH_JOBS:
            if response_type != "list":
                return ROUTE_RESPOND
            if (
                ROUTE_SEARCH_JOBS in self._state.completed_actions(state)
                and not selection.get("assessment_requested")
            ):
                return ROUTE_RESPOND
        return self.route_after_agent_action(state)
