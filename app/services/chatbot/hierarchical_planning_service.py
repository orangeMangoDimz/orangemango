"""End-to-end parent, CV, and job planner model calls."""

from __future__ import annotations

import json
from typing import Any

from app.config.const.chatbot import MAX_ROUTER_CHARS
from app.config.const.chatbot_prompts import (
    CV_PLANNER_DATA_HEADER,
    CV_PLANNER_PROMPT,
    JOB_PLANNER_DATA_HEADER,
    JOB_PLANNER_PROMPT,
    PARENT_PLANNER_DATA_HEADER,
    PARENT_PLANNER_PROMPT,
)
from app.models.chat_model import ChatModel
from app.models.chatbot.schemas import (
    CvPlan,
    JobPlan,
    ParentIntent,
    ParentPlan,
    ParentPlanStep,
)
from app.models.chatbot.state import (
    ConversationState,
    CvSubagentState,
    JobSubagentState,
)
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.services.chatbot.message_reader import MessageReader


class HierarchicalPlanningService:
    """Keep each planner responsible for one level of the graph."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        messages: MessageReader,
        chat_model: ChatModel,
    ) -> None:
        self._state = state
        self._messages = messages
        self._chat_model = chat_model

    async def parent_plan_node(self, state: ConversationState) -> dict[str, Any]:
        latest: str = self._state.last_user_text(state)[:MAX_ROUTER_CHARS]
        if not latest:
            fallback = ParentPlan(
                intent=ParentIntent(query="No user query", goal="Respond safely"),
                plan=[
                    ParentPlanStep(
                        node="respond",
                        expected="Explain that no user request is available.",
                        reason="No executable user query was supplied.",
                    )
                ],
            )
            return fallback.model_dump()
        args: dict[str, Any] = dict(state.get("args") or {})
        context: dict[str, Any] = {
            "query": latest,
            "cv": [
                {
                    "id": item.get("id"),
                    "filename": item.get("filename"),
                    "has_content": bool(str(item.get("content") or "").strip()),
                    "has_features": bool(item.get("features")),
                }
                for item in (args.get("cv") or [])
                if isinstance(item, dict)
            ],
            "job": [
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "company": item.get("company"),
                    "has_features": bool(item.get("features")),
                }
                for item in (args.get("job") or [])
                if isinstance(item, dict)
            ],
            "recent_timeline": list(state.get("timeline") or [])[-8:],
            "recent_messages": [
                {
                    "role": self._messages.message_role(message),
                    "content": self._messages.message_text(message),
                }
                for message in (state.get("messages") or [])[-6:]
                if self._messages.message_text(message)
            ],
        }
        planner: Any = self._chat_model.structured(ParentPlan)
        plan: ParentPlan = await planner.ainvoke(
            [
                {"role": "system", "content": PARENT_PLANNER_PROMPT},
                {
                    "role": "user",
                    "content": PARENT_PLANNER_DATA_HEADER
                    + json.dumps(context, ensure_ascii=False),
                },
            ]
        )
        return plan.model_dump()

    async def cv_plan_node(self, state: CvSubagentState) -> dict[str, Any]:
        args: dict[str, Any] = dict(state.get("args") or {})
        dependency: dict[str, Any] = dict(state.get("dependency") or {})
        context: dict[str, Any] = {
            "dependency": dependency,
            "cv": [
                {
                    "id": item.get("id"),
                    "filename": item.get("filename"),
                    "has_content": bool(str(item.get("content") or "").strip()),
                    "has_features": bool(item.get("features")),
                    "has_review": bool(item.get("review")),
                }
                for item in (args.get("cv") or [])
                if isinstance(item, dict)
            ],
            "need_to_extract": int(args.get("need_to_extract") or 0),
        }
        planner: Any = self._chat_model.structured(CvPlan)
        result: CvPlan = await planner.ainvoke(
            [
                {"role": "system", "content": CV_PLANNER_PROMPT},
                {
                    "role": "user",
                    "content": CV_PLANNER_DATA_HEADER
                    + json.dumps(context, ensure_ascii=False),
                },
            ]
        )
        payload: dict[str, Any] = result.model_dump()
        return {
            "plan": payload.pop("plan"),
            "args": {**args, **payload},
        }

    async def job_plan_node(self, state: JobSubagentState) -> dict[str, Any]:
        args: dict[str, Any] = dict(state.get("args") or {})
        dependency: dict[str, Any] = dict(state.get("dependency") or {})
        context: dict[str, Any] = {
            "dependency": dependency,
            "cv": [
                {
                    "id": item.get("id"),
                    "filename": item.get("filename"),
                    "features": item.get("features") or {},
                }
                for item in (args.get("cv") or [])
                if isinstance(item, dict)
            ],
            "job": [
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "company": item.get("company"),
                    "validation_status": item.get("validation_status"),
                }
                for item in (args.get("job") or [])
                if isinstance(item, dict)
            ],
            "need_to_search": bool(args.get("need_to_search")),
            "need_to_extract": int(args.get("need_to_extract") or 0),
        }
        planner: Any = self._chat_model.structured(JobPlan)
        result: JobPlan = await planner.ainvoke(
            [
                {"role": "system", "content": JOB_PLANNER_PROMPT},
                {
                    "role": "user",
                    "content": JOB_PLANNER_DATA_HEADER
                    + json.dumps(context, ensure_ascii=False),
                },
            ]
        )
        payload: dict[str, Any] = result.model_dump()
        search: dict[str, Any] = dict(payload.pop("search") or {})
        return {
            "plan": payload.pop("plan"),
            "args": {**args, **payload, "search": search},
        }
