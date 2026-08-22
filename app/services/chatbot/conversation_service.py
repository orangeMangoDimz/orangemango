"""Shape conversation turns for routing, summarization, and the response model."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages.utils import count_tokens_approximately

from app.config.const.chatbot import (
    CONTEXT_SUMMARY_TRIGGER_RATIO,
    DEFAULT_CONTEXT_OUTPUT_RESERVE_TOKENS,
    DEFAULT_CONTEXT_PROMPT_RESERVE_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    ENV_CONTEXT_OUTPUT_RESERVE_TOKENS,
    ENV_CONTEXT_PROMPT_RESERVE_TOKENS,
    ENV_CONTEXT_WINDOW_TOKENS,
    MAX_CONTEXT_MESSAGES,
    MAX_ROUTER_HISTORY_CHARS,
    MAX_ROUTER_HISTORY_MESSAGES,
    MIN_CONTEXT_INPUT_BUDGET,
)
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.services.chatbot.message_reader import MessageReader
from app.services.chatbot.text_utils import TextUtils


class ConversationService:
    """Turn extraction, context budgeting, and history windowing."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        messages: MessageReader,
    ) -> None:
        self._state = state
        self._messages = messages

    def _positive_int_env(self, name: str, default: int) -> int:
        value: str = os.getenv(name, "").strip()
        if not value:
            return default
        try:
            parsed: int = int(value)
        except ValueError:
            return default
        return parsed if parsed > 0 else default


    def context_input_budget(self) -> int:
        context_window: int = self._positive_int_env(
            ENV_CONTEXT_WINDOW_TOKENS,
            DEFAULT_CONTEXT_WINDOW_TOKENS,
        )
        output_reserve: int = self._positive_int_env(
            ENV_CONTEXT_OUTPUT_RESERVE_TOKENS,
            DEFAULT_CONTEXT_OUTPUT_RESERVE_TOKENS,
        )
        prompt_reserve: int = self._positive_int_env(
            ENV_CONTEXT_PROMPT_RESERVE_TOKENS,
            DEFAULT_CONTEXT_PROMPT_RESERVE_TOKENS,
        )
        return max(MIN_CONTEXT_INPUT_BUDGET, context_window - output_reserve - prompt_reserve)

    def conversation_text_messages(self, state: ConversationState) -> list[dict[str, str]]:
        """Return model-readable turns while excluding tool protocol messages."""
        raw_messages: list[Any] = state.get("messages") or []
        if not raw_messages:
            return []

        last_user_index: int = -1
        for index, message in enumerate(raw_messages):
            if self._messages.message_role(message) in {"human", "user"}:
                last_user_index = index

        start: int = last_user_index if last_user_index >= 0 else 0

        turns: list[dict[str, str]] = []
        for message in raw_messages[start:]:
            role_name: str = self._messages.message_role(message)
            if role_name in {"tool"} or self._messages.message_tool_calls(message):
                continue
            content: str = self._messages.message_text(message).strip()
            if not content:
                continue
            if role_name in {"human", "user"}:
                normalized: str = self._messages.normalize_user_turn_text(content)
                if not normalized:
                    continue
                turns.append({"role": "user", "content": normalized})
            elif role_name in {"ai", "assistant"}:
                if not self.should_retain_job_messages(state) and self._appears_like_job_message(
                    content
                ):
                    continue
                turns.append({"role": "assistant", "content": content})
        return turns

    def should_retain_job_messages(self, state: ConversationState) -> bool:
        return self._state.request_job_response(state) == "list"


    def _appears_like_job_message(self, content: str) -> bool:
        lines: list[str] = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        numbered_item: bool = False
        for line in lines[:6]:
            first_segment: str = line.split(" ", 1)[0]
            if first_segment and first_segment[-1] in {".", ")"} and first_segment[:-1].isdigit():
                numbered_item = True
                break
        has_job_fields: bool = any(
            "Location:" in line
            or "Posted:" in line
            or "Salary:" in line
            or "Description:" in line
            or "View Job" in line
            for line in lines
        )
        return bool(numbered_item and (has_job_fields or "http" in content.lower()))


    def should_summarize_conversation(self, state: ConversationState) -> bool:
        turns: list[dict[str, str]] = self.conversation_text_messages(state)
        if len(turns) <= MAX_CONTEXT_MESSAGES:
            return False
        input_budget: int = self.context_input_budget()
        trigger_tokens: int = int(input_budget * CONTEXT_SUMMARY_TRIGGER_RATIO)
        return count_tokens_approximately(turns) >= trigger_tokens

    def router_recent_conversation(self, state: ConversationState) -> list[dict[str, str]]:
        history: list[dict[str, str]] = []
        for message in state.get("messages") or []:
            role_name: str = self._messages.message_role(message)
            if role_name in {"tool"}:
                continue
            tool_calls: Any = (
                message.get("tool_calls")
                if isinstance(message, dict)
                else getattr(message, "tool_calls", None)
            )
            if tool_calls:
                continue
            content: str = TextUtils.short_text(self._messages.message_text(message), MAX_ROUTER_HISTORY_CHARS)
            if not content:
                continue
            if role_name in {"human", "user"}:
                history.append({"role": "user", "content": content})
            elif role_name in {"ai", "assistant"}:
                history.append({"role": "assistant", "content": content})
        return history[-MAX_ROUTER_HISTORY_MESSAGES:]

    def bounded_conversation(self, state: ConversationState) -> list[Any]:
        result: list[Any] = []
        for message in self.conversation_text_messages(state)[-MAX_CONTEXT_MESSAGES:]:
            content: str = TextUtils.short_text(message["content"], 1800)
            if content:
                result.append({"role": message["role"], "content": content})
        return result
