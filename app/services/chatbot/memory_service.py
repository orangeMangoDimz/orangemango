"""Compact older conversation turns into durable structured memory."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage

from app.config.const.chatbot import MAX_CONTEXT_MESSAGES
from app.config.const.chatbot_prompts import CONVERSATION_SUMMARY_PROMPT
from app.models.chat_model import ChatModel
from app.models.chatbot.schemas import ConversationMemory
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.services.chatbot.conversation_service import ConversationService


class ConversationMemoryService:
    """Summarize aged turns; compaction failure never blocks a turn."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        conversation: ConversationService,
        chat_model: ChatModel,
    ) -> None:
        self._state = state
        self._conversation = conversation
        self._chat_model = chat_model

    async def summarize_conversation_node(
        self, state: ConversationState
    ) -> dict[str, Any]:
        if not self._conversation.should_summarize_conversation(state):
            return {}

        turns: list[dict[str, str]] = self._conversation.conversation_text_messages(
            state
        )
        keep_from: int = max(0, len(turns) - MAX_CONTEXT_MESSAGES)
        cursor: int = min(self._state.conversation_memory_cursor(state), keep_from)
        older_turns: list[dict[str, str]] = turns[cursor:keep_from]
        if not older_turns:
            return {}

        payload: dict[str, Any] = {
            "existing_memory": self._state.conversation_memory(state),
            "new_turns": older_turns,
        }
        try:
            writer: Any = self._chat_model.structured(ConversationMemory)
            result: Any = await writer.ainvoke(
                [
                    SystemMessage(content=CONVERSATION_SUMMARY_PROMPT),
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ]
            )
            memory: ConversationMemory = (
                result
                if isinstance(result, ConversationMemory)
                else ConversationMemory.model_validate(result)
            )
        except Exception:
            # Compaction is an optimization; a failed summary must not block a turn.
            return {}

        return {
            "conversation_memory": memory.model_dump(),
            "conversation_memory_cursor": keep_from,
        }
