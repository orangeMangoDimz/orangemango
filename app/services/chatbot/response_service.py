"""Stream the final assistant response from the presentation payload."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.config.const.chatbot_errors import (
    ERROR_RESPONSE_MODEL_EMPTY,
    ERROR_RESPONSE_MODEL_FAILED,
)
from app.config.const.chatbot_prompts import CHAT_PROMPT, PRESENTATION_DATA_HEADER
from app.models.chat_model import ChatModel
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.services.chatbot.conversation_service import ConversationService
from app.services.chatbot.message_reader import MessageReader
from app.services.chatbot.presentation_service import PresentationService


class ResponseService:
    """Compose and stream the user-facing reply."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        presentation: PresentationService,
        conversation: ConversationService,
        messages: MessageReader,
        chat_model: ChatModel,
    ) -> None:
        self._state = state
        self._presentation = presentation
        self._conversation = conversation
        self._messages = messages
        self._chat_model = chat_model

    def is_usable_model_response(self, response: str) -> bool:
        normalized: str = response.strip().casefold()
        return normalized not in {"", "none", "null", "n/a", "na"}

    async def respond_node(
        self,
        state: ConversationState,
        config: RunnableConfig | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = self._presentation.presentation_payload(state)
        try:
            assistant: Any = self._chat_model.response()
            response_parts: list[str] = []
            async for chunk in assistant.astream(
                [
                    SystemMessage(
                        content=CHAT_PROMPT
                        + PRESENTATION_DATA_HEADER
                        + json.dumps(payload, ensure_ascii=False)
                    ),
                    *self._conversation.bounded_conversation(state),
                ],
                config=config,
            ):
                content: str = self._messages.message_text(chunk)
                if content:
                    response_parts.append(content)

            response = "".join(response_parts)
            if not self.is_usable_model_response(response):
                return {
                    "response": None,
                    "errors": self._state.state_errors(state, [ERROR_RESPONSE_MODEL_EMPTY]),
                    "job_list": payload.get("job_list", []),
                }
            result: AIMessage = AIMessage(content=response.strip())
            return {
                "messages": [result],
                "response": response.strip(),
                "job_list": payload.get("job_list", []),
            }
        except Exception as exc:
            return {
                "response": None,
                "job_list": payload.get("job_list", []),
                "errors": self._state.state_errors(
                    state,
                    [f"{ERROR_RESPONSE_MODEL_FAILED}{type(exc).__name__}"],
                ),
            }
