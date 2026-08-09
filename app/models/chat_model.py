from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config.const.chat import DEFAULT_MODEL, MAX_OUTPUT_TOKENS


class ChatConfigurationError(RuntimeError):
    """Raised when the configured chat model cannot be constructed."""


def _max_tokens_from_env() -> int:
    value = os.getenv("OPENAI_MAX_TOKENS", "")
    if not value.strip():
        return MAX_OUTPUT_TOKENS

    try:
        max_tokens = int(value)
    except ValueError as exc:
        raise ChatConfigurationError(
            "OPENAI_MAX_TOKENS must be a positive integer"
        ) from exc

    if max_tokens < 1:
        raise ChatConfigurationError("OPENAI_MAX_TOKENS must be a positive integer")
    return max_tokens


class ChatModel:
    """Application-owned wrapper around the configured OpenAI chat model."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        max_tokens: int,
    ) -> None:
        if not api_key.strip():
            raise ChatConfigurationError("OPENAI_API_KEY is not configured")
        if not model_name.strip():
            raise ChatConfigurationError("OPENAI_MODEL must not be blank")
        if max_tokens < 1:
            raise ChatConfigurationError("max_tokens must be positive")

        self._model_name = model_name
        self._client = ChatOpenAI(
            model=model_name,
            temperature=0,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    @classmethod
    def from_env(cls) -> ChatModel:
        load_dotenv(override=False)
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            max_tokens=_max_tokens_from_env(),
        )

    def structured(self, schema: type[BaseModel]) -> Any:
        """Return a runnable that emits instances of the provided schema."""
        return self._client.with_structured_output(schema)

    def response(self) -> ChatOpenAI:
        """Return the configured response runnable for LangGraph streaming."""
        return self._client

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name
