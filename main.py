from __future__ import annotations

from collections.abc import AsyncIterator
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator


DEFAULT_MODEL = "gpt-4o-mini"
MAX_MESSAGE_LENGTH = 10_000
GENERIC_STREAM_ERROR = "Unable to generate a response."

app = FastAPI(title="Orangemango API")


class ChatConfigurationError(RuntimeError):
    pass


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message must not be blank")
        return normalized


def create_chat_model() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key.strip():
        raise ChatConfigurationError

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        temperature=0,
        api_key=api_key,
    )


def chunk_content(chunk: object) -> str:
    if isinstance(chunk, str):
        return chunk

    content = getattr(chunk, "content", "")
    return content if isinstance(content, str) else ""


async def stream_chat(message: str, model: ChatOpenAI) -> AsyncIterator[str]:
    try:
        async for chunk in model.astream(message):
            content = chunk_content(chunk)
            if content:
                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"
    except Exception:
        yield f"data: {json.dumps({'error': GENERIC_STREAM_ERROR})}\n\n"
    else:
        yield "data: [DONE]\n\n"


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/mango/chat", response_class=StreamingResponse)
async def chat(request: ChatRequest) -> StreamingResponse:
    try:
        model = create_chat_model()
    except ChatConfigurationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Chat service is not configured",
        ) from exc

    return StreamingResponse(
        stream_chat(request.message, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
