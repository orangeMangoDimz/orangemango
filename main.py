from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator


MAX_MESSAGE_LENGTH = 10_000

app = FastAPI(title="Orangemango API")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message must not be blank")
        return normalized


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
