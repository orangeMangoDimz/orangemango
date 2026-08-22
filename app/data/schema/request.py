from pydantic import BaseModel, Field, ValidationInfo, field_validator

from app.config.const.api_res import (
    MESSAGE_MUST_NOT_BE_BLANK,
    THREAD_ID_MUST_NOT_BE_BLANK,
)
from app.config.const.chat import MAX_MESSAGE_LENGTH, MAX_THREAD_ID_LENGTH


class MessageRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=MAX_THREAD_ID_LENGTH)
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)

    @field_validator("thread_id", "message")
    @classmethod
    def validate_text(cls, value: str, info: ValidationInfo) -> str:
        normalized = value.strip()
        if not normalized:
            if info.field_name == "thread_id":
                raise ValueError(THREAD_ID_MUST_NOT_BE_BLANK)
            raise ValueError(MESSAGE_MUST_NOT_BE_BLANK)
        return normalized


class CvExtractionRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=MAX_THREAD_ID_LENGTH)

    @field_validator("thread_id")
    @classmethod
    def validate_thread_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError(THREAD_ID_MUST_NOT_BE_BLANK)
        return normalized
