from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class AcceptedMessageResponse(BaseModel):
    thread_id: str
    request_id: UUID
    status: Literal["accepted"]


class CvUploadResponse(BaseModel):
    cv_id: UUID
    filename: str
    content_type: Literal["application/pdf"]
    file_size: int
    status: Literal["uploaded"]


class CvProcessResponse(BaseModel):
    status: Literal["ok"]


class AcceptedCvExtractionResponse(BaseModel):
    extraction_id: UUID
    cv_id: UUID
    thread_id: str
    status: Literal["accepted"]


class CvExtractionResponse(BaseModel):
    extraction_id: UUID
    cv_id: UUID
    thread_id: str
    status: Literal["processing", "completed", "failed"]
    result: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
