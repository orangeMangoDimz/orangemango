from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class AcceptedMessageResponse(BaseModel):
    thread_id: str
    request_id: UUID
    status: Literal["accepted"]
