from __future__ import annotations

from fastapi import APIRouter

from app.config.const.api_path import HEALTH_PATH
from app.config.const.api_res import (
    HEALTH_API_TAG,
    HEALTH_ENDPOINT_DESCRIPTION,
    HEALTH_ENDPOINT_SUMMARY,
    HEALTH_STATUS_OK,
)
from app.data.schema.response import HealthResponse


class HealthController:
    """Expose service health endpoints."""

    def __init__(self) -> None:
        self.router: APIRouter = APIRouter()
        self.router.add_api_route(
            HEALTH_PATH,
            self.healthz,
            methods=["GET"],
            response_model=HealthResponse,
            summary=HEALTH_ENDPOINT_SUMMARY,
            description=HEALTH_ENDPOINT_DESCRIPTION,
            tags=[HEALTH_API_TAG],
        )

    async def healthz(self) -> HealthResponse:
        return HealthResponse(status=HEALTH_STATUS_OK)
