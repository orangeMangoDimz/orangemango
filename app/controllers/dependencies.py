from __future__ import annotations

from fastapi import HTTPException, status

from app.config.const.api_res import CHAT_SERVICE_NOT_CONFIGURED
from app.db.session import DatabaseConfigurationError
from app.models.chat_model import ChatConfigurationError
from app.services.chat_service import ChatService
from app.services.cv_service import CvService


class ServiceProvider:
    """Create and own the shared application services."""

    def __init__(self) -> None:
        self._chat_service: ChatService | None = None

    def get_chat_service(self) -> ChatService:
        if self._chat_service is None:
            try:
                self._chat_service = ChatService.from_env()
            except (ChatConfigurationError, DatabaseConfigurationError) as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=CHAT_SERVICE_NOT_CONFIGURED,
                ) from exc
        return self._chat_service

    def get_cv_service(self) -> CvService:
        return self.get_chat_service().cv_service

    async def shutdown(self) -> None:
        if self._chat_service is not None:
            await self._chat_service.shutdown()


service_provider: ServiceProvider = ServiceProvider()
