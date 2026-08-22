from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.config.const.api_path import EVENTS_PATH, MESSAGE_PATH
from app.config.const.api_res import (
    CHAT_API_TAG,
    CHAT_ENDPOINT_DESCRIPTION,
    CHAT_ENDPOINT_SUMMARY,
    CHAT_SERVICE_NOT_CONFIGURED,
    CHAT_THREAD_BUSY,
    CHAT_THREAD_NOT_FOUND,
    EVENT_API_TAG,
    EVENT_ENDPOINT_DESCRIPTION,
    EVENT_ENDPOINT_SUMMARY,
    THREAD_ID_MUST_NOT_BE_BLANK,
)
from app.config.const.chat import MAX_THREAD_ID_LENGTH
from app.controllers.dependencies import service_provider
from app.data.schema.request import MessageRequest
from app.data.schema.response import AcceptedMessageResponse
from app.repositories.chat_repository import ChatPersistenceError
from app.security.auth import require_api_token
from app.services.chat_service import (
    ChatService,
    ChatThreadBusyError,
    ChatThreadNotFoundError,
)


class ChatController:
    """Handle chat submission and event-stream HTTP endpoints."""

    def __init__(self) -> None:
        self.router: APIRouter = APIRouter(
            dependencies=[Depends(require_api_token)],
        )
        self.router.add_api_route(
            MESSAGE_PATH,
            self.message,
            methods=["POST"],
            response_model=AcceptedMessageResponse,
            status_code=status.HTTP_202_ACCEPTED,
            summary=CHAT_ENDPOINT_SUMMARY,
            description=CHAT_ENDPOINT_DESCRIPTION,
            tags=[CHAT_API_TAG],
        )
        self.router.add_api_route(
            EVENTS_PATH,
            self.events,
            methods=["GET"],
            response_class=StreamingResponse,
            summary=EVENT_ENDPOINT_SUMMARY,
            description=EVENT_ENDPOINT_DESCRIPTION,
            tags=[EVENT_API_TAG],
        )

    async def message(
        self,
        request: MessageRequest,
        service: ChatService = Depends(service_provider.get_chat_service),
    ) -> AcceptedMessageResponse:
        try:
            return await service.accept_message(request.thread_id, request.message)
        except ChatThreadBusyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=CHAT_THREAD_BUSY,
            ) from exc
        except ChatPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=CHAT_SERVICE_NOT_CONFIGURED,
            ) from exc

    async def events(
        self,
        thread_id: str = Query(..., min_length=1, max_length=MAX_THREAD_ID_LENGTH),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
        service: ChatService = Depends(service_provider.get_chat_service),
    ) -> StreamingResponse:
        normalized_thread_id: str = self._normalize_thread_id(thread_id)
        try:
            await service.ensure_thread(normalized_thread_id)
        except ChatThreadNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=CHAT_THREAD_NOT_FOUND,
            ) from exc
        except ChatPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=CHAT_SERVICE_NOT_CONFIGURED,
            ) from exc

        return StreamingResponse(
            self._format_sse(
                service.events(
                    normalized_thread_id,
                    self._last_event_id(last_event_id),
                )
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @staticmethod
    def _normalize_thread_id(thread_id: str) -> str:
        normalized: str = thread_id.strip()
        if not normalized:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=THREAD_ID_MUST_NOT_BE_BLANK,
            )
        return normalized

    @staticmethod
    def _last_event_id(value: str | None) -> int | None:
        if value is None:
            return None
        try:
            parsed: int = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None

    @staticmethod
    async def _format_sse(events: AsyncIterator[object]) -> AsyncIterator[str]:
        async for event in events:
            event_type = getattr(event, "event_type")
            event_id = getattr(event, "event_id")
            data = getattr(event, "data")
            yield (
                f"event: {event_type}\n"
                f"id: {event_id}\n"
                f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            )
