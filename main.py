from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from app.config.const.api_res import (
    CHAT_API_TAG,
    CHAT_ENDPOINT_DESCRIPTION,
    CHAT_ENDPOINT_SUMMARY,
    CHAT_SERVICE_NOT_CONFIGURED,
    CHAT_THREAD_BUSY,
    CHAT_THREAD_NOT_FOUND,
    CV_API_TAG,
    CV_EXTRACTION_BUSY,
    CV_EXTRACTION_DESCRIPTION,
    CV_EXTRACTION_NOT_FOUND,
    CV_EXTRACTION_RESULT_DESCRIPTION,
    CV_EXTRACTION_RESULT_SUMMARY,
    CV_EXTRACTION_SUMMARY,
    CV_FILE_INVALID,
    CV_NOT_FOUND,
    CV_UPLOAD_DESCRIPTION,
    CV_UPLOAD_SUMMARY,
    EVENT_API_TAG,
    EVENT_ENDPOINT_DESCRIPTION,
    EVENT_ENDPOINT_SUMMARY,
    HEALTH_API_TAG,
    HEALTH_ENDPOINT_DESCRIPTION,
    HEALTH_ENDPOINT_SUMMARY,
    HEALTH_STATUS_OK,
    OPENAPI_TAGS,
    THREAD_ID_MUST_NOT_BE_BLANK,
)
from app.config.const.chat import MAX_CV_FILE_BYTES, MAX_THREAD_ID_LENGTH
from app.data.schema.request import CvExtractionRequest, MessageRequest
from app.data.schema.response import (
    AcceptedCvExtractionResponse,
    AcceptedMessageResponse,
    CvExtractionResponse,
    CvUploadResponse,
    HealthResponse,
)
from app.db.session import DatabaseConfigurationError
from app.logger import configure_logging, log_exception, logger
from app.models.chat_model import ChatConfigurationError
from app.repositories.chat_repository import ChatPersistenceError
from app.repositories.cv_repository import (
    CvExtractionNotFoundError,
    CvNotFoundError,
    CvPersistenceError,
)
from app.security.auth import require_api_token
from app.services.chat_service import (
    ChatService,
    ChatThreadBusyError,
    ChatThreadNotFoundError,
)
from app.services.cv_document import CvInputError
from app.services.cv_service import CvService, CvThreadBusyError
from app.middleware.middleware import configure_middleware
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse


configure_logging()

_chat_service: ChatService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Orangemango API starting")
    try:
        yield
    finally:
        if _chat_service is not None:
            await _chat_service.shutdown()
        logger.info("Orangemango API stopped")


app = FastAPI(
    title="Orangemango API",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)
configure_middleware(app)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    log_exception(
        "Unhandled API exception",
        exc=exc,
        request=request,
        status_code=500,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        try:
            _chat_service = ChatService.from_environment()
        except (ChatConfigurationError, DatabaseConfigurationError) as exc:
            raise HTTPException(
                status_code=503,
                detail=CHAT_SERVICE_NOT_CONFIGURED,
            ) from exc
    return _chat_service


def get_cv_service() -> CvService:
    return get_chat_service().cv_service


def _normalize_thread_id(thread_id: str) -> str:
    normalized = thread_id.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail=THREAD_ID_MUST_NOT_BE_BLANK)
    return normalized


def _last_event_id(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


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


@app.get(
    "/healthz",
    response_model=HealthResponse,
    summary=HEALTH_ENDPOINT_SUMMARY,
    description=HEALTH_ENDPOINT_DESCRIPTION,
    tags=[HEALTH_API_TAG],
)
async def healthz() -> HealthResponse:
    return HealthResponse(status=HEALTH_STATUS_OK)


@app.post(
    "/message",
    response_model=AcceptedMessageResponse,
    status_code=202,
    summary=CHAT_ENDPOINT_SUMMARY,
    description=CHAT_ENDPOINT_DESCRIPTION,
    tags=[CHAT_API_TAG],
    dependencies=[Depends(require_api_token)],
)
async def post_message(
    request: MessageRequest,
    service: ChatService = Depends(get_chat_service),
) -> AcceptedMessageResponse:
    try:
        return await service.accept_message(request.thread_id, request.message)
    except ChatThreadBusyError as exc:
        raise HTTPException(status_code=409, detail=CHAT_THREAD_BUSY) from exc
    except ChatPersistenceError as exc:
        raise HTTPException(
            status_code=503,
            detail=CHAT_SERVICE_NOT_CONFIGURED,
        ) from exc


@app.post(
    "/cv",
    response_model=CvUploadResponse,
    status_code=201,
    summary=CV_UPLOAD_SUMMARY,
    description=CV_UPLOAD_DESCRIPTION,
    tags=[CV_API_TAG],
    dependencies=[Depends(require_api_token)],
)
async def upload_cv(
    file: UploadFile = File(...),
    service: CvService = Depends(get_cv_service),
) -> CvUploadResponse:
    try:
        content = await file.read(MAX_CV_FILE_BYTES + 1)
        return await service.upload(
            filename=file.filename,
            content_type=file.content_type,
            content=content,
        )
    except CvInputError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc) or CV_FILE_INVALID,
        ) from exc
    except CvPersistenceError as exc:
        raise HTTPException(
            status_code=503,
            detail=CHAT_SERVICE_NOT_CONFIGURED,
        ) from exc
    finally:
        await file.close()


@app.post(
    "/cv/{cv_id}/extract",
    response_model=AcceptedCvExtractionResponse,
    status_code=202,
    summary=CV_EXTRACTION_SUMMARY,
    description=CV_EXTRACTION_DESCRIPTION,
    tags=[CV_API_TAG],
    dependencies=[Depends(require_api_token)],
)
async def start_cv_extraction(
    cv_id: UUID,
    request: CvExtractionRequest,
    service: CvService = Depends(get_cv_service),
) -> AcceptedCvExtractionResponse:
    try:
        return await service.accept_extraction(
            cv_id=cv_id,
            thread_id=request.thread_id,
        )
    except CvNotFoundError as exc:
        raise HTTPException(status_code=404, detail=CV_NOT_FOUND) from exc
    except CvThreadBusyError as exc:
        raise HTTPException(status_code=409, detail=CV_EXTRACTION_BUSY) from exc
    except CvPersistenceError as exc:
        raise HTTPException(
            status_code=503,
            detail=CHAT_SERVICE_NOT_CONFIGURED,
        ) from exc


@app.get(
    "/cv/extractions/{extraction_id}",
    response_model=CvExtractionResponse,
    summary=CV_EXTRACTION_RESULT_SUMMARY,
    description=CV_EXTRACTION_RESULT_DESCRIPTION,
    tags=[CV_API_TAG],
    dependencies=[Depends(require_api_token)],
)
async def get_cv_extraction(
    extraction_id: UUID,
    service: CvService = Depends(get_cv_service),
) -> CvExtractionResponse:
    try:
        return await service.get_extraction(extraction_id)
    except CvExtractionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=CV_EXTRACTION_NOT_FOUND,
        ) from exc
    except CvPersistenceError as exc:
        raise HTTPException(
            status_code=503,
            detail=CHAT_SERVICE_NOT_CONFIGURED,
        ) from exc


@app.get(
    "/events",
    response_class=StreamingResponse,
    summary=EVENT_ENDPOINT_SUMMARY,
    description=EVENT_ENDPOINT_DESCRIPTION,
    tags=[EVENT_API_TAG],
    dependencies=[Depends(require_api_token)],
)
async def get_events(
    thread_id: str = Query(..., min_length=1, max_length=MAX_THREAD_ID_LENGTH),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    normalized_thread_id = _normalize_thread_id(thread_id)
    try:
        await service.ensure_thread(normalized_thread_id)
    except ChatThreadNotFoundError as exc:
        raise HTTPException(status_code=404, detail=CHAT_THREAD_NOT_FOUND) from exc
    except ChatPersistenceError as exc:
        raise HTTPException(
            status_code=503,
            detail=CHAT_SERVICE_NOT_CONFIGURED,
        ) from exc

    return StreamingResponse(
        _format_sse(
            service.events(
                normalized_thread_id,
                _last_event_id(last_event_id),
            )
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
