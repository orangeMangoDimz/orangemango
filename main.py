from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.config.const.api_res import OPENAPI_TAGS
from app.controllers.chat_controller import ChatController
from app.controllers.cv_controller import CvController
from app.controllers.dependencies import service_provider
from app.controllers.health_controller import HealthController
from app.logger import configure_logging, log_exception, logger
from app.middleware.middleware import configure_middleware


configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Orangemango API starting")
    try:
        yield
    finally:
        await service_provider.shutdown()
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
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


app.include_router(HealthController().router)
app.include_router(ChatController().router)
app.include_router(CvController().router)
