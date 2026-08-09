from __future__ import annotations

import logging
import sys
import threading
import traceback
from types import TracebackType
from typing import Any
from uuid import uuid4

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from starlette.requests import Request

from app.db.models import AppErrorLog
from app.db.session import DatabaseConfigurationError, sync_database_url


_CONFIGURED = False
_ENGINE: Engine | None = None
_ENGINE_LOCK = threading.Lock()

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def _get_engine() -> Engine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = create_engine(
                sync_database_url(),
                pool_pre_ping=True,
                pool_size=2,
                max_overflow=2,
            )
        return _ENGINE


def configure_logging(*, level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=_LOG_FORMAT,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False

    _CONFIGURED = True


def _request_fields(request: Request | None) -> dict[str, Any]:
    if request is None:
        return {
            "method": None,
            "path": None,
            "client_ip": None,
        }
    return {
        "method": request.method,
        "path": request.url.path,
        "client_ip": request.client.host if request.client else None,
    }


def _format_traceback(
    exc: BaseException | None,
    exc_info: tuple[type[BaseException], BaseException, TracebackType | None]
    | bool
    | None,
) -> tuple[str | None, str | None]:
    if exc is not None:
        return (
            type(exc).__name__,
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )
    if exc_info is True:
        return (
            None,
            "".join(traceback.format_exception(*sys.exc_info())),
        )
    if isinstance(exc_info, tuple) and exc_info[0] is not None:
        return (
            exc_info[0].__name__,
            "".join(traceback.format_exception(*exc_info)),
        )
    return None, None


def persist_error_log(
    message: str,
    *,
    level: str = "ERROR",
    exc: BaseException | None = None,
    exc_info: tuple[type[BaseException], BaseException, TracebackType | None]
    | bool
    | None = None,
    request: Request | None = None,
    status_code: int | None = None,
    thread_id: str | None = None,
    request_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    exception_type, formatted_traceback = _format_traceback(exc, exc_info)
    request_fields = _request_fields(request)
    normalized_level = level.upper()
    if normalized_level not in {"ERROR", "CRITICAL"}:
        normalized_level = "ERROR"

    try:
        engine = _get_engine()
    except DatabaseConfigurationError:
        logger.warning("Skipping error log persistence; DATABASE_URL is not configured")
        return

    row = AppErrorLog(
        id=uuid4(),
        level=normalized_level,
        message=message,
        exception_type=exception_type,
        traceback=formatted_traceback,
        method=request_fields["method"],
        path=request_fields["path"],
        status_code=status_code,
        client_ip=request_fields["client_ip"],
        thread_id=thread_id,
        request_id=request_id,
        context=context or {},
    )

    try:
        with Session(engine) as session:
            session.add(row)
            session.commit()
    except Exception:
        logger.opt(exception=True).warning("Failed to persist application error log")


def log_exception(
    message: str,
    *,
    exc: BaseException | None = None,
    exc_info: tuple[type[BaseException], BaseException, TracebackType | None]
    | bool
    | None = None,
    request: Request | None = None,
    status_code: int | None = None,
    thread_id: str | None = None,
    request_id: str | None = None,
    level: str = "ERROR",
    **context: Any,
) -> None:
    bound = logger.bind(
        thread_id=thread_id,
        request_id=request_id,
        status_code=status_code,
        **{key: value for key, value in context.items() if value is not None},
    )
    if exc is not None:
        bound.opt(exception=exc).log(level.upper(), message)
    elif exc_info:
        bound.opt(exception=True).log(level.upper(), message)
    else:
        bound.log(level.upper(), message)

    persist_error_log(
        message,
        level=level,
        exc=exc,
        exc_info=exc_info if exc is None else None,
        request=request,
        status_code=status_code,
        thread_id=thread_id,
        request_id=request_id,
        context=context,
    )


__all__ = [
    "configure_logging",
    "log_exception",
    "logger",
    "persist_error_log",
]
