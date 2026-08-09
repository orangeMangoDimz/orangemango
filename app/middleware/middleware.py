from __future__ import annotations

import asyncio
from collections import defaultdict, deque
import math
import os
from time import monotonic

from app.config.const.api_res import (
    INVALID_RATE_LIMIT_REQUESTS,
    INVALID_RATE_LIMIT_WINDOW,
    TOO_MANY_CHAT_REQUESTS,
)
from app.config.const.chat import (
    CHAT_RATE_LIMIT_REQUESTS,
    CHAT_RATE_LIMIT_WINDOW_SECONDS,
)
from app.logger import log_exception, logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)
_RATE_LIMITED_PATH = "/message"
_MAX_TRACKED_CLIENTS = 10_000


def _get_allowed_origins() -> list[str]:
    configured_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in configured_origins.split(",") if origin.strip()]
    return origins or list(_DEFAULT_CORS_ORIGINS)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        started_at = monotonic()
        client_ip = request.client.host if request.client else "unknown"
        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = max(0, round((monotonic() - started_at) * 1000))
            logger.opt(exception=exc).error(
                "Unhandled request failure {method} {path} ({duration_ms}ms) ip={client_ip}",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
                client_ip=client_ip,
            )
            raise

        duration_ms = max(0, round((monotonic() - started_at) * 1000))
        logger.info(
            "{method} {path} -> {status} ({duration_ms}ms) ip={client_ip}",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
            client_ip=client_ip,
        )
        if response.status_code >= 500:
            log_exception(
                "API returned server error",
                request=request,
                status_code=response.status_code,
                duration_ms=duration_ms,
                client_ip=client_ip,
            )
        return response


class ChatRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        *,
        max_requests: int,
        window_seconds: int,
    ) -> None:
        super().__init__(app)
        if max_requests < 1:
            raise ValueError(INVALID_RATE_LIMIT_REQUESTS)
        if window_seconds < 1:
            raise ValueError(INVALID_RATE_LIMIT_WINDOW)

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._request_times: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.method != "POST" or request.url.path != _RATE_LIMITED_PATH:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = monotonic()

        async with self._lock:
            timestamps = self._request_times[client_ip]
            cutoff = now - self.window_seconds
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                retry_after = max(
                    1,
                    math.ceil(timestamps[0] + self.window_seconds - now),
                )
                logger.warning(
                    "Rate limit exceeded for {client_ip} on {path}",
                    client_ip=client_ip,
                    path=request.url.path,
                )
                return JSONResponse(
                    status_code=429,
                    content={"detail": TOO_MANY_CHAT_REQUESTS},
                    headers={"Retry-After": str(retry_after)},
                )

            timestamps.append(now)
            if len(self._request_times) > _MAX_TRACKED_CLIENTS:
                stale_clients = [
                    ip for ip, request_times in self._request_times.items() if not request_times
                ]
                for ip in stale_clients:
                    del self._request_times[ip]

        return await call_next(request)


def configure_middleware(app: FastAPI) -> None:
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        ChatRateLimitMiddleware,
        max_requests=CHAT_RATE_LIMIT_REQUESTS,
        window_seconds=CHAT_RATE_LIMIT_WINDOW_SECONDS,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_get_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
