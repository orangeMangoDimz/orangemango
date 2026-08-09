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
    # Add the limiter first so CORS remains the outer layer and decorates 429 responses.
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
