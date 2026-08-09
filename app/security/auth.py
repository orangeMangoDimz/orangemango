from __future__ import annotations

import hmac
import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config.const.api_res import (
    API_AUTH_NOT_CONFIGURED,
    API_AUTH_TOKEN_ENV,
    INVALID_API_AUTH_TOKEN,
)


_bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="BearerAuth",
    description="Bearer token required for protected chat endpoints.",
)


def _configured_token() -> str:
    load_dotenv(override=False)
    return os.getenv(API_AUTH_TOKEN_ENV, "").strip()


def require_api_token(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(_bearer_scheme),
    ],
) -> None:
    configured_token = _configured_token()
    if not configured_token:
        raise HTTPException(
            status_code=503,
            detail=API_AUTH_NOT_CONFIGURED,
        )

    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not hmac.compare_digest(credentials.credentials, configured_token)
    ):
        raise HTTPException(
            status_code=401,
            detail=INVALID_API_AUTH_TOKEN,
            headers={"WWW-Authenticate": "Bearer"},
        )
