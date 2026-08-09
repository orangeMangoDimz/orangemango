from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession


class DatabaseConfigurationError(RuntimeError):
    """Raised when the application cannot construct a database connection."""


def _raw_database_url() -> str:
    load_dotenv(override=False)
    value = os.getenv("DATABASE_URL", "").strip()
    if not value:
        raise DatabaseConfigurationError("DATABASE_URL is not configured")
    return value


def _postgres_url(value: str, driver: str) -> str:
    normalized = value
    if normalized.startswith("postgres://"):
        normalized = "postgresql://" + normalized.removeprefix("postgres://")

    for scheme in (
        "postgresql+asyncpg://",
        "postgresql+psycopg://",
        "postgresql+psycopg2://",
        "postgresql://",
    ):
        if normalized.startswith(scheme):
            return f"postgresql+{driver}://{normalized[len(scheme) :]}"

    raise DatabaseConfigurationError("DATABASE_URL must use PostgreSQL")


def async_database_url() -> str:
    """Return the configured URL using the asyncpg SQLAlchemy driver."""
    return _postgres_url(_raw_database_url(), "asyncpg")


def sync_database_url() -> str:
    """Return the configured URL using the psycopg SQLAlchemy driver."""
    return _postgres_url(_raw_database_url(), "psycopg")


def postgres_checkpointer_url() -> str:
    """Return a driver-neutral PostgreSQL URL for psycopg-based libraries."""
    normalized = _raw_database_url()
    if normalized.startswith("postgres://"):
        return "postgresql://" + normalized.removeprefix("postgres://")

    for scheme in (
        "postgresql+asyncpg://",
        "postgresql+psycopg://",
        "postgresql+psycopg2://",
        "postgresql://",
    ):
        if normalized.startswith(scheme):
            return "postgresql://" + normalized[len(scheme) :]

    raise DatabaseConfigurationError("DATABASE_URL must use PostgreSQL")


class Database:
    """Own the application engine and per-operation async session factory."""

    def __init__(self, *, url: str | None = None) -> None:
        self.engine: AsyncEngine = create_async_engine(
            url or async_database_url(),
            pool_pre_ping=True,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @classmethod
    def from_environment(cls) -> Database:
        return cls()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session

    async def dispose(self) -> None:
        await self.engine.dispose()
