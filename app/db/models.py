from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgreSQLUUID
from sqlmodel import Field, SQLModel


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ChatThread(SQLModel, table=True):
    __tablename__ = "chat_threads"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'archived')",
            name="ck_chat_threads_status",
        ),
        CheckConstraint(
            "btrim(id) <> ''",
            name="ck_chat_threads_id_not_blank",
        ),
        Index("ix_chat_threads_updated", text("updated_at DESC")),
    )

    id: str = Field(
        sa_column=Column(String(128), primary_key=True, nullable=False),
    )
    status: str = Field(
        default="active",
        sa_column=Column(
            String(16),
            nullable=False,
            server_default=text("'active'"),
        ),
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
    )
    updated_at: datetime = Field(
        default_factory=_utc_now,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
    )


class ChatRequest(SQLModel, table=True):
    __tablename__ = "chat_requests"
    __table_args__ = (
        CheckConstraint(
            "btrim(message) <> '' AND char_length(message) <= 10000",
            name="ck_chat_requests_message",
        ),
        CheckConstraint(
            "status IN ('accepted', 'processing', 'completed', 'failed', 'cancelled')",
            name="ck_chat_requests_status",
        ),
        Index(
            "ix_chat_requests_thread_created",
            "thread_id",
            text("created_at DESC"),
        ),
        Index(
            "uq_active_chat_request_per_thread",
            "thread_id",
            unique=True,
            postgresql_where=text("status IN ('accepted', 'processing')"),
        ),
    )

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(
            PostgreSQLUUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
    )
    thread_id: str = Field(
        sa_column=Column(
            String(128),
            ForeignKey("chat_threads.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    message: str = Field(
        sa_column=Column(Text, nullable=False),
    )
    provider: str = Field(
        default="openai",
        sa_column=Column(
            String(32),
            nullable=False,
            server_default=text("'openai'"),
        ),
    )
    model: str = Field(
        sa_column=Column(String(128), nullable=False),
    )
    status: str = Field(
        default="accepted",
        sa_column=Column(
            String(16),
            nullable=False,
            server_default=text("'accepted'"),
        ),
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
    )
    started_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    finished_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    error_code: str | None = Field(
        default=None,
        sa_column=Column(String(64), nullable=True),
    )
    error_message: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )


class ChatResponse(SQLModel, table=True):
    __tablename__ = "chat_responses"
    __table_args__ = (
        CheckConstraint(
            "status IN ('completed', 'partial', 'failed')",
            name="ck_chat_responses_status",
        ),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_chat_responses_input_tokens",
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_chat_responses_output_tokens",
        ),
        CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0",
            name="ck_chat_responses_total_tokens",
        ),
        CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name="ck_chat_responses_latency_ms",
        ),
        UniqueConstraint("request_id", name="uq_chat_responses_request"),
    )

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(
            PostgreSQLUUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
    )
    request_id: UUID = Field(
        sa_column=Column(
            PostgreSQLUUID(as_uuid=True),
            ForeignKey("chat_requests.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    content: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    status: str = Field(
        sa_column=Column(String(16), nullable=False),
    )
    finish_reason: str | None = Field(
        default=None,
        sa_column=Column(String(64), nullable=True),
    )
    input_tokens: int | None = Field(
        default=None,
        sa_column=Column(Integer, nullable=True),
    )
    output_tokens: int | None = Field(
        default=None,
        sa_column=Column(Integer, nullable=True),
    )
    total_tokens: int | None = Field(
        default=None,
        sa_column=Column(Integer, nullable=True),
    )
    latency_ms: int | None = Field(
        default=None,
        sa_column=Column(Integer, nullable=True),
    )
    provider_request_id: str | None = Field(
        default=None,
        sa_column=Column(String(255), nullable=True),
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
    )
    error_message: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
