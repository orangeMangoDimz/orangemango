"""Create chat request and response persistence tables."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_chat_persistence"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_threads",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default=sa.text("'active'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('active', 'archived')",
            name="ck_chat_threads_status",
        ),
        sa.CheckConstraint(
            "btrim(id) <> ''",
            name="ck_chat_threads_id_not_blank",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_threads_updated",
        "chat_threads",
        [sa.text("updated_at DESC")],
    )

    op.create_table(
        "chat_requests",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column(
            "provider",
            sa.String(length=32),
            server_default=sa.text("'openai'"),
            nullable=False,
        ),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default=sa.text("'accepted'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "btrim(message) <> '' AND char_length(message) <= 10000",
            name="ck_chat_requests_message",
        ),
        sa.CheckConstraint(
            "status IN ('accepted', 'processing', 'completed', 'failed', 'cancelled')",
            name="ck_chat_requests_status",
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["chat_threads.id"],
            name="fk_chat_requests_thread_id_chat_threads",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_chat_requests_thread_created",
        "chat_requests",
        ["thread_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "uq_active_chat_request_per_thread",
        "chat_requests",
        ["thread_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('accepted', 'processing')"),
    )

    op.create_table(
        "chat_responses",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "request_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("finish_reason", sa.String(length=64), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("provider_request_id", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('completed', 'partial', 'failed')",
            name="ck_chat_responses_status",
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_chat_responses_input_tokens",
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_chat_responses_output_tokens",
        ),
        sa.CheckConstraint(
            "total_tokens IS NULL OR total_tokens >= 0",
            name="ck_chat_responses_total_tokens",
        ),
        sa.CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name="ck_chat_responses_latency_ms",
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["chat_requests.id"],
            name="fk_chat_responses_request_id_chat_requests",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id", name="uq_chat_responses_request"),
    )


def downgrade() -> None:
    op.drop_table("chat_responses")
    op.drop_index(
        "uq_active_chat_request_per_thread",
        table_name="chat_requests",
    )
    op.drop_index(
        "ix_chat_requests_thread_created",
        table_name="chat_requests",
    )
    op.drop_table("chat_requests")
    op.drop_index("ix_chat_threads_updated", table_name="chat_threads")
    op.drop_table("chat_threads")
