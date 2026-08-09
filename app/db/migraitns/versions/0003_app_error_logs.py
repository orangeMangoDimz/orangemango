"""Create application error log persistence table."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0003_app_error_logs"
down_revision: Union[str, None] = "0002_chat_history"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "app_error_logs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("level", sa.String(length=16), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("exception_type", sa.String(length=255), nullable=True),
        sa.Column("traceback", sa.Text(), nullable=True),
        sa.Column("method", sa.String(length=16), nullable=True),
        sa.Column("path", sa.String(length=512), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=True),
        sa.Column("client_ip", sa.String(length=64), nullable=True),
        sa.Column("thread_id", sa.String(length=128), nullable=True),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column(
            "context",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "level IN ('ERROR', 'CRITICAL')",
            name="ck_app_error_logs_level",
        ),
        sa.CheckConstraint(
            "btrim(message) <> ''",
            name="ck_app_error_logs_message_not_blank",
        ),
        sa.CheckConstraint(
            "status_code IS NULL OR (status_code >= 100 AND status_code <= 599)",
            name="ck_app_error_logs_status_code",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_app_error_logs_created",
        "app_error_logs",
        [sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_app_error_logs_path_created",
        "app_error_logs",
        ["path", sa.text("created_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_app_error_logs_path_created", table_name="app_error_logs")
    op.drop_index("ix_app_error_logs_created", table_name="app_error_logs")
    op.drop_table("app_error_logs")
