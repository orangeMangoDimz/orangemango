"""Store uploaded CV bytes and support asynchronous extraction runs."""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

revision: str = "0004_cv_documents_extract"
down_revision: Union[str, None] = "0003_app_error_logs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(bind: sa.Connection, table_name: str) -> bool:
    if context.is_offline_mode():
        return False
    return sa.inspect(bind).has_table(table_name)


def _column_names(bind: sa.Connection, table_name: str) -> set[str]:
    if context.is_offline_mode() or not _table_exists(bind, table_name):
        return set()
    return {column["name"] for column in sa.inspect(bind).get_columns(table_name)}


def _index_names(bind: sa.Connection, table_name: str) -> set[str]:
    if context.is_offline_mode() or not _table_exists(bind, table_name):
        return set()
    return {index["name"] for index in sa.inspect(bind).get_indexes(table_name)}


def _foreign_key_names(bind: sa.Connection, table_name: str) -> set[str]:
    if context.is_offline_mode() or not _table_exists(bind, table_name):
        return set()
    return {
        foreign_key.get("name")
        for foreign_key in sa.inspect(bind).get_foreign_keys(table_name)
        if foreign_key.get("name")
    }


def _create_cv_documents() -> None:
    op.create_table(
        "cv_documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("thread_id", sa.String(length=128), nullable=True),
        sa.Column(
            "request_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "mime_type",
            sa.String(length=127),
            server_default=sa.text("'application/pdf'"),
            nullable=False,
        ),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("content", sa.LargeBinary(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default=sa.text("'uploaded'"),
            nullable=False,
        ),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('uploaded', 'processing', 'completed', 'failed')",
            name="ck_cv_documents_status",
        ),
        sa.CheckConstraint(
            "btrim(filename) <> ''",
            name="ck_cv_documents_filename_not_blank",
        ),
        sa.CheckConstraint(
            "btrim(storage_key) <> ''",
            name="ck_cv_documents_storage_key_not_blank",
        ),
        sa.CheckConstraint(
            "size_bytes >= 0",
            name="ck_cv_documents_size_bytes",
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["chat_threads.id"],
            name="fk_cv_documents_thread_id_chat_threads",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["chat_requests.id"],
            name="fk_cv_documents_request_id_chat_requests",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "thread_id",
            "sha256",
            name="uq_cv_documents_thread_sha256",
        ),
    )
    op.create_index(
        "ix_cv_documents_thread_uploaded",
        "cv_documents",
        ["thread_id", sa.text("uploaded_at DESC")],
    )


def _create_cv_extraction_runs() -> None:
    op.create_table(
        "cv_extraction_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
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
            server_default=sa.text("'processing'"),
            nullable=False,
        ),
        sa.Column("validation_status", sa.String(length=16), nullable=True),
        sa.Column(
            "raw_result",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "matching_features",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "warnings",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "errors",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "version >= 1",
            name="ck_cv_extraction_runs_version",
        ),
        sa.CheckConstraint(
            "status IN ('processing', 'completed', 'failed')",
            name="ck_cv_extraction_runs_status",
        ),
        sa.CheckConstraint(
            "validation_status IS NULL OR validation_status IN ('valid', 'invalid')",
            name="ck_cv_extraction_runs_validation_status",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["cv_documents.id"],
            name="fk_cv_extraction_runs_document_id_cv_documents",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["thread_id"],
            ["chat_threads.id"],
            name="fk_cv_extraction_runs_thread_id_chat_threads",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "version",
            name="uq_cv_extraction_runs_document_version",
        ),
    )
    op.create_index(
        "ix_cv_extraction_runs_document_version",
        "cv_extraction_runs",
        ["document_id", sa.text("version DESC")],
    )
    op.create_index(
        "uq_active_cv_extraction",
        "cv_extraction_runs",
        ["document_id", "thread_id"],
        unique=True,
        postgresql_where=sa.text("status = 'processing'"),
    )


def upgrade() -> None:
    bind = op.get_bind()

    if not _table_exists(bind, "cv_documents"):
        _create_cv_documents()
    else:
        document_columns = _column_names(bind, "cv_documents")
        required_columns = {
            "id",
            "filename",
            "storage_key",
            "sha256",
            "mime_type",
            "size_bytes",
            "status",
            "uploaded_at",
        }
        missing_columns = required_columns - document_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise RuntimeError(
                f"Existing cv_documents table is incompatible; missing: {missing}"
            )
        if "content" not in document_columns:
            op.add_column(
                "cv_documents",
                sa.Column("content", sa.LargeBinary(), nullable=True),
            )
        if "thread_id" in document_columns:
            op.alter_column(
                "cv_documents",
                "thread_id",
                existing_type=sa.String(length=128),
                nullable=True,
            )

    if not _table_exists(bind, "cv_extraction_runs"):
        _create_cv_extraction_runs()
        return

    extraction_columns = _column_names(bind, "cv_extraction_runs")
    required_columns = {
        "id",
        "document_id",
        "version",
        "provider",
        "model",
        "status",
        "validation_status",
        "raw_result",
        "matching_features",
        "warnings",
        "errors",
        "started_at",
    }
    missing_columns = required_columns - extraction_columns
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(
            f"Existing cv_extraction_runs table is incompatible; missing: {missing}"
        )

    if "thread_id" not in extraction_columns:
        op.add_column(
            "cv_extraction_runs",
            sa.Column("thread_id", sa.String(length=128), nullable=True),
        )
        if "fk_cv_extraction_runs_thread_id_chat_threads" not in _foreign_key_names(
            bind, "cv_extraction_runs"
        ):
            op.create_foreign_key(
                "fk_cv_extraction_runs_thread_id_chat_threads",
                "cv_extraction_runs",
                "chat_threads",
                ["thread_id"],
                ["id"],
                ondelete="CASCADE",
            )

    if "uq_active_cv_extraction" not in _index_names(bind, "cv_extraction_runs"):
        op.create_index(
            "uq_active_cv_extraction",
            "cv_extraction_runs",
            ["document_id", "thread_id"],
            unique=True,
            postgresql_where=sa.text("status = 'processing'"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    if not _table_exists(bind, "cv_documents"):
        return

    has_legacy_cv_schema = _table_exists(bind, "cv_work_experiences")
    if has_legacy_cv_schema:
        if "uq_active_cv_extraction" in _index_names(bind, "cv_extraction_runs"):
            op.drop_index(
                "uq_active_cv_extraction",
                table_name="cv_extraction_runs",
            )
        extraction_columns = _column_names(bind, "cv_extraction_runs")
        if "thread_id" in extraction_columns:
            foreign_keys = _foreign_key_names(bind, "cv_extraction_runs")
            if "fk_cv_extraction_runs_thread_id_chat_threads" in foreign_keys:
                op.drop_constraint(
                    "fk_cv_extraction_runs_thread_id_chat_threads",
                    "cv_extraction_runs",
                    type_="foreignkey",
                )
            op.drop_column("cv_extraction_runs", "thread_id")
        if "content" in _column_names(bind, "cv_documents"):
            op.drop_column("cv_documents", "content")
        op.alter_column(
            "cv_documents",
            "thread_id",
            existing_type=sa.String(length=128),
            nullable=False,
        )
        return

    if _table_exists(bind, "cv_extraction_runs"):
        if "uq_active_cv_extraction" in _index_names(bind, "cv_extraction_runs"):
            op.drop_index(
                "uq_active_cv_extraction",
                table_name="cv_extraction_runs",
            )
        op.drop_table("cv_extraction_runs")
    op.drop_index("ix_cv_documents_thread_uploaded", table_name="cv_documents")
    op.drop_table("cv_documents")
