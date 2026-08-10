"""Support synchronous CV processing without a chat thread."""

from typing import Sequence, Union

from alembic import context, op
import sqlalchemy as sa


revision: str = "0006_cv_sync_processing"
down_revision: Union[str, None] = "0005_schema_comments"
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


def upgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "cv_documents"):
        document_columns = _column_names(bind, "cv_documents")
        if "extracted_text" not in document_columns:
            op.add_column(
                "cv_documents",
                sa.Column("extracted_text", sa.Text(), nullable=True),
            )

    if _table_exists(bind, "cv_extraction_runs"):
        extraction_columns = _column_names(bind, "cv_extraction_runs")
        if "thread_id" in extraction_columns:
            op.alter_column(
                "cv_extraction_runs",
                "thread_id",
                existing_type=sa.String(length=128),
                nullable=True,
            )


def downgrade() -> None:
    bind = op.get_bind()

    if _table_exists(bind, "cv_extraction_runs"):
        extraction_columns = _column_names(bind, "cv_extraction_runs")
        if "thread_id" in extraction_columns:
            op.alter_column(
                "cv_extraction_runs",
                "thread_id",
                existing_type=sa.String(length=128),
                nullable=False,
            )

    if _table_exists(bind, "cv_documents"):
        document_columns = _column_names(bind, "cv_documents")
        if "extracted_text" in document_columns:
            op.drop_column("cv_documents", "extracted_text")
