from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from app.db.models import CvDocument, CvExtraction


class CvPersistenceError(RuntimeError):
    """Raised when CV data cannot be persisted or loaded."""


class CvNotFoundError(LookupError):
    """Raised when a requested CV does not exist."""


class CvExtractionNotFoundError(LookupError):
    """Raised when a requested extraction does not exist."""


class CvExtractionBusyError(RuntimeError):
    """Raised when an extraction already exists for a CV and thread."""


class CvRepository:
    """Persist CV files and their asynchronous extraction lifecycle."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    async def create_document(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
        extracted_text: str | None = None,
    ) -> CvDocument:
        record = CvDocument(
            filename=filename,
            storage_key="pending",
            sha256=hashlib.sha256(content).hexdigest(),
            mime_type=content_type,
            size_bytes=len(content),
            extracted_text=extracted_text,
            content=content,
        )
        record.storage_key = f"db://cv/{record.id}"
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    session.add(record)
                    await session.flush()
        except SQLAlchemyError as exc:
            raise CvPersistenceError("Unable to persist CV document") from exc
        return record

    async def get_document(self, cv_id: UUID) -> CvDocument:
        try:
            async with self._session_factory() as session:
                record = await session.get(CvDocument, cv_id)
        except SQLAlchemyError as exc:
            raise CvPersistenceError("Unable to load CV document") from exc
        if record is None:
            raise CvNotFoundError(str(cv_id))
        return record

    async def update_extracted_text(
        self,
        cv_id: UUID,
        *,
        extracted_text: str,
    ) -> None:
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    record = await session.get(CvDocument, cv_id)
                    if record is None:
                        raise CvNotFoundError(str(cv_id))
                    record.extracted_text = extracted_text
        except CvNotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise CvPersistenceError("Unable to persist extracted CV text") from exc

    async def create_extraction(
        self,
        *,
        cv_id: UUID,
        thread_id: str | None,
    ) -> CvExtraction:
        record = CvExtraction(
            document_id=cv_id,
            thread_id=thread_id,
            version=0,
            status="processing",
            warnings=[],
        )
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    latest_version = await session.execute(
                        select(
                            func.coalesce(func.max(CvExtraction.version), 0) + 1
                        ).where(CvExtraction.document_id == cv_id)
                    )
                    record.version = int(latest_version.scalar_one())
                    session.add(record)
                    await session.flush()
        except IntegrityError as exc:
            if "uq_active_cv_extraction" in str(exc):
                raise CvExtractionBusyError(thread_id) from exc
            raise CvPersistenceError("Unable to persist CV extraction") from exc
        except SQLAlchemyError as exc:
            raise CvPersistenceError("Unable to persist CV extraction") from exc
        return record

    async def mark_processing(self, extraction_id: UUID) -> None:
        now = datetime.now(UTC)
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    record = await session.get(CvExtraction, extraction_id)
                    if record is None:
                        raise CvExtractionNotFoundError(str(extraction_id))
                    record.status = "processing"
                    record.started_at = now
        except CvExtractionNotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise CvPersistenceError("Unable to update CV extraction") from exc

    async def complete_extraction(
        self,
        extraction_id: UUID,
        *,
        result: dict[str, Any],
        warnings: list[str],
    ) -> None:
        now = datetime.now(UTC)
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    record = await session.get(CvExtraction, extraction_id)
                    if record is None:
                        raise CvExtractionNotFoundError(str(extraction_id))
                    record.status = "completed"
                    record.extraction_result = result
                    matching_features = result.get("matching_features")
                    record.matching_features = (
                        matching_features
                        if isinstance(matching_features, dict)
                        else {}
                    )
                    record.warnings = warnings
                    record.errors = []
                    validation_status = result.get("validation_status")
                    record.validation_status = (
                        validation_status
                        if isinstance(validation_status, str)
                        and validation_status in {"valid", "invalid"}
                        else None
                    )
                    record.finished_at = now
        except CvExtractionNotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise CvPersistenceError("Unable to persist CV extraction result") from exc

    async def fail_extraction(
        self,
        extraction_id: UUID,
        *,
        error_message: str,
    ) -> None:
        now = datetime.now(UTC)
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    record = await session.get(CvExtraction, extraction_id)
                    if record is None:
                        raise CvExtractionNotFoundError(str(extraction_id))
                    record.status = "failed"
                    record.errors = [error_message]
                    record.finished_at = now
        except CvExtractionNotFoundError:
            raise
        except SQLAlchemyError as exc:
            raise CvPersistenceError("Unable to persist failed CV extraction") from exc

    async def get_extraction(self, extraction_id: UUID) -> CvExtraction:
        try:
            async with self._session_factory() as session:
                record = await session.get(CvExtraction, extraction_id)
        except SQLAlchemyError as exc:
            raise CvPersistenceError("Unable to load CV extraction") from exc
        if record is None:
            raise CvExtractionNotFoundError(str(extraction_id))
        return record
