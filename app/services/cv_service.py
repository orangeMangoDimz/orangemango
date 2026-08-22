from __future__ import annotations

import asyncio
import importlib.util
import sys
from functools import lru_cache
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from app.config.const.api_res import CV_EXTRACTION_ERROR
from app.data.schema.response import (
    AcceptedCvExtractionResponse,
    CvExtractionResponse,
    CvUploadResponse,
)
from app.db.models import CvDocument, CvExtraction
from app.logger import log_exception, logger
from app.repositories.chat_repository import (
    ChatPersistenceError,
    ChatRepository,
    ChatRequestBusyError,
    ThreadNotFoundError,
)
from app.repositories.cv_repository import (
    CvExtractionBusyError,
    CvPersistenceError,
    CvRepository,
)
from app.services.cv_document import (
    extract_pdf_text,
    validate_extracted_text,
    validate_pdf_upload,
)


class CvThreadBusyError(RuntimeError):
    """Raised when a thread already has an active operation."""


@lru_cache(maxsize=1)
def _load_cv_graph() -> Any:
    project_root: Path = Path(__file__).resolve().parents[2]
    graph_path: Path = project_root / "studio" / "cv-extraction" / "graph.py"
    module_name: str = "orangemango_api_cv_extraction"
    existing: ModuleType | None = sys.modules.get(module_name)
    if existing is not None and hasattr(existing, "graph"):
        return existing.graph

    spec: ModuleSpec | None = importlib.util.spec_from_file_location(
        module_name,
        graph_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CV extraction graph from {graph_path}")

    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "graph"):
        raise RuntimeError("CV extraction graph does not export graph")
    return module.graph


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    return str(value)


def _result_without_source_text(value: Any) -> dict[str, Any]:
    result: Any = _json_safe(value)
    if not isinstance(result, dict):
        raise ValueError("CV extraction returned an invalid result")
    result.pop("cv_text", None)
    return result


def _result_warnings(result: dict[str, Any]) -> list[str]:
    warnings: Any = result.get("warnings")
    if not isinstance(warnings, list):
        return []
    return [str(item) for item in warnings if str(item).strip()]


class CvService:
    """Coordinate stored CV documents, extraction runs, and thread events."""

    def __init__(
        self,
        *,
        event_repository: ChatRepository,
        repository: CvRepository,
    ) -> None:
        self._event_repository = event_repository
        self._repository = repository
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def upload(
        self,
        *,
        filename: str | None,
        content_type: str | None,
        content: bytes,
    ) -> CvUploadResponse:
        safe_name: str = validate_pdf_upload(
            filename=filename,
            content_type=content_type,
            content=content,
        )
        extracted_text: str = extract_pdf_text(content)
        record: CvDocument = await self._repository.create_document(
            filename=safe_name,
            content_type="application/pdf",
            content=content,
            extracted_text=extracted_text,
        )
        return CvUploadResponse(
            cv_id=record.id,
            filename=record.filename,
            content_type=record.mime_type,
            file_size=record.size_bytes,
            status="uploaded",
        )

    async def process_cv(self, cv_id: UUID) -> None:
        """Synchronously extract and persist one CV."""
        cv_text: str = await self._get_extracted_text(cv_id)
        extraction: CvExtraction = await self._repository.create_extraction(
            cv_id=cv_id,
            thread_id=None,
        )
        try:
            await self._repository.mark_processing(extraction.id)
            await self._run_agent(
                cv_text=cv_text,
                extraction_id=extraction.id,
            )
        except asyncio.CancelledError:
            await self._persist_failed_run(
                extraction_id=extraction.id,
                cv_id=cv_id,
            )
            raise
        except Exception as exc:
            log_exception(
                "Synchronous CV extraction failed",
                exc=exc,
                cv_id=str(cv_id),
                extraction_id=str(extraction.id),
            )
            await self._persist_failed_run(
                extraction_id=extraction.id,
                cv_id=cv_id,
            )
            raise

    async def accept_extraction(
        self,
        *,
        cv_id: UUID,
        thread_id: str,
    ) -> AcceptedCvExtractionResponse:
        await self._repository.get_document(cv_id)
        try:
            await self._event_repository.begin_activity(thread_id)
        except ChatRequestBusyError as exc:
            raise CvThreadBusyError(thread_id) from exc
        except ChatPersistenceError as exc:
            raise CvPersistenceError("Unable to reserve chat thread") from exc

        try:
            extraction: CvExtraction = await self._repository.create_extraction(
                cv_id=cv_id,
                thread_id=thread_id,
            )
        except CvExtractionBusyError as exc:
            await self._event_repository.end_run(thread_id)
            raise CvThreadBusyError(thread_id) from exc
        except Exception:
            await self._event_repository.end_run(thread_id)
            raise

        try:
            task: asyncio.Task[None] = asyncio.create_task(
                self._run_extraction(
                    cv_id=cv_id,
                    thread_id=thread_id,
                    extraction_id=extraction.id,
                )
            )
        except BaseException:
            await self._repository.fail_extraction(
                extraction.id,
                error_message=CV_EXTRACTION_ERROR,
            )
            await self._event_repository.end_run(thread_id)
            raise

        self._tasks[thread_id] = task
        logger.info(
            "Accepted CV extraction cv_id={cv_id} extraction_id={extraction_id} "
            "thread_id={thread_id}",
            cv_id=str(cv_id),
            extraction_id=str(extraction.id),
            thread_id=thread_id,
        )
        return AcceptedCvExtractionResponse(
            extraction_id=extraction.id,
            cv_id=cv_id,
            thread_id=thread_id,
            status="accepted",
        )

    async def get_extraction(self, extraction_id: UUID) -> CvExtractionResponse:
        record: CvExtraction = await self._repository.get_extraction(extraction_id)
        return CvExtractionResponse(
            extraction_id=record.id,
            cv_id=record.document_id,
            thread_id=record.thread_id or "",
            status=record.status,
            result=record.extraction_result,
            warnings=list(record.warnings or []),
            error_message=(record.errors[0] if record.errors else None),
            created_at=record.started_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
        )

    async def load_latest_valid_thread_context(
        self,
        thread_id: str,
    ) -> dict[str, Any] | None:
        """Load the persisted CV context that should be active for a chat thread."""
        extraction: CvExtraction | None = (
            await self._repository.get_latest_valid_extraction_for_thread(
            thread_id
            )
        )
        if extraction is None:
            return None

        document: CvDocument = await self._repository.get_document(extraction.document_id)
        cv_text: str = validate_extracted_text(document.extracted_text)
        cv_result: dict[str, Any] = _result_without_source_text(
            extraction.extraction_result
        )
        matching_features: Any = extraction.matching_features
        if not isinstance(matching_features, dict) or not matching_features:
            raise CvPersistenceError("Stored CV extraction has no matching features")

        cv_result["matching_features"] = _json_safe(matching_features)
        cv_result["validation_status"] = "valid"
        return {
            "cv": {
                "documents": [
                    {
                        "id": str(extraction.document_id),
                        "filename": str(document.filename or "cv.pdf"),
                        "cv_text": cv_text,
                        "cv_result": cv_result,
                        "cv_features": _json_safe(matching_features),
                        "cv_review": None,
                    }
                ],
                "needs_extraction": False,
            }
        }

    async def _get_extracted_text(self, cv_id: UUID) -> str:
        document: CvDocument = await self._repository.get_document(cv_id)
        if document.extracted_text and document.extracted_text.strip():
            return validate_extracted_text(document.extracted_text)

        extracted_text: str = extract_pdf_text(document.content or b"")
        await self._repository.update_extracted_text(
            cv_id,
            extracted_text=extracted_text,
        )
        persisted: CvDocument = await self._repository.get_document(cv_id)
        return validate_extracted_text(persisted.extracted_text)

    async def _run_agent(
        self,
        *,
        cv_text: str,
        extraction_id: UUID,
    ) -> dict[str, Any]:
        graph: Any = _load_cv_graph()
        raw_result: Any = await graph.ainvoke({"cv_text": cv_text})
        result: dict[str, Any] = _result_without_source_text(raw_result)
        await self._repository.complete_extraction(
            extraction_id,
            result=result,
            warnings=_result_warnings(result),
        )
        return result

    async def _run_extraction(
        self,
        *,
        cv_id: UUID,
        thread_id: str,
        extraction_id: UUID,
    ) -> None:
        try:
            await self._repository.mark_processing(extraction_id)
            cv_text: str = await self._get_extracted_text(cv_id)
            result: dict[str, Any] = await self._run_agent(
                cv_text=cv_text,
                extraction_id=extraction_id,
            )
            await self._event_repository.publish(
                thread_id,
                "cv_extraction",
                {
                    "thread_id": thread_id,
                    "extraction_id": str(extraction_id),
                    "cv_id": str(cv_id),
                    "status": "completed",
                    "result": result,
                },
            )
            await self._event_repository.publish(
                thread_id,
                "done",
                {
                    "thread_id": thread_id,
                    "extraction_id": str(extraction_id),
                    "operation": "cv_extraction",
                    "status": "completed",
                },
            )
        except asyncio.CancelledError:
            await self._fail_run(
                extraction_id=extraction_id,
                thread_id=thread_id,
                cv_id=cv_id,
                error=CV_EXTRACTION_ERROR,
            )
            raise
        except Exception as exc:
            log_exception(
                "CV extraction failed",
                exc=exc,
                cv_id=str(cv_id),
                extraction_id=str(extraction_id),
                thread_id=thread_id,
            )
            await self._fail_run(
                extraction_id=extraction_id,
                thread_id=thread_id,
                cv_id=cv_id,
                error=CV_EXTRACTION_ERROR,
            )
        finally:
            await self._event_repository.end_run(thread_id)
            current_task: asyncio.Task[None] | None = asyncio.current_task()
            if self._tasks.get(thread_id) is current_task:
                self._tasks.pop(thread_id, None)

    async def _persist_failed_run(
        self,
        *,
        extraction_id: UUID,
        cv_id: UUID,
        thread_id: str | None = None,
        error: str = CV_EXTRACTION_ERROR,
    ) -> None:
        try:
            await self._repository.fail_extraction(
                extraction_id,
                error_message=error,
            )
        except Exception as exc:
            log_exception(
                "Unable to persist failed CV extraction",
                exc=exc,
                cv_id=str(cv_id),
                extraction_id=str(extraction_id),
                thread_id=thread_id,
            )

    async def _fail_run(
        self,
        *,
        extraction_id: UUID,
        thread_id: str,
        cv_id: UUID,
        error: str,
    ) -> None:
        await self._persist_failed_run(
            extraction_id=extraction_id,
            cv_id=cv_id,
            thread_id=thread_id,
            error=error,
        )

        try:
            await self._event_repository.publish(
                thread_id,
                "error",
                {
                    "thread_id": thread_id,
                    "extraction_id": str(extraction_id),
                    "operation": "cv_extraction",
                    "message": error,
                },
            )
            await self._event_repository.publish(
                thread_id,
                "done",
                {
                    "thread_id": thread_id,
                    "extraction_id": str(extraction_id),
                    "operation": "cv_extraction",
                    "status": "failed",
                },
            )
        except ThreadNotFoundError as exc:
            log_exception(
                "Unable to publish CV extraction failure",
                exc=exc,
                cv_id=str(cv_id),
                extraction_id=str(extraction_id),
                thread_id=thread_id,
            )

    async def shutdown(self) -> None:
        tasks: list[asyncio.Task[None]] = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
