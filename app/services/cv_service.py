from __future__ import annotations

import asyncio
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from app.config.const.api_res import CV_EXTRACTION_ERROR
from app.data.schema.response import (
    AcceptedCvExtractionResponse,
    CvExtractionResponse,
    CvUploadResponse,
)
from app.logger import log_exception, logger
from app.repositories.chat_repository import (
    ChatRequestBusyError,
    ChatPersistenceError,
    ChatRepository,
    ThreadNotFoundError,
)
from app.repositories.cv_repository import (
    CvExtractionBusyError,
    CvPersistenceError,
    CvRepository,
)
from app.services.cv_document import extract_pdf_text, validate_pdf_upload


class CvThreadBusyError(RuntimeError):
    """Raised when a thread already has an active operation."""


@lru_cache(maxsize=1)
def _load_cv_graph() -> Any:
    project_root = Path(__file__).resolve().parents[2]
    graph_path = project_root / "studio" / "cv-extraction" / "graph.py"
    module_name = "orangemango_api_cv_extraction"
    existing = sys.modules.get(module_name)
    if existing is not None and hasattr(existing, "graph"):
        return existing.graph

    spec = importlib.util.spec_from_file_location(module_name, graph_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load CV extraction graph from {graph_path}")

    module = importlib.util.module_from_spec(spec)
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
    result = _json_safe(value)
    if not isinstance(result, dict):
        raise ValueError("CV extraction returned an invalid result")
    result.pop("cv_text", None)
    return result


def _result_warnings(result: dict[str, Any]) -> list[str]:
    warnings = result.get("warnings")
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
        safe_name = validate_pdf_upload(
            filename=filename,
            content_type=content_type,
            content=content,
        )
        record = await self._repository.create_document(
            filename=safe_name,
            content_type="application/pdf",
            content=content,
        )
        return CvUploadResponse(
            cv_id=record.id,
            filename=record.filename,
            content_type=record.mime_type,
            file_size=record.size_bytes,
            status="uploaded",
        )

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
            extraction = await self._repository.create_extraction(
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
            task = asyncio.create_task(
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
        record = await self._repository.get_extraction(extraction_id)
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

    async def _run_extraction(
        self,
        *,
        cv_id: UUID,
        thread_id: str,
        extraction_id: UUID,
    ) -> None:
        try:
            await self._repository.mark_processing(extraction_id)
            document = await self._repository.get_document(cv_id)
            cv_text = extract_pdf_text(document.content or b"")
            graph = _load_cv_graph()
            raw_result = await graph.ainvoke({"cv_text": cv_text})
            result = _result_without_source_text(raw_result)
            warnings = _result_warnings(result)

            await self._repository.complete_extraction(
                extraction_id,
                result=result,
                warnings=warnings,
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
            current_task = asyncio.current_task()
            if self._tasks.get(thread_id) is current_task:
                self._tasks.pop(thread_id, None)

    async def _fail_run(
        self,
        *,
        extraction_id: UUID,
        thread_id: str,
        cv_id: UUID,
        error: str,
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
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
