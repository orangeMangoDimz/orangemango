from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config.const.api_path import (
    CV_EXTRACT_PATH,
    CV_EXTRACT_TEXT_PATH,
    CV_EXTRACTIONS_PATH,
    CV_PATH,
)
from app.config.const.api_res import (
    CHAT_SERVICE_NOT_CONFIGURED,
    CV_API_TAG,
    CV_EXTRACTION_BUSY,
    CV_EXTRACTION_DESCRIPTION,
    CV_EXTRACTION_NOT_FOUND,
    CV_EXTRACTION_RESULT_DESCRIPTION,
    CV_EXTRACTION_RESULT_SUMMARY,
    CV_EXTRACTION_SUMMARY,
    CV_FILE_INVALID,
    CV_NOT_FOUND,
    CV_TEXT_EXTRACTION_DESCRIPTION,
    CV_TEXT_EXTRACTION_SUMMARY,
    CV_UPLOAD_DESCRIPTION,
    CV_UPLOAD_SUMMARY,
)
from app.config.const.chat import MAX_CV_FILE_BYTES
from app.controllers.dependencies import service_provider
from app.data.schema.request import CvExtractionRequest
from app.data.schema.response import (
    AcceptedCvExtractionResponse,
    CvExtractionResponse,
    CvProcessResponse,
    CvUploadResponse,
)
from app.repositories.cv_repository import (
    CvExtractionBusyError,
    CvExtractionNotFoundError,
    CvNotFoundError,
    CvPersistenceError,
)
from app.security.auth import require_api_token
from app.services.cv_document import CvInputError
from app.services.cv_service import CvService, CvThreadBusyError


class CvController:
    """Handle CV upload, extraction, and result HTTP endpoints."""

    def __init__(self) -> None:
        self.router: APIRouter = APIRouter(
            dependencies=[Depends(require_api_token)],
        )
        self.router.add_api_route(
            CV_PATH,
            self.cv,
            methods=["POST"],
            response_model=CvUploadResponse,
            status_code=status.HTTP_201_CREATED,
            summary=CV_UPLOAD_SUMMARY,
            description=CV_UPLOAD_DESCRIPTION,
            tags=[CV_API_TAG],
        )
        self.router.add_api_route(
            CV_EXTRACT_TEXT_PATH,
            self.cv_extract_text,
            methods=["POST"],
            response_model=CvProcessResponse,
            summary=CV_TEXT_EXTRACTION_SUMMARY,
            description=CV_TEXT_EXTRACTION_DESCRIPTION,
            tags=[CV_API_TAG],
        )
        self.router.add_api_route(
            CV_EXTRACT_PATH,
            self.cv_extract,
            methods=["POST"],
            response_model=AcceptedCvExtractionResponse,
            status_code=status.HTTP_202_ACCEPTED,
            summary=CV_EXTRACTION_SUMMARY,
            description=CV_EXTRACTION_DESCRIPTION,
            tags=[CV_API_TAG],
        )
        self.router.add_api_route(
            CV_EXTRACTIONS_PATH,
            self.cv_extractions,
            methods=["GET"],
            response_model=CvExtractionResponse,
            summary=CV_EXTRACTION_RESULT_SUMMARY,
            description=CV_EXTRACTION_RESULT_DESCRIPTION,
            tags=[CV_API_TAG],
        )

    async def cv(
        self,
        file: UploadFile = File(...),
        service: CvService = Depends(service_provider.get_cv_service),
    ) -> CvUploadResponse:
        try:
            content: bytes = await file.read(MAX_CV_FILE_BYTES + 1)
            return await service.upload(
                filename=file.filename,
                content_type=file.content_type,
                content=content,
            )
        except CvInputError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc) or CV_FILE_INVALID,
            ) from exc
        except CvPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=CHAT_SERVICE_NOT_CONFIGURED,
            ) from exc
        finally:
            await file.close()

    async def cv_extract_text(
        self,
        cv_id: UUID,
        service: CvService = Depends(service_provider.get_cv_service),
    ) -> CvProcessResponse:
        try:
            await service.process_cv(cv_id)
        except CvNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=CV_NOT_FOUND,
            ) from exc
        except CvInputError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            ) from exc
        except CvExtractionBusyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=CV_EXTRACTION_BUSY,
            ) from exc
        except CvPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=CHAT_SERVICE_NOT_CONFIGURED,
            ) from exc
        return CvProcessResponse(status="ok")

    async def cv_extract(
        self,
        cv_id: UUID,
        request: CvExtractionRequest,
        service: CvService = Depends(service_provider.get_cv_service),
    ) -> AcceptedCvExtractionResponse:
        try:
            return await service.accept_extraction(
                cv_id=cv_id,
                thread_id=request.thread_id,
            )
        except CvNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=CV_NOT_FOUND,
            ) from exc
        except CvThreadBusyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=CV_EXTRACTION_BUSY,
            ) from exc
        except CvPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=CHAT_SERVICE_NOT_CONFIGURED,
            ) from exc

    async def cv_extractions(
        self,
        extraction_id: UUID,
        service: CvService = Depends(service_provider.get_cv_service),
    ) -> CvExtractionResponse:
        try:
            return await service.get_extraction(extraction_id)
        except CvExtractionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=CV_EXTRACTION_NOT_FOUND,
            ) from exc
        except CvPersistenceError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=CHAT_SERVICE_NOT_CONFIGURED,
            ) from exc
