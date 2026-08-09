from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.config.const.api_res import (
    CV_FILE_EMPTY,
    CV_FILE_INVALID,
    CV_FILE_MUST_BE_PDF,
    CV_FILE_TOO_LARGE,
)
from app.config.const.chat import MAX_CV_FILE_BYTES, MAX_CV_FILENAME_LENGTH


class CvInputError(ValueError):
    """Raised when an uploaded CV fails validation or text extraction."""


def validate_pdf_upload(
    *,
    filename: str | None,
    content_type: str | None,
    content: bytes,
) -> str:
    """Validate an uploaded PDF and return its safe original filename."""
    raw_filename = filename or ""
    normalized_filename = raw_filename.replace("\\", "/")
    safe_name = Path(normalized_filename).name

    if (
        not raw_filename
        or normalized_filename != raw_filename
        or safe_name != normalized_filename
        or safe_name in {".", ".."}
        or len(safe_name) > MAX_CV_FILENAME_LENGTH
    ):
        raise CvInputError(CV_FILE_INVALID)

    if Path(safe_name).suffix.casefold() != ".pdf":
        raise CvInputError(CV_FILE_MUST_BE_PDF)

    if content_type not in (None, "", "application/pdf"):
        raise CvInputError(CV_FILE_MUST_BE_PDF)

    if not content:
        raise CvInputError(CV_FILE_EMPTY)
    if len(content) > MAX_CV_FILE_BYTES:
        raise CvInputError(CV_FILE_TOO_LARGE)
    if not content.startswith(b"%PDF-"):
        raise CvInputError(CV_FILE_INVALID)

    try:
        PdfReader(BytesIO(content))
    except Exception as exc:
        raise CvInputError(CV_FILE_INVALID) from exc

    return safe_name


def extract_pdf_text(content: bytes) -> str:
    """Extract and normalize text from validated PDF bytes."""
    try:
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception as exc:
        raise CvInputError(CV_FILE_INVALID) from exc

    if not text:
        raise CvInputError("No text could be extracted from the CV PDF")
    return text
