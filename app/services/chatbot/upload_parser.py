"""Parse CV uploads out of chat file blocks and sanitize persisted messages.

Deliberately state-free: ``app.models.chatbot.state`` imports this module for the
``add_chat_messages`` reducer, so nothing here may import state.
"""

from __future__ import annotations

import base64
import binascii
import uuid
from pathlib import Path
from typing import Any

from app.config.const.chatbot import PDF_UPLOAD_MARKER
from app.services.cv_document import extract_pdf_text, validate_pdf_upload


class UploadParser:
    """Stateless extraction and sanitization of PDF CV uploads."""

    @staticmethod
    def _decode_upload_content(value: Any) -> bytes:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Uploaded CV content must be base64-encoded PDF bytes")

        encoded: str = value.strip()
        if encoded.startswith("data:"):
            try:
                encoded = encoded.split(",", 1)[1]
            except IndexError as exc:
                raise ValueError("Uploaded CV data URL is malformed") from exc
        encoded = "".join(encoded.split())
        try:
            return base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Uploaded CV content is not valid base64") from exc


    @staticmethod
    def _block_as_dict(block: Any) -> dict[str, Any] | None:
        if isinstance(block, dict):
            return block
        if hasattr(block, "model_dump"):
            dumped: Any = block.model_dump()
            return dumped if isinstance(dumped, dict) else None
        if hasattr(block, "dict"):
            dumped: Any = block.dict()
            return dumped if isinstance(dumped, dict) else None
        return None


    @staticmethod
    def _is_file_block(block: dict[str, Any]) -> bool:
        block_type: str = str(block.get("type") or "").casefold()
        if block_type == "file":
            return True
        mime: str = str(block.get("mime_type") or block.get("mimeType") or "").casefold()
        return mime == "application/pdf"


    @staticmethod
    def _nested_file_dict(block: dict[str, Any]) -> dict[str, Any] | None:
        nested: Any = block.get("file")
        return nested if isinstance(nested, dict) else None


    @staticmethod
    def _file_block_filename(block: dict[str, Any]) -> str:
        nested: dict[str, Any] = UploadParser._nested_file_dict(block) or {}
        metadata: Any = block.get("metadata")
        metadata_name: str = ""
        if isinstance(metadata, dict):
            metadata_name = str(
                metadata.get("filename")
                or metadata.get("name")
                or metadata.get("title")
                or ""
            )
        extras: Any = block.get("extras")
        if not metadata_name and isinstance(extras, dict):
            nested_meta: Any = extras.get("metadata")
            if isinstance(nested_meta, dict):
                metadata_name = str(
                    nested_meta.get("filename")
                    or nested_meta.get("name")
                    or nested_meta.get("title")
                    or ""
                )
        return str(
            block.get("filename")
            or block.get("name")
            or nested.get("filename")
            or nested.get("name")
            or metadata_name
            or "cv.pdf"
        )


    @staticmethod
    def _payload_from_mapping(mapping: dict[str, Any]) -> Any:
        for key in (
            "file_data",
            "content_base64",
            "data",
            "base64",
            "content",
            "url",
        ):
            value: Any = mapping.get(key)
            if value not in (None, ""):
                return value
        source: Any = mapping.get("source")
        if isinstance(source, dict):
            return UploadParser._payload_from_mapping(source)
        return None


    @staticmethod
    def _file_block_payload(block: dict[str, Any]) -> Any:
        payload: Any = UploadParser._payload_from_mapping(block)
        if payload not in (None, ""):
            return payload
        nested: dict[str, Any] | None = UploadParser._nested_file_dict(block)
        if nested is not None:
            return UploadParser._payload_from_mapping(nested)
        return None


    @staticmethod
    def upload_from_file_block(block: Any) -> dict[str, Any] | None:
        parsed: dict[str, Any] | None = UploadParser._block_as_dict(block)
        if parsed is None or not UploadParser._is_file_block(parsed):
            return None
        payload: Any = UploadParser._file_block_payload(parsed)
        if payload in (None, ""):
            return None
        filename: str = UploadParser._file_block_filename(parsed)
        safe_name: str = Path(str(filename).replace("\\", "/")).name or "cv.pdf"
        if Path(safe_name).suffix.casefold() != ".pdf":
            safe_name = f"{safe_name}.pdf" if safe_name else "cv.pdf"
        return {"filename": safe_name, "content_base64": payload}


    @staticmethod
    def message_content_blocks(message: Any) -> list[Any]:
        content: Any = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, list):
            return content
        return []


    @staticmethod
    def extract_uploads_from_message(message: Any) -> list[dict[str, Any]]:
        uploads: list[dict[str, Any]] = []
        for block in UploadParser.message_content_blocks(message):
            upload: dict[str, Any] | None = UploadParser.upload_from_file_block(block)
            if upload is not None:
                uploads.append(upload)
        return uploads


    @staticmethod
    def extract_upload_from_message(message: Any) -> dict[str, Any] | None:
        uploads: list[dict[str, Any]] = UploadParser.extract_uploads_from_message(message)
        return uploads[0] if uploads else None


    @staticmethod
    def read_uploaded_cv(uploaded_file: Any) -> str:
        if not isinstance(uploaded_file, dict):
            raise ValueError(
                "pending_cv_upload must contain filename and base64 PDF content"
            )

        filename: str = str(
            uploaded_file.get("filename") or uploaded_file.get("name") or ""
        )
        safe_name: str = Path(filename.replace("\\", "/")).name
        if (
            not filename
            or safe_name != filename
            or Path(safe_name).suffix.casefold() != ".pdf"
        ):
            raise ValueError("Only .pdf CV uploads are supported")

        content: Any = uploaded_file.get("content_base64")
        if content is None:
            content = uploaded_file.get("content")
        if content is None:
            content = uploaded_file.get("data")
        if content is None:
            content = uploaded_file.get("base64")
        payload: bytes = UploadParser._decode_upload_content(content)
        validate_pdf_upload(
            filename=filename,
            content_type="application/pdf",
            content=payload,
        )
        return extract_pdf_text(payload)


    @staticmethod
    def cv_document_from_upload(uploaded_file: Any) -> dict[str, Any]:
        filename: str = str(
            uploaded_file.get("filename") or uploaded_file.get("name") or ""
        )
        safe_name: str = Path(filename.replace("\\", "/")).name or "cv.pdf"
        return {
            "id": str(uuid.uuid4()),
            "filename": safe_name,
            "cv_text": UploadParser.read_uploaded_cv(uploaded_file),
            "cv_result": None,
            "cv_features": None,
            "cv_review": None,
        }


    @staticmethod
    def with_message_content(message: Any, content: str | list[dict[str, Any]]) -> Any:
        if isinstance(message, dict):
            updated: dict[str, Any] = {**message, "content": content}
            additional: dict[str, Any] = dict(updated.get("additional_kwargs") or {})
            additional.pop("pending_cv_upload", None)
            additional.pop("pending_cv_uploads", None)
            if additional:
                updated["additional_kwargs"] = additional
            elif "additional_kwargs" in updated:
                updated = {
                    key: value
                    for key, value in updated.items()
                    if key != "additional_kwargs"
                }
            return updated

        additional: dict[str, Any] = dict(getattr(message, "additional_kwargs", None) or {})
        additional.pop("pending_cv_upload", None)
        additional.pop("pending_cv_uploads", None)
        if hasattr(message, "model_copy"):
            return message.model_copy(
                update={"content": content, "additional_kwargs": additional}
            )

        updated: dict[str, Any] = {"role": "user", "content": content}
        message_id: Any = getattr(message, "id", None)
        if message_id:
            updated["id"] = message_id
        if additional:
            updated["additional_kwargs"] = additional
        return updated


    @staticmethod
    def sanitize_file_message(
        message: Any,
        *,
        stash_upload: bool = False,
    ) -> Any | None:
        content: list[Any] = UploadParser.message_content_blocks(message)
        if not content:
            return None

        file_blocks: list[Any] = [
            block
            for block in content
            if (parsed := UploadParser._block_as_dict(block)) is not None and UploadParser._is_file_block(parsed)
        ]
        if not file_blocks:
            return None

        text_parts: list[str] = [
            str(parsed.get("text")).strip()
            for block in content
            if (parsed := UploadParser._block_as_dict(block)) is not None
            and parsed.get("type") == "text"
            and str(parsed.get("text") or "").strip()
        ]
        text_parts.append(PDF_UPLOAD_MARKER)
        sanitized: Any = UploadParser.with_message_content(message, "\n".join(text_parts))

        if not stash_upload:
            return sanitized

        uploads: list[dict[str, Any]] = UploadParser.extract_uploads_from_message(message)
        if not uploads:
            first_file: dict[str, Any] = UploadParser._block_as_dict(file_blocks[0]) or {}
            uploads = [
                {
                    "filename": UploadParser._file_block_filename(first_file),
                    "content_base64": "",
                    "missing_bytes": True,
                }
            ]

        if isinstance(sanitized, dict):
            additional: dict[str, Any] = dict(sanitized.get("additional_kwargs") or {})
            additional["pending_cv_uploads"] = uploads
            additional["pending_cv_upload"] = uploads[0]
            return {**sanitized, "additional_kwargs": additional}

        additional: dict[str, Any] = dict(
            getattr(sanitized, "additional_kwargs", None) or {}
        )
        additional["pending_cv_uploads"] = uploads
        additional["pending_cv_upload"] = uploads[0]
        if hasattr(sanitized, "model_copy"):
            return sanitized.model_copy(update={"additional_kwargs": additional})
        return sanitized

    @staticmethod
    def sanitize_file_messages(messages: list[Any]) -> list[Any]:
        return [
            sanitized
            for message in messages
            if (sanitized := UploadParser.sanitize_file_message(message, stash_upload=False)) is not None
        ]


    @staticmethod
    def pending_uploads_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
        for message in reversed(messages):
            uploads: list[dict[str, Any]] = UploadParser.extract_uploads_from_message(message)
            if uploads:
                return uploads
            additional: Any = (
                message.get("additional_kwargs")
                if isinstance(message, dict)
                else getattr(message, "additional_kwargs", None)
            )
            if not isinstance(additional, dict):
                continue
            stashed_list: Any = additional.get("pending_cv_uploads")
            if isinstance(stashed_list, list) and stashed_list:
                return [item for item in stashed_list if isinstance(item, dict)]
            stashed: Any = additional.get("pending_cv_upload")
            if isinstance(stashed, dict):
                return [stashed]
        return []


    @staticmethod
    def pending_upload_from_messages(messages: list[Any]) -> dict[str, Any] | None:
        uploads: list[dict[str, Any]] = UploadParser.pending_uploads_from_messages(messages)
        return uploads[0] if uploads else None


    @staticmethod
    def clear_stashed_uploads(messages: list[Any]) -> list[Any]:
        cleared: list[Any] = []
        for message in messages:
            additional: Any = (
                message.get("additional_kwargs")
                if isinstance(message, dict)
                else getattr(message, "additional_kwargs", None)
            )
            if not isinstance(additional, dict):
                continue
            if (
                "pending_cv_upload" not in additional
                and "pending_cv_uploads" not in additional
            ):
                continue
            remaining: dict[str, Any] = {
                key: value
                for key, value in additional.items()
                if key not in {"pending_cv_upload", "pending_cv_uploads"}
            }
            if isinstance(message, dict):
                updated: dict[str, Any] = {**message, "additional_kwargs": remaining}
                cleared.append(updated)
                continue
            if hasattr(message, "model_copy"):
                cleared.append(message.model_copy(update={"additional_kwargs": remaining}))
                continue
            updated: dict[str, Any] = {
                "role": "user",
                "content": UploadParser.message_content_blocks(message)
                or getattr(message, "content", ""),
                "additional_kwargs": remaining,
            }
            message_id: Any = getattr(message, "id", None)
            if message_id:
                updated["id"] = message_id
            cleared.append(updated)
        return cleared
