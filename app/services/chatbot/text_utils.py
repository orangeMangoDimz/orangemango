"""Pure text, hashing, and normalization helpers used across the chatbot graph."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from app.config.const.chatbot import MAX_FIELD_CHARS
from app.services.text_normalization import casefolded_text


class TextUtils:
    """Stateless text helpers shared by repositories and services."""

    @staticmethod
    def short_text(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
        text: str = "" if value is None else str(value).strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    @staticmethod
    def short_list(value: Any, limit: int, item_limit: int) -> list[str]:
        if isinstance(value, str):
            value: list[Any] = [value]
        if not isinstance(value, list):
            return []
        return [
            item
            for item in (
                TextUtils.short_text(item, item_limit) for item in value[:limit]
            )
            if item
        ]

    @staticmethod
    def canonical_json_hash(value: Any) -> str:
        encoded: bytes = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def normalize_fingerprint_text(value: Any) -> str:
        return casefolded_text(value)

    @staticmethod
    def parse_executed_at(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed: datetime = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def normalize_role_constraints(values: Any) -> list[str]:
        seen: set[str] = set()
        constraints: list[str] = []
        for item in values or []:
            text: str = TextUtils.normalize_fingerprint_text(item)
            if text and text not in seen:
                seen.add(text)
                constraints.append(text)
        return constraints

    @staticmethod
    def display_role_constraints(constraints: list[str]) -> list[str]:
        return [item.title() for item in constraints]

    @staticmethod
    def first_contiguous_phrase(text: str, candidates: Any) -> str | None:
        """Return the first candidate as written in text, ignoring case."""
        folded_text: str = text.casefold()
        for candidate in candidates or []:
            phrase: str = str(candidate or "").strip()
            if not phrase:
                continue
            start: int = folded_text.find(phrase.casefold())
            if start >= 0:
                return text[start : start + len(phrase)]
        return None
