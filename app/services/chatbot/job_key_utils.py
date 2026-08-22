"""Identity and versioning helpers for job result entries.

State-free by design: the scrape parser needs these below the repository layer.
"""

from __future__ import annotations

import hashlib
from typing import Any

from app.services.chatbot.text_utils import TextUtils


class JobKeyUtils:
    """Stable keys and content versions for scraped or pasted job results."""

    @staticmethod
    def job_selection_key(item: dict[str, Any], index: int) -> str:
        card: Any = item.get("job_card") if isinstance(item, dict) else None
        if isinstance(card, dict):
            url: str = str(card.get("url") or "").strip()
            if url:
                return "url:" + url.casefold()
            title: str = str(card.get("title") or "").strip().casefold()
            company: str = str(card.get("company") or "").strip().casefold()
            if title or company:
                return f"title:{title}|company:{company}"
            description: str = str(card.get("description") or "").strip()
            if description:
                digest: str = hashlib.sha1(
                    description.encode("utf-8"),
                    usedforsecurity=False,
                ).hexdigest()[:16]
                return f"desc:{digest}"
        return f"idx:{index}"

    @staticmethod
    def merge_job_results(
        existing: list[dict[str, Any]] | None,
        incoming: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        by_key: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for source in (existing or [], incoming or []):
            for index, item in enumerate(source):
                if not isinstance(item, dict):
                    continue
                key: str = JobKeyUtils.job_selection_key(item, index)
                if key not in by_key:
                    order.append(key)
                by_key[key] = item
        return [by_key[key] for key in order]

    @staticmethod
    def job_content_version(item: dict[str, Any]) -> str:
        features: Any = item.get("matching_features")
        if isinstance(features, dict):
            content_hash: Any = features.get("content_hash")
            if content_hash:
                return str(content_hash)
        card: Any = (
            item.get("job_card") if isinstance(item.get("job_card"), dict) else item
        )
        return TextUtils.canonical_json_hash(card or {})
