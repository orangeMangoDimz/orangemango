"""Decode and compact raw MCP scrape responses into job cards."""

from __future__ import annotations

import json
from typing import Any

from app.config.const.chatbot import MAX_REQUIREMENT_CHARS, MAX_REQUIREMENTS
from app.config.const.chatbot_errors import (
    JOB_CARD_UNKNOWN_COMPANY,
    JOB_CARD_UNTITLED,
)
from app.services.chatbot.job_key_utils import JobKeyUtils
from app.services.chatbot.text_utils import TextUtils


class ScrapeResponseParser:
    """Normalize heterogeneous MCP payloads into a stable card envelope."""

    @staticmethod
    def decode_mcp_result(raw: Any) -> Any:
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"text": TextUtils.short_text(raw)}

        if hasattr(raw, "artifact") and raw.artifact is not None:
            return ScrapeResponseParser.decode_mcp_result(raw.artifact)

        if hasattr(raw, "content") and not isinstance(raw, (dict, list)):
            return ScrapeResponseParser.decode_mcp_result(raw.content)

        if isinstance(raw, list):
            text_parts: list[str] = [
                item["text"]
                for item in raw
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            if text_parts and len(text_parts) == len(raw):
                return ScrapeResponseParser.decode_mcp_result("\n".join(text_parts))
            return [ScrapeResponseParser.decode_mcp_result(item) for item in raw]

        if isinstance(raw, dict):
            for key in (
                "structuredContent",
                "structured_content",
                "artifact",
                "result",
            ):
                if key in raw and raw[key] is not None:
                    return ScrapeResponseParser.decode_mcp_result(raw[key])
            if isinstance(raw.get("text"), str) and len(raw) <= 2:
                return ScrapeResponseParser.decode_mcp_result(raw["text"])

        return raw

    @staticmethod
    def extract_job_payloads(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            result: list[dict[str, Any]] = []
            for item in payload:
                result.extend(ScrapeResponseParser.extract_job_payloads(item))
            return result

        if not isinstance(payload, dict):
            return []

        if isinstance(payload.get("job"), dict):
            return [payload["job"]]

        for key in ("jobs", "results", "items", "sites"):
            value: Any = payload.get(key)
            if isinstance(value, list):
                return ScrapeResponseParser.extract_job_payloads(value)
            if isinstance(value, dict):
                nested: list[dict[str, Any]] = (
                    ScrapeResponseParser.extract_job_payloads(value)
                )
                if nested:
                    return nested

        if any(key in payload for key in ("title", "job_title", "url", "job_url")):
            return [payload]

        return []

    @staticmethod
    def _is_requirement_noise(line: str) -> bool:
        normalized: str = line.strip().casefold()
        if not normalized:
            return True
        if normalized == "...":
            return True
        if normalized.startswith("about "):
            return True
        return normalized in {
            "requirements",
            "job requirements",
            "qualification",
            "qualifications",
        }

    @staticmethod
    def compact_job_card(
        job: dict[str, Any], envelope: dict[str, Any]
    ) -> dict[str, Any]:
        def first(*keys: str) -> Any:
            for key in keys:
                if job.get(key) not in (None, "", []):
                    return job[key]
            return None

        requirements: Any = first("requirements", "required_skills", "qualifications")
        if isinstance(requirements, list):
            requirements = [
                TextUtils.short_text(
                    item.get("name") if isinstance(item, dict) else item,
                    MAX_REQUIREMENT_CHARS,
                )
                for item in requirements[:MAX_REQUIREMENTS]
            ]
            requirements = [item for item in requirements if item]
        elif isinstance(requirements, str):
            requirement_lines: list[str] = [
                TextUtils.short_text(line, MAX_REQUIREMENT_CHARS)
                for line in requirements.splitlines()
                if str(line).strip()
            ]
            requirements = [
                line
                for line in requirement_lines[:MAX_REQUIREMENTS]
                if not ScrapeResponseParser._is_requirement_noise(line)
            ]
        else:
            requirements = []

        description = TextUtils.short_text(first("description", "summary", "content"))
        if not description and requirements:
            description = TextUtils.short_text(requirements[0], MAX_REQUIREMENT_CHARS)

        return {
            "title": TextUtils.short_text(first("title", "raw_title", "job_title"))
            or JOB_CARD_UNTITLED,
            "company": TextUtils.short_text(first("company", "company_name"))
            or JOB_CARD_UNKNOWN_COMPANY,
            "location": TextUtils.short_text(first("location", "locations")),
            "url": TextUtils.short_text(first("url", "job_url", "link"), 1000),
            "salary": TextUtils.short_text(first("salary", "salary_range")),
            "posted_date": TextUtils.short_text(
                first("posted_date", "date_posted"), 160
            ),
            "posted_at": TextUtils.short_text(first("posted_at"), 160),
            "work_type": TextUtils.short_text(first("work_type", "remote_type"), 80),
            "employment_type": TextUtils.short_text(
                first("employment_type", "job_type"), 80
            ),
            "experience_level": TextUtils.short_text(
                first("experience_level", "seniority"), 80
            ),
            "description": description,
            "requirements": requirements,
            "site": TextUtils.short_text(first("site") or envelope.get("site"), 80),
            "scrape_errors": TextUtils.short_list(envelope.get("errors"), 2, 200),
        }

    @staticmethod
    def compact_scrape_response(raw: Any) -> dict[str, Any]:
        decoded: Any = ScrapeResponseParser.decode_mcp_result(raw)
        envelope: dict[str, Any] = decoded if isinstance(decoded, dict) else {}
        cards: list[dict[str, Any]] = []
        seen: set[str] = set()

        for index, job in enumerate(ScrapeResponseParser.extract_job_payloads(decoded)):
            card: dict[str, Any] = ScrapeResponseParser.compact_job_card(job, envelope)
            key: str = JobKeyUtils.job_selection_key({"job_card": card}, index)
            if key in seen:
                continue
            seen.add(key)
            cards.append(card)

        return {
            "total": len(cards),
            "truncated": bool(
                envelope.get("truncated")
                or envelope.get("is_truncated")
                or envelope.get("has_more")
            ),
            "cards": cards,
            "errors": TextUtils.short_list(envelope.get("errors"), 5, 300),
        }

    @staticmethod
    def filter_scrape_args(tool: Any, request: dict[str, Any]) -> dict[str, Any]:
        args: dict[str, Any] = {
            key: value
            for key, value in request.items()
            if value not in (None, "", [], {})
        }
        schema: Any = getattr(tool, "args_schema", None)
        allowed: set[str]
        if hasattr(schema, "model_fields"):
            allowed = set(schema.model_fields)
        elif isinstance(schema, dict):
            allowed = set((schema.get("properties") or {}).keys())
        else:
            return args
        return {key: value for key, value in args.items() if key in allowed}
