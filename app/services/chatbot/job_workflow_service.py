"""Run the job search, pasted-job extraction, and matching workflows."""

from __future__ import annotations

import asyncio
from typing import Any

from app.config.const.chatbot_errors import (
    ERROR_CV_FEATURES_REQUIRED_FOR_MATCHING,
    ERROR_JOB_EXTRACTION_FAILED,
    ERROR_JOB_SCRAPING_FAILED,
    ERROR_JOB_TARGETS_MISSING_FOR_MATCHING,
    ERROR_MATCHING_FAILED,
    ERROR_PASTED_JOB_DESCRIPTION_MISSING,
    ERROR_PASTED_JOB_EXTRACTION_FAILED,
    JOB_CARD_PASTED_SITE,
)
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.cv_state_repository import CvStateRepository
from app.repositories.chatbot.job_scraper_repository import JobScraperRepository
from app.repositories.chatbot.job_state_repository import JobStateRepository
from app.repositories.chatbot.subgraph_repository import SubgraphRepository
from app.services.chatbot.job_key_utils import JobKeyUtils
from app.services.chatbot.result_projection_service import ResultProjectionService
from app.services.chatbot.scrape_parser import ScrapeResponseParser


class JobWorkflowService:
    """Job scraping, extraction, and CV-to-job scoring."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        cvs: CvStateRepository,
        jobs: JobStateRepository,
        scraper: JobScraperRepository,
        parser: ScrapeResponseParser,
        subgraphs: SubgraphRepository,
        projection: ResultProjectionService,
    ) -> None:
        self._state = state
        self._cvs = cvs
        self._jobs = jobs
        self._scraper = scraper
        self._parser = parser
        self._subgraphs = subgraphs
        self._projection = projection

    def scrape_payload_from_card(
        self,
        card: dict[str, Any],
        request: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "keyword": ", ".join((request or {}).get("keywords") or []),
            "site": card.get("site"),
            "max_age_hours": (request or {}).get("max_age_hours"),
            "job": card,
            "errors": [],
        }

    async def scrape_jobs_with_mcp(self, state: ConversationState) -> dict[str, Any]:
        existing_results: list[dict[str, Any]] = [
            item
            for item in (self._state.jobs_bucket(state).get("results") or [])
            if isinstance(item, dict)
        ]
        try:
            tool: Any = await self._scraper.scrape_jobs_tool()
            request: dict[str, Any] = dict(
                self._state.jobs_bucket(state).get("scrape_request") or {}
            )
            raw: Any = await tool.ainvoke(
                self._parser.filter_scrape_args(tool, request)
            )
            compact: dict[str, Any] = self._parser.compact_scrape_response(raw)
            job_results: list[dict[str, Any]]
            extraction_errors: list[str]
            job_results, extraction_errors = await self.extract_job_cards(
                compact["cards"], request
            )
            return {
                "jobs": {
                    "scrape_total": compact["total"],
                    "scrape_truncated": compact["truncated"],
                    "results": JobKeyUtils.merge_job_results(
                        existing_results, job_results
                    ),
                    "active_job_keys": [
                        JobKeyUtils.job_selection_key(item, index)
                        for index, item in enumerate(job_results)
                    ],
                    "matches": [],
                },
                "errors": self._state.state_errors(
                    state,
                    compact["errors"] + extraction_errors,
                ),
            }
        except Exception as exc:
            return {
                "jobs": {
                    "scrape_total": 0,
                    "scrape_truncated": False,
                    "active_job_keys": [],
                    "matches": [],
                },
                "errors": self._state.state_errors(
                    state,
                    [f"{ERROR_JOB_SCRAPING_FAILED}{type(exc).__name__}"],
                ),
            }

    async def run_one_job_agent(
        self,
        card: dict[str, Any],
        request: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            result: dict[str, Any] = await self._subgraphs.job_extraction.ainvoke(
                {"scraped_job": self.scrape_payload_from_card(card, request)}
            )
            return self._projection.compact_job_result(card, result)
        except Exception as exc:
            return {
                "job_card": card,
                "matching_features": None,
                "validation_status": "invalid",
                "validation_errors": [f"{type(exc).__name__}: {exc}"],
                "warnings": [],
            }

    def pasted_job_card(self, text: str) -> dict[str, Any]:
        return {
            "title": "",
            "company": "",
            "location": "",
            "url": "",
            "salary": "",
            "posted_date": "",
            "posted_at": "",
            "work_type": "",
            "employment_type": "",
            "experience_level": "",
            "description": text,
            "requirements": [],
            "site": JOB_CARD_PASTED_SITE,
            "scrape_errors": [],
        }

    async def extract_pasted_job(self, state: ConversationState) -> dict[str, Any]:
        text: str = (
            self._state.selection_bucket(state).get("job_input_text") or ""
        ).strip()
        existing_results: list[dict[str, Any]] = [
            item
            for item in (self._state.jobs_bucket(state).get("results") or [])
            if isinstance(item, dict)
        ]
        if not text:
            return {
                "jobs": {
                    "matches": [],
                },
                "errors": self._state.state_errors(
                    state,
                    [ERROR_PASTED_JOB_DESCRIPTION_MISSING],
                ),
            }

        card: dict[str, Any] = self.pasted_job_card(text)
        result: dict[str, Any] = await self.run_one_job_agent(card, None)
        errors: list[str] = []
        if result.get("validation_status") != "valid":
            errors.append(ERROR_PASTED_JOB_EXTRACTION_FAILED)

        return {
            "jobs": {
                "scrape_total": self._state.jobs_bucket(state).get("scrape_total", 0),
                "scrape_truncated": self._state.jobs_bucket(state).get(
                    "scrape_truncated", False
                ),
                "results": JobKeyUtils.merge_job_results(existing_results, [result]),
                "active_job_keys": [JobKeyUtils.job_selection_key(result, 0)],
                "matches": [],
            },
            "errors": self._state.state_errors(state, errors),
        }

    async def extract_job_cards(
        self,
        cards: list[dict[str, Any]],
        request: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        results: list[dict[str, Any]] = list(
            await asyncio.gather(
                *(self.run_one_job_agent(card, request) for card in cards)
            )
        )
        errors: list[str] = [
            f"{ERROR_JOB_EXTRACTION_FAILED}{item['job_card'].get('title', 'job')}"
            for item in results
            if item.get("validation_status") != "valid"
        ]
        return results, errors

    async def calculate_job_matches(self, state: ConversationState) -> dict[str, Any]:
        selected_cvs: list[dict[str, Any]] = self._cvs.resolve_selected_cvs(state)
        usable_cvs: list[dict[str, Any]] = [
            document
            for document in selected_cvs
            if isinstance(document.get("cv_result"), dict)
            and document.get("cv_features")
        ]
        if not usable_cvs:
            return {
                "jobs": {"matches": []},
                "errors": self._state.state_errors(
                    state,
                    [ERROR_CV_FEATURES_REQUIRED_FOR_MATCHING],
                ),
            }

        selected_jobs: list[dict[str, Any]] = self._jobs.resolve_selected_jobs(state)
        if not selected_jobs:
            return {
                "jobs": {"matches": []},
                "errors": self._state.state_errors(
                    state,
                    [ERROR_JOB_TARGETS_MISSING_FOR_MATCHING],
                ),
            }
        job_results: Any = self._state.jobs_bucket(state).get("results") or []
        job_key_by_id: dict[int, str] = {
            id(item): JobKeyUtils.job_selection_key(item, index)
            for index, item in enumerate(job_results)
            if isinstance(item, dict)
        }
        errors: list[str] = []

        matches: list[dict[str, Any]] = []
        for document in usable_cvs:
            cv_result: Any = document.get("cv_result")
            cv_id: str = str(document.get("id") or "")
            cv_filename: str = str(document.get("filename") or "cv.pdf")
            for item in selected_jobs:
                job_title: str = str(item["job_card"].get("title") or "job")
                try:
                    result: dict[
                        str, Any
                    ] = await self._subgraphs.matching_score.ainvoke(
                        {"cv_result": cv_result, "job_result": item}
                    )
                    score: Any = result.get("score")
                    if score is not None:
                        matches.append(
                            {
                                "cv_id": cv_id,
                                "cv_filename": cv_filename,
                                "job_key": job_key_by_id.get(
                                    id(item),
                                    JobKeyUtils.job_selection_key(item, 0),
                                ),
                                "job_card": item["job_card"],
                                "score": score,
                            }
                        )
                except Exception as exc:
                    errors.append(
                        f"{ERROR_MATCHING_FAILED}{cv_filename}:{job_title}:{type(exc).__name__}"
                    )
                    matches.append(
                        {
                            "cv_id": cv_id,
                            "cv_filename": cv_filename,
                            "job_key": job_key_by_id.get(
                                id(item),
                                JobKeyUtils.job_selection_key(item, 0),
                            ),
                            "job_card": item["job_card"],
                            "score": {
                                "fit_verdict": "unknown",
                                "verdict_reason_code": "ASSESSMENT_UNAVAILABLE",
                                "review_reason_codes": [],
                            },
                        }
                    )

        matches.sort(
            key=lambda item: (
                item["score"].get("normalized_score") is not None,
                item["score"].get("normalized_score") or -1,
            ),
            reverse=True,
        )
        return {
            "jobs": {"matches": matches, "pending_match": None},
            "errors": self._state.state_errors(state, errors),
        }
