"""Project verbose action results into slim payloads for the language model."""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from app.config.const.chatbot import (
    MAX_ACTION_RESULT_CHARS,
    ROUTE_COMPARE_CVS,
    ROUTE_EXTRACT_JOB,
    ROUTE_MATCH_JOBS,
    ROUTE_REVIEW_CV,
    ROUTE_SEARCH_JOBS,
)
from app.config.const.chatbot_errors import (
    ERROR_CV_COMPARISON_FAILED,
    ERROR_CV_REVIEW_FAILED,
    JOB_CARD_PASTED_TITLE,
    STATUS_UNAVAILABLE,
)
from app.models.chatbot.literals import AgentAction
from app.models.chatbot.state import ConversationState
from app.repositories.chatbot.conversation_state_repository import (
    ConversationStateRepository,
)
from app.repositories.chatbot.job_state_repository import JobStateRepository
from app.repositories.chatbot.subgraph_repository import SubgraphRepository
from app.services.chatbot.job_key_utils import JobKeyUtils
from app.services.chatbot.text_utils import TextUtils


class ResultProjectionService:
    """Slim, model-facing views of CV, job, search, and match results."""

    def __init__(
        self,
        *,
        state: ConversationStateRepository,
        jobs: JobStateRepository,
        subgraphs: SubgraphRepository,
    ) -> None:
        self._state = state
        self._jobs = jobs
        self._subgraphs = subgraphs

    def slim_review_result(
        self, review: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not isinstance(review, dict):
            return None
        feedback: list[dict[str, Any]] = []
        for item in review.get("feedback") or []:
            if not isinstance(item, dict):
                continue
            feedback.append(
                {
                    "title": item.get("title"),
                    "observation": item.get("observation"),
                    "recommendation": item.get("recommendation"),
                }
            )
        return {
            "status": review.get("status"),
            "mode": review.get("mode"),
            "focus": review.get("focus"),
            "target_role": review.get("target_role"),
            "overall_score": review.get("overall_score"),
            "feedback": feedback,
        }

    def slim_comparison_result(
        self, comparison: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not isinstance(comparison, dict):
            return None
        candidates: list[dict[str, Any]] = []
        for item in comparison.get("candidates") or []:
            if not isinstance(item, dict):
                continue
            candidates.append(
                {
                    "filename": item.get("filename"),
                    "strengths": item.get("strengths") or [],
                    "weaknesses": item.get("weaknesses") or item.get("gaps") or [],
                    "summary": item.get("summary"),
                }
            )
        return {
            "overview": comparison.get("overview"),
            "candidates": candidates,
            "recommendation": comparison.get("recommendation"),
        }

    def slim_job_card(
        self,
        card: dict[str, Any] | None,
        *,
        include_description: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(card, dict):
            return {}
        fields: tuple[str, ...] = (
            "company",
            "title",
            "location",
            "posted_date",
            "salary",
            "url",
        )
        if include_description:
            fields = (*fields, "description")
        return {
            key: card.get(key)
            for key in fields
            if card.get(key) is not None and card.get(key) != "" and card.get(key) != []
        }

    def slim_search_result(
        self, jobs_update: dict[str, Any], state: ConversationState
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = [
            item
            for item in (
                jobs_update.get("results")
                or self._state.jobs_bucket(state).get("results")
                or []
            )
            if isinstance(item, dict)
        ]
        active_keys: Any = jobs_update.get("active_job_keys")
        if not isinstance(active_keys, list) or not active_keys:
            active_keys = self._state.jobs_bucket(state).get("active_job_keys")
        if isinstance(active_keys, list) and active_keys:
            wanted: set[str] = {
                str(key).strip() for key in active_keys if str(key).strip()
            }
            results = [
                item
                for index, item in enumerate(results)
                if JobKeyUtils.job_selection_key(item, index) in wanted
            ]
        jobs: list[dict[str, Any]] = []
        for item in results:
            raw_card: Any = (
                item.get("job_card") if isinstance(item.get("job_card"), dict) else None
            )
            card: dict[str, Any] = {
                field: raw_card.get(field)
                for field in ("title", "location", "posted_date", "salary", "url")
                if isinstance(raw_card, dict)
                and raw_card.get(field) not in (None, "", [])
            }
            if card:
                jobs.append(card)
        return {
            "job_count": len(jobs),
            "scrape_total": jobs_update.get(
                "scrape_total", self._state.jobs_bucket(state).get("scrape_total", 0)
            ),
            "scrape_truncated": bool(
                jobs_update.get(
                    "scrape_truncated",
                    self._state.jobs_bucket(state).get("scrape_truncated", False),
                )
            ),
            "jobs": jobs,
        }

    def slim_match_result(
        self,
        jobs_update: dict[str, Any],
        *,
        show_score: bool = True,
    ) -> dict[str, Any]:
        assessment: dict[str, Any] = self.build_match_assessment(
            [
                item
                for item in (jobs_update.get("matches") or [])
                if isinstance(item, dict)
            ],
            show_score=show_score,
        )
        return {
            "match_count": assessment["match_count"],
            "match_assessment": assessment,
            "matches": assessment["matches"],
        }

    def project_match_item(
        self,
        item: dict[str, Any],
        *,
        show_score: bool = True,
    ) -> dict[str, Any] | None:
        score: Any = item.get("score")
        if not isinstance(score, dict):
            score = {}
        fit_verdict: str | None = score.get("fit_verdict")
        verdict_reason_code: str | None = score.get("verdict_reason_code")
        if fit_verdict not in {"yes", "no", "uncertain", "unknown"}:
            fit_verdict, verdict_reason_code = self._subgraphs.classify_fit_verdict(
                normalized_score=score.get("normalized_score"),
                score_coverage=score.get("score_coverage"),
                decision=score.get("decision"),
            )
        card: dict[str, Any] = self.slim_job_card(
            item.get("job_card") if isinstance(item.get("job_card"), dict) else None
        )
        projected: dict[str, Any] = {
            "cv_id": item.get("cv_id"),
            "cv_filename": item.get("cv_filename"),
            "job_key": item.get("job_key"),
            "job": card,
            "title": card.get("title") or item.get("title"),
            "company": card.get("company") or item.get("company"),
            "url": card.get("url") or item.get("url"),
            "fit_verdict": fit_verdict,
            "verdict_reason_code": verdict_reason_code,
            "review_reason_codes": list(score.get("review_reason_codes") or [])[:5],
        }
        if not show_score:
            return projected
        projected["decision"] = score.get("decision")
        projected["score_coverage"] = score.get("score_coverage")
        normalized: Any = score.get("normalized_score")
        if fit_verdict == "uncertain":
            if normalized is not None:
                projected["provisional_score"] = normalized
        elif normalized is not None:
            projected["score"] = normalized
        return projected

    def build_match_assessment(
        self,
        matches: list[dict[str, Any]],
        *,
        show_score: bool = True,
    ) -> dict[str, Any]:
        projected: list[dict[str, Any]] = []
        yes_count = 0
        no_count = 0
        uncertain_count = 0
        for item in matches:
            if not isinstance(item, dict):
                continue
            row: dict[str, Any] | None = self.project_match_item(
                item, show_score=show_score
            )
            if row is None:
                continue
            projected.append(row)
            verdict: str = str(row.get("fit_verdict") or "uncertain")
            if verdict == "yes":
                yes_count += 1
            elif verdict == "no":
                no_count += 1
            else:
                uncertain_count += 1

        aggregate: str
        if not projected:
            aggregate = "uncertain"
        elif uncertain_count == len(projected):
            aggregate = "uncertain"
        elif yes_count == len(projected):
            aggregate = "yes"
        elif no_count == len(projected):
            aggregate = "no"
        elif yes_count > 0:
            aggregate = "some"
        else:
            aggregate = "uncertain"

        return {
            "verdict": aggregate,
            "match_count": len(projected),
            "yes_count": yes_count,
            "no_count": no_count,
            "uncertain_count": uncertain_count,
            "matches": projected,
        }

    def slim_extract_job_result(
        self,
        jobs_update: dict[str, Any],
        state: ConversationState,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = [
            item
            for item in (jobs_update.get("results") or [])
            if isinstance(item, dict)
        ]
        if not results:
            results = [
                item
                for item in (self._state.jobs_bucket(state).get("results") or [])
                if isinstance(item, dict)
            ]
        latest: dict[str, Any] = results[-1] if results else {}
        return {
            "job_count": len(results),
            "validation_status": latest.get("validation_status"),
            "job": self.slim_job_card(
                latest.get("job_card")
                if isinstance(latest.get("job_card"), dict)
                else None
            ),
        }

    def slim_action_result(
        self,
        action: AgentAction,
        update: dict[str, Any],
        state: ConversationState,
    ) -> dict[str, Any] | None:
        errors: list[str] = [
            str(item)
            for item in (update.get("errors") or state.get("errors") or [])[-3:]
            if item
        ]
        if action == ROUTE_REVIEW_CV:
            review: Any = (update.get("cv") or {}).get("review")
            slim: dict[str, Any] | None = self.slim_review_result(
                review if isinstance(review, dict) else None
            )
            if slim is None:
                return {
                    "ok": False,
                    "action": action,
                    "errors": errors or [ERROR_CV_REVIEW_FAILED],
                }
            return {
                "ok": slim.get("status") != STATUS_UNAVAILABLE,
                "action": action,
                "review": slim,
                "errors": errors,
            }
        if action == ROUTE_COMPARE_CVS:
            comparison: Any = (update.get("cv") or {}).get("comparison")
            slim_comparison: dict[str, Any] | None = self.slim_comparison_result(
                comparison if isinstance(comparison, dict) else None
            )
            if slim_comparison is None:
                return {
                    "ok": False,
                    "action": action,
                    "errors": errors or [ERROR_CV_COMPARISON_FAILED],
                }
            return {
                "ok": True,
                "action": action,
                "comparison": slim_comparison,
                "errors": errors,
            }
        if action == ROUTE_SEARCH_JOBS:
            jobs_update: dict[str, Any] = (
                dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
            )
            return {
                "ok": True,
                "action": action,
                **self.slim_search_result(jobs_update, state),
                "errors": errors,
            }
        if action == ROUTE_MATCH_JOBS:
            jobs_update = (
                dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
            )
            return {
                "ok": bool(jobs_update.get("matches")),
                "action": action,
                **self.slim_match_result(
                    jobs_update,
                    show_score=(
                        bool(self._state.selection_bucket(state).get("show_score"))
                        if "show_score" in self._state.selection_bucket(state)
                        else True
                    ),
                ),
                "errors": errors,
            }
        if action == ROUTE_EXTRACT_JOB:
            jobs_update = (
                dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
            )
            payload: dict[str, Any] = self.slim_extract_job_result(jobs_update, state)
            return {
                "ok": payload.get("validation_status") == "valid",
                "action": action,
                **payload,
                "errors": errors,
            }
        return None

    def build_action_result_messages(
        self,
        action: AgentAction,
        payload: dict[str, Any],
    ) -> list[AnyMessage]:
        tool_call_id: str = f"{action}:{uuid.uuid4()}"
        content: str = TextUtils.short_text(
            json.dumps(payload, ensure_ascii=False),
            MAX_ACTION_RESULT_CHARS,
        )
        return [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": action,
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name=action,
            ),
        ]

    def compact_cv_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "matching_features": result.get("matching_features"),
            "validation_status": result.get("validation_status"),
            "validation_errors": TextUtils.short_list(
                result.get("validation_errors"), 5, 300
            ),
            "warnings": TextUtils.short_list(result.get("warnings"), 8, 300),
            "confirmation_required": TextUtils.short_list(
                result.get("confirmation_required"),
                8,
                120,
            ),
        }

    def compact_job_result(
        self, card: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        extract: dict[str, Any] = result.get("extract") or {}
        enriched_card: dict[str, Any] = dict(card)
        if not enriched_card.get("title"):
            enriched_card["title"] = (
                extract.get("normalized_title")
                or extract.get("raw_title")
                or JOB_CARD_PASTED_TITLE
            )
        if not enriched_card.get("company"):
            enriched_card["company"] = extract.get("company") or ""
        if not enriched_card.get("url"):
            enriched_card["url"] = extract.get("job_url") or ""
        if not enriched_card.get("site"):
            enriched_card["site"] = extract.get("source") or ""
        return {
            "job_card": enriched_card,
            "matching_features": result.get("matching_features"),
            "validation_status": result.get("validation_status"),
            "validation_errors": TextUtils.short_list(
                result.get("validation_errors"), 5, 300
            ),
            "warnings": TextUtils.short_list(result.get("warnings"), 8, 300),
        }
