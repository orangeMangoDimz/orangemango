from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from studio.chatbot import graph as chatbot_graph


def test_build_action_result_messages_pair() -> None:
    messages = chatbot_graph.build_action_result_messages(
        "review_cv",
        {"ok": True, "action": "review_cv", "review": {"status": "complete"}},
    )

    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert messages[0].tool_calls
    assert messages[0].tool_calls[0]["name"] == "review_cv"
    assert isinstance(messages[1], ToolMessage)
    assert messages[1].tool_call_id == messages[0].tool_calls[0]["id"]
    payload = json.loads(str(messages[1].content))
    assert payload["ok"] is True
    assert payload["action"] == "review_cv"


def test_record_completed_action_emits_review_result_messages() -> None:
    state: dict[str, Any] = {
        "router": {"completed_actions": []},
        "cv": {},
        "jobs": {},
        "errors": [],
    }
    update = {
        "cv": {
            "review": {
                "status": "complete",
                "mode": "general",
                "focus": None,
                "target_role": None,
                "overall_score": None,
                "feedback": [
                    {
                        "title": "Summary",
                        "observation": "Clear",
                        "recommendation": "Keep metrics",
                    }
                ],
            }
        }
    }

    result = chatbot_graph.record_completed_action(
        state,
        "review_cv",
        update,
        emit_result=True,
    )

    assert result["router"]["completed_actions"] == ["review_cv"]
    messages = result["messages"]
    assert len(messages) == 2
    assert isinstance(messages[0], AIMessage)
    assert isinstance(messages[1], ToolMessage)
    assert messages[1].name == "review_cv"
    payload = json.loads(str(messages[1].content))
    assert payload["review"]["feedback"][0]["title"] == "Summary"


def test_record_completed_action_skips_extract_cv_result_messages() -> None:
    state: dict[str, Any] = {"router": {"completed_actions": []}, "errors": []}
    result = chatbot_graph.record_completed_action(
        state,
        "extract_cv",
        {"cv": {"documents": []}},
        emit_result=True,
    )
    assert "messages" not in result


def test_bounded_conversation_preserves_tool_pair() -> None:
    tool_call_id = "review_cv:test"
    state: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "How about this guy?"},
            {
                "role": "assistant",
                "content": "Earlier Dimas vs Sinan comparison...",
            },
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": "review_cv",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=json.dumps(
                    {
                        "ok": True,
                        "action": "review_cv",
                        "review": {"status": "complete", "feedback": []},
                    }
                ),
                tool_call_id=tool_call_id,
                name="review_cv",
            ),
        ]
    }

    projected = chatbot_graph.bounded_conversation(state)

    assert projected[0]["role"] == "user"
    assert projected[1]["role"] == "assistant"
    assert isinstance(projected[2], AIMessage)
    assert projected[2].tool_calls
    assert isinstance(projected[3], ToolMessage)
    assert projected[3].name == "review_cv"
    assert "review_cv" in str(projected[3].content)


def test_slim_action_result_for_match_jobs() -> None:
    update = {
        "jobs": {
            "matches": [
                {
                    "cv_id": "1",
                    "cv_filename": "a.pdf",
                    "job_key": "url:https://example.com/a",
                    "job_card": {
                        "title": "Backend",
                        "company": "Acme",
                        "url": "https://example.com/a",
                    },
                    "score": {
                        "normalized_score": 88,
                        "score_coverage": 0.8,
                        "decision": "ready",
                        "fit_verdict": "yes",
                        "verdict_reason": "Confirmed match.",
                        "review_reasons": [],
                    },
                }
            ]
        }
    }

    payload = chatbot_graph.slim_action_result("match_jobs", update, {})

    assert payload is not None
    assert payload["ok"] is True
    assert payload["match_count"] == 1
    assert payload["match_assessment"]["verdict"] == "yes"
    assert payload["match_assessment"]["yes_count"] == 1
    assert payload["matches"][0]["cv_filename"] == "a.pdf"
    assert payload["matches"][0]["score"] == 88
    assert payload["matches"][0]["fit_verdict"] == "yes"
    assert "provisional_score" not in payload["matches"][0]


def test_build_match_assessment_aggregates_some_and_uncertain() -> None:
    matches = [
        {
            "cv_id": "1",
            "cv_filename": "a.pdf",
            "job_key": "url:https://example.com/a",
            "job_card": {"title": "Backend", "company": "Acme"},
            "score": {
                "normalized_score": 90,
                "score_coverage": 0.8,
                "decision": "ready",
                "fit_verdict": "yes",
                "verdict_reason": "Confirmed match.",
                "review_reasons": [],
            },
        },
        {
            "cv_id": "1",
            "cv_filename": "a.pdf",
            "job_key": "url:https://example.com/b",
            "job_card": {"title": "QA", "company": "Beta"},
            "score": {
                "normalized_score": 40,
                "score_coverage": 0.75,
                "decision": "ready",
                "fit_verdict": "no",
                "verdict_reason": "Confirmed non-match.",
                "review_reasons": [],
            },
        },
    ]
    assessment = chatbot_graph.build_match_assessment(matches)
    assert assessment["verdict"] == "some"
    assert assessment["yes_count"] == 1
    assert assessment["no_count"] == 1
    assert assessment["uncertain_count"] == 0


def test_build_match_assessment_sparse_scores_are_uncertain() -> None:
    matches = [
        {
            "cv_id": "1",
            "cv_filename": "a.pdf",
            "job_key": "url:https://example.com/cermati",
            "job_card": {
                "title": "Software Engineer",
                "company": "cermati.com",
                "url": "https://example.com/cermati",
            },
            "score": {
                "normalized_score": 100.0,
                "score_coverage": 0.3,
                "decision": "needs_review",
                "fit_verdict": "uncertain",
                "verdict_reason": "Insufficient evidence.",
                "review_reasons": ["No requirements or responsibilities provided."],
            },
        },
        {
            "cv_id": "1",
            "cv_filename": "a.pdf",
            "job_key": "url:https://example.com/kitacakap",
            "job_card": {
                "title": "Software Engineer",
                "company": "KitaCakap",
                "url": "https://example.com/kitacakap",
            },
            "score": {
                "normalized_score": 86.36,
                "score_coverage": 0.55,
                "decision": "needs_review",
                "fit_verdict": "uncertain",
                "verdict_reason": "Insufficient evidence.",
                "review_reasons": ["required_skills: unknown"],
            },
        },
    ]

    assessment = chatbot_graph.build_match_assessment(matches)
    payload = chatbot_graph.slim_match_result({"matches": matches})

    assert assessment["verdict"] == "uncertain"
    assert assessment["uncertain_count"] == 2
    assert assessment["yes_count"] == 0
    assert payload["matches"][0]["fit_verdict"] == "uncertain"
    assert payload["matches"][0]["provisional_score"] == 100.0
    assert "score" not in payload["matches"][0]
    assert payload["matches"][0]["score_coverage"] == 0.3
    assert payload["matches"][0]["review_reasons"]
    assert payload["match_assessment"]["verdict"] == "uncertain"


def test_record_completed_action_persists_successful_match_snapshot() -> None:
    matches = [
        {
            "cv_id": "1",
            "cv_filename": "a.pdf",
            "job_key": "url:https://example.com/a",
            "job_card": {"title": "Backend", "company": "Acme"},
            "score": {
                "normalized_score": 88,
                "score_coverage": 0.8,
                "decision": "ready",
                "fit_verdict": "yes",
            },
        }
    ]
    job = {
        "job_card": {
            "title": "Backend",
            "company": "Acme",
            "url": "https://example.com/a",
        },
        "validation_status": "valid",
        "matching_features": {"content_hash": "abc"},
    }
    state: dict[str, Any] = {
        "router": {"completed_actions": []},
        "cv": {
            "documents": [
                {
                    "id": "1",
                    "filename": "a.pdf",
                    "cv_text": "text",
                    "cv_result": {"validation_status": "valid"},
                    "cv_features": {"role_tags": ["Backend Engineer"]},
                }
            ]
        },
        "selection": {"job_source": "existing", "score_requested": True},
        "jobs": {"results": [job], "matches": []},
        "errors": [],
    }

    result = chatbot_graph.record_completed_action(
        state,
        "match_jobs",
        {"jobs": {"matches": matches}},
        emit_result=True,
    )

    stored = result["action_results"]["match_jobs"]
    assert stored["fingerprint"] == chatbot_graph.action_fingerprint(
        "match_jobs",
        {**state, "jobs": {**state["jobs"], "matches": matches}},
    )
    assert stored["snapshot"]["jobs"]["matches"][0]["cv_id"] == "1"


def test_record_completed_action_skips_invalid_extract_job() -> None:
    state: dict[str, Any] = {
        "router": {"completed_actions": []},
        "selection": {"job_input_text": "not a job"},
        "jobs": {"results": []},
        "errors": [],
    }
    result = chatbot_graph.record_completed_action(
        state,
        "extract_job",
        {
            "jobs": {
                "results": [
                    {
                        "job_card": {"title": "Untitled job"},
                        "validation_status": "invalid",
                    }
                ]
            }
        },
        emit_result=True,
    )
    assert "extract_job" not in (result.get("action_results") or {})
