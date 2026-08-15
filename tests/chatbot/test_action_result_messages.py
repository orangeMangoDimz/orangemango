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
                        "decision": "strong_fit",
                    },
                }
            ]
        }
    }

    payload = chatbot_graph.slim_action_result("match_jobs", update, {})

    assert payload is not None
    assert payload["ok"] is True
    assert payload["match_count"] == 1
    assert payload["matches"][0]["cv_filename"] == "a.pdf"
    assert payload["matches"][0]["normalized_score"] == 88
