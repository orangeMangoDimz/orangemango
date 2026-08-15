import asyncio
from typing import Any

import pytest

from studio.chatbot import graph as chatbot_graph


def _cv_doc(
    *,
    doc_id: str,
    filename: str,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "filename": filename,
        "cv_text": f"text for {filename}",
        "cv_result": {"validation_status": "valid"} if features else None,
        "cv_features": features,
        "cv_review": None,
    }


def test_is_vague_cv_feedback_matches_how_about_this_with_upload_marker() -> None:
    text = f"How about this?\n{chatbot_graph.PDF_UPLOAD_MARKER}"
    assert chatbot_graph.is_vague_cv_feedback(text) is True


def test_ingest_appends_new_cv_and_preserves_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = [
        _cv_doc(
            doc_id="1",
            filename="a.pdf",
            features={"role_tags": ["Backend Engineer"]},
        ),
        _cv_doc(
            doc_id="2",
            filename="b.pdf",
            features={"role_tags": ["Intern"]},
        ),
    ]
    state: dict[str, Any] = {
        "messages": [{"role": "user", "content": "How about this?"}],
        "pending_cv_uploads": [{"filename": "c.pdf", "content_base64": "unused"}],
        "cv": {
            "documents": existing,
            "comparison": {
                "overview": "previous comparison",
                "candidates": [],
                "recommendation": "keep a",
            },
        },
    }

    def fake_from_upload(upload: dict[str, Any]) -> dict[str, Any]:
        return _cv_doc(doc_id="3", filename=str(upload["filename"]))

    monkeypatch.setattr(chatbot_graph, "cv_document_from_upload", fake_from_upload)

    result = chatbot_graph.ingest_input(state)
    documents = result["cv"]["documents"]

    assert [doc["id"] for doc in documents] == ["1", "2", "3"]
    assert documents[0]["cv_features"] == {"role_tags": ["Backend Engineer"]}
    assert documents[1]["cv_features"] == {"role_tags": ["Intern"]}
    assert documents[2]["cv_features"] is None
    assert result["cv"]["comparison"] is None
    assert result["cv"]["needs_extraction"] is True


def test_ingest_without_upload_does_not_clear_comparison() -> None:
    state: dict[str, Any] = {
        "messages": [{"role": "user", "content": "thanks"}],
        "cv": {
            "documents": [
                _cv_doc(
                    doc_id="1",
                    filename="a.pdf",
                    features={"role_tags": ["Backend Engineer"]},
                )
            ],
            "comparison": {
                "overview": "keep me",
                "candidates": [],
                "recommendation": "a",
            },
        },
    }

    result = chatbot_graph.ingest_input(state)

    assert "documents" not in result["cv"]
    assert "comparison" not in result["cv"]


def test_route_forces_compare_for_vague_multi_cv_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        _cv_doc(
            doc_id="1",
            filename="a.pdf",
            features={"role_tags": ["Backend Engineer"]},
        ),
        _cv_doc(
            doc_id="2",
            filename="b.pdf",
            features={"role_tags": ["Intern"]},
        ),
        _cv_doc(doc_id="3", filename="c.pdf", features={"role_tags": ["Junior"]}),
    ]
    state: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": f"How about this?\n{chatbot_graph.PDF_UPLOAD_MARKER}",
            }
        ],
        "cv": {"documents": documents},
        "router": {"completed_actions": ["extract_cv"]},
        "selection": {},
        "jobs": {"results": []},
    }

    class FakeDecision:
        route = "review_cv"
        reason = "The user asked for feedback on the uploaded CV."
        job_source = "none"
        score_requested = False
        review_target_role = None
        review_mode = "general"
        review_focus = None
        review_mode_reason = "The user asked for feedback on the uploaded CV."
        needs_cv_text = False
        selected_cv_id = "3"
        selected_job_keys = None
        scrape_request = chatbot_graph.ScrapeRequest()
        needs_cv_features = False
        is_follow_up = False

    class FakeRouter:
        async def ainvoke(self, _messages: list[Any]) -> FakeDecision:
            return FakeDecision()

    class FakeChatModel:
        def structured(self, _schema: Any) -> FakeRouter:
            return FakeRouter()

        @classmethod
        def from_env(cls) -> Any:
            return cls()

    monkeypatch.setattr(chatbot_graph, "ChatModel", FakeChatModel)

    result = asyncio.run(chatbot_graph.route_message(state, FakeChatModel()))

    assert result["router"]["route"] == "compare_cvs"
    assert result["selection"]["selected_cv_id"] is None


def _extracted_docs() -> list[dict[str, Any]]:
    return [
        _cv_doc(
            doc_id="1",
            filename="a.pdf",
            features={"role_tags": ["Backend Engineer"]},
        ),
        _cv_doc(
            doc_id="2",
            filename="b.pdf",
            features={"role_tags": ["Intern"]},
        ),
    ]


def _pending_docs() -> list[dict[str, Any]]:
    return [
        _cv_doc(doc_id="1", filename="a.pdf"),
        _cv_doc(doc_id="2", filename="b.pdf"),
    ]


def _fake_router(
    monkeypatch: pytest.MonkeyPatch,
    *,
    route: str,
    job_source: str = "none",
    score_requested: bool = False,
    needs_cv_features: bool = False,
    selected_cv_id: str | None = None,
    is_follow_up: bool = False,
) -> None:
    class FakeDecision:
        def __init__(self) -> None:
            self.route = route
            self.reason = f"Fake route {route}."
            self.job_source = job_source
            self.score_requested = score_requested
            self.review_target_role = None
            self.review_mode = "general"
            self.review_focus = None
            self.review_mode_reason = None
            self.needs_cv_text = False
            self.needs_cv_features = needs_cv_features
            self.is_follow_up = is_follow_up
            self.selected_cv_id = selected_cv_id
            self.selected_job_keys = None
            self.scrape_request = chatbot_graph.ScrapeRequest()

    class FakeRouter:
        async def ainvoke(self, _messages: list[Any]) -> FakeDecision:
            return FakeDecision()

    class FakeChatModel:
        def structured(self, _schema: Any) -> FakeRouter:
            return FakeRouter()

        @classmethod
        def from_env(cls) -> Any:
            return cls()

    monkeypatch.setattr(chatbot_graph, "ChatModel", FakeChatModel)


def test_route_coerces_extract_cv_to_respond_when_already_extracted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": "Well, what type of jobs that suitable for all of them?",
            }
        ],
        "cv": {"documents": _extracted_docs(), "needs_extraction": False},
        "router": {"completed_actions": []},
        "selection": {},
        "jobs": {"results": []},
    }
    _fake_router(monkeypatch, route="extract_cv", needs_cv_features=True)

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "respond"
    assert result["router"]["needs_cv_features"] is True


def test_route_after_router_skips_extract_when_profiles_ready() -> None:
    state: dict[str, Any] = {
        "cv": {"documents": _extracted_docs(), "needs_extraction": False},
        "router": {
            "route": "respond",
            "needs_cv_features": True,
            "completed_actions": [],
        },
        "selection": {},
        "jobs": {"results": []},
    }

    assert chatbot_graph.route_after_router(state) == "respond"


def test_route_after_router_extracts_then_resumes_respond_for_role_question() -> None:
    pending: dict[str, Any] = {
        "cv": {"documents": _pending_docs(), "needs_extraction": True},
        "router": {
            "route": "respond",
            "needs_cv_features": True,
            "completed_actions": [],
        },
        "selection": {},
        "jobs": {"results": []},
    }
    assert chatbot_graph.route_after_router(pending) == "extract_cv"

    after_extract: dict[str, Any] = {
        "cv": {"documents": _extracted_docs(), "needs_extraction": False},
        "router": {
            "route": "respond",
            "needs_cv_features": True,
            "completed_actions": ["extract_cv"],
        },
        "selection": {},
        "jobs": {"results": []},
    }
    assert chatbot_graph.route_after_cv_subagent(after_extract) == "respond"


def test_route_after_router_search_jobs_skips_pending_extraction() -> None:
    state: dict[str, Any] = {
        "cv": {"documents": _pending_docs(), "needs_extraction": True},
        "router": {"route": "search_jobs", "completed_actions": []},
        "selection": {"job_source": "search", "score_requested": False},
        "jobs": {"results": []},
    }

    assert chatbot_graph.route_after_router(state) == "search_jobs"


def test_route_after_router_match_jobs_with_existing_jobs() -> None:
    state: dict[str, Any] = {
        "cv": {"documents": _extracted_docs(), "needs_extraction": False},
        "router": {"route": "match_jobs", "completed_actions": []},
        "selection": {"job_source": "existing"},
        "jobs": {
            "results": [
                {
                    "job_card": {
                        "title": "Backend",
                        "url": "https://example.com/backend",
                    },
                    "validation_status": "valid",
                    "matching_features": {"role": "Backend"},
                }
            ]
        },
    }

    assert chatbot_graph.route_after_router(state) == "match_jobs"


def test_route_after_router_review_and_compare_still_dispatch() -> None:
    review_state: dict[str, Any] = {
        "cv": {"documents": _extracted_docs(), "needs_extraction": False},
        "router": {"route": "review_cv", "completed_actions": []},
        "selection": {},
        "jobs": {"results": []},
    }
    compare_state: dict[str, Any] = {
        "cv": {"documents": _extracted_docs(), "needs_extraction": False},
        "router": {"route": "compare_cvs", "completed_actions": []},
        "selection": {},
        "jobs": {"results": []},
    }

    assert chatbot_graph.route_after_router(review_state) == "review_cv"
    assert chatbot_graph.route_after_router(compare_state) == "compare_cvs"


def test_search_only_job_workflow_ends_at_respond() -> None:
    after_search: dict[str, Any] = {
        "cv": {"documents": _extracted_docs(), "needs_extraction": False},
        "router": {
            "route": "search_jobs",
            "completed_actions": ["search_jobs"],
        },
        "selection": {"job_source": "search", "score_requested": False},
        "jobs": {
            "results": [
                {
                    "job_card": {
                        "title": "Software Engineer",
                        "url": "https://example.com/se",
                    }
                }
            ]
        },
    }

    assert chatbot_graph.route_after_search_or_extract(after_search) == "end"
    assert chatbot_graph.route_after_job_subagent(after_search) == "respond"


def test_match_with_search_source_chains_search_then_match() -> None:
    before_search: dict[str, Any] = {
        "cv": {"documents": _extracted_docs(), "needs_extraction": False},
        "router": {"route": "match_jobs", "completed_actions": []},
        "selection": {"job_source": "search", "score_requested": True},
        "jobs": {"results": []},
    }
    assert chatbot_graph.route_into_job_subagent(before_search) == "scrape_jobs"

    after_search: dict[str, Any] = {
        "cv": {"documents": _extracted_docs(), "needs_extraction": False},
        "router": {
            "route": "match_jobs",
            "completed_actions": ["search_jobs"],
        },
        "selection": {"job_source": "search", "score_requested": True},
        "jobs": {
            "results": [
                {
                    "job_card": {
                        "title": "Software Engineer",
                        "url": "https://example.com/se",
                    }
                }
            ]
        },
    }
    assert chatbot_graph.route_after_search_or_extract(after_search) == "match_jobs"

    after_match: dict[str, Any] = {
        **after_search,
        "router": {
            "route": "match_jobs",
            "completed_actions": ["search_jobs", "match_jobs"],
        },
    }
    assert chatbot_graph.route_after_job_subagent(after_match) == "respond"


def test_route_message_keeps_plain_search_as_search_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": "Could u pls search for Software engineer jobs right now?",
            }
        ],
        "cv": {"documents": _extracted_docs(), "needs_extraction": False},
        "router": {"completed_actions": []},
        "selection": {},
        "jobs": {"results": []},
    }
    _fake_router(
        monkeypatch,
        route="match_jobs",
        job_source="search",
        score_requested=False,
    )

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "search_jobs"
    assert result["selection"]["job_source"] == "search"
    assert result["selection"]["score_requested"] is False


def test_route_message_coerces_follow_up_to_respond(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "Rate this CV"},
            {
                "role": "assistant",
                "content": "Your CV has an overall score of 75 out of 100.",
            },
            {"role": "user", "content": "I mean 1 - 5"},
        ],
        "cv": {
            "documents": _extracted_docs()[:1],
            "needs_extraction": False,
            "review": {
                "status": "complete",
                "mode": "scored",
                "overall_score": 75.0,
                "feedback": [{"title": "Strong summary"}],
            },
        },
        "router": {"completed_actions": []},
        "selection": {},
        "jobs": {"results": []},
    }
    _fake_router(monkeypatch, route="review_cv", is_follow_up=True)

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "respond"
    assert "follow" in result["router"]["route_reason"].lower()


def test_router_recent_conversation_skips_tool_call_shells() -> None:
    state: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "Rate this CV"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "review_cv", "args": {}}],
            },
            {
                "role": "tool",
                "name": "review_cv",
                "tool_call_id": "1",
                "content": '{"ok": true, "overall_score": 75}',
            },
            {
                "role": "assistant",
                "content": "Your CV scored 75 out of 100.",
            },
            {"role": "user", "content": "I mean 1 - 5"},
        ]
    }

    history = chatbot_graph.router_recent_conversation(state)

    assert history == [
        {"role": "user", "content": "Rate this CV"},
        {"role": "assistant", "content": "Your CV scored 75 out of 100."},
        {"role": "user", "content": "I mean 1 - 5"},
    ]


def test_review_cv_workflow_ends_at_respond() -> None:
    after_review: dict[str, Any] = {
        "cv": {
            "documents": _extracted_docs()[:1],
            "needs_extraction": False,
            "review": {"status": "complete", "overall_score": 75.0},
        },
        "router": {
            "route": "review_cv",
            "completed_actions": ["review_cv"],
        },
        "selection": {},
        "jobs": {"results": []},
    }

    assert chatbot_graph.route_after_cv_subagent(after_review) == "respond"


def test_route_message_reuses_existing_review_for_improve_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = {
        "status": "partial",
        "mode": "general",
        "focus": None,
        "target_role": None,
        "overall_score": None,
        "feedback": [
            {
                "title": "Strong Summary Statement",
                "observation": "The summary highlights backend skills.",
                "recommendation": "Add a metric.",
            }
        ],
    }
    state: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": "This is my cv, what do u think about this?\n[PDF CV uploaded separately]",
            },
            {
                "role": "assistant",
                "content": "Your CV has several strengths and areas for improvement.",
            },
            {"role": "user", "content": "What should I improve based on my CV?"},
        ],
        "cv": {
            "documents": _extracted_docs()[:1],
            "needs_extraction": False,
            "review": review,
        },
        "router": {"completed_actions": []},
        "selection": {},
        "jobs": {"results": []},
        "action_results": {},
    }
    fingerprint = chatbot_graph.action_fingerprint(
        "review_cv",
        {**state, "selection": {"review_mode": "general"}},
    )
    state["action_results"] = {
        "review_cv": {
            "fingerprint": fingerprint,
            "executed_at": "2026-08-15T10:00:00+00:00",
            "snapshot": {"cv": {"review": review}},
        }
    }
    _fake_router(monkeypatch, route="review_cv", is_follow_up=False)

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "respond"
    assert result["cv"]["review"]["feedback"][0]["title"] == "Strong Summary Statement"
