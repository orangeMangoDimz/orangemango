from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from studio.chatbot import graph as chatbot_graph


def _cv_doc(
    *,
    doc_id: str,
    filename: str,
    features: dict[str, Any] | None = None,
    review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "filename": filename,
        "cv_text": f"text for {filename}",
        "cv_result": {"validation_status": "valid"} if features else None,
        "cv_features": features,
        "cv_review": review,
    }


def _extracted_cv(doc_id: str = "1", filename: str = "a.pdf") -> dict[str, Any]:
    return _cv_doc(
        doc_id=doc_id,
        filename=filename,
        features={"role_tags": ["Backend Engineer"]},
    )


def _review(*, status: str = "complete", mode: str = "general") -> dict[str, Any]:
    return {
        "status": status,
        "mode": mode,
        "focus": None,
        "target_role": None,
        "overall_score": 75.0 if mode == "scored" else None,
        "feedback": [
            {
                "title": "Summary",
                "observation": "Clear",
                "recommendation": "Add metrics",
            }
        ],
    }


def _job(
    title: str,
    *,
    url: str = "",
    company: str = "",
    description: str = "",
    content_hash: str = "hash-a",
) -> dict[str, Any]:
    return {
        "job_card": {
            "title": title,
            "company": company,
            "url": url,
            "description": description,
        },
        "validation_status": "valid",
        "matching_features": {"content_hash": content_hash, "role": title},
    }


def _action_entry(
    fingerprint: str,
    snapshot: dict[str, Any],
    *,
    executed_at: datetime | None = None,
) -> dict[str, Any]:
    stamp = executed_at or datetime.now(timezone.utc)
    return {
        "fingerprint": fingerprint,
        "executed_at": stamp.isoformat(),
        "snapshot": snapshot,
    }


def _fake_router(
    monkeypatch: pytest.MonkeyPatch,
    *,
    route: str,
    job_source: str = "none",
    score_requested: bool = False,
    is_follow_up: bool = False,
    review_mode: str = "general",
    review_focus: str | None = None,
    review_target_role: str | None = None,
    selected_cv_id: str | None = None,
    selected_job_keys: list[str] | None = None,
    scrape_request: chatbot_graph.ScrapeRequest | None = None,
) -> None:
    class FakeDecision:
        def __init__(self) -> None:
            self.route = route
            self.reason = f"Fake route {route}."
            self.job_source = job_source
            self.score_requested = score_requested
            self.review_target_role = review_target_role
            self.review_mode = review_mode
            self.review_focus = review_focus
            self.review_mode_reason = "Fake review mode."
            self.needs_cv_text = False
            self.needs_cv_features = False
            self.is_follow_up = is_follow_up
            self.selected_cv_id = selected_cv_id
            self.selected_job_keys = selected_job_keys
            self.scrape_request = scrape_request or chatbot_graph.ScrapeRequest()

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


def _review_state(
    *,
    message: str,
    review: dict[str, Any] | None = None,
    action_results: dict[str, Any] | None = None,
    documents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    docs = documents or [_extracted_cv()]
    return {
        "messages": [{"role": "user", "content": message}],
        "cv": {
            "documents": docs,
            "needs_extraction": False,
            "review": review,
        },
        "router": {"completed_actions": []},
        "selection": {},
        "jobs": {"results": []},
        "action_results": action_results or {},
    }


def test_is_explicit_refresh_detects_review_again() -> None:
    assert chatbot_graph.is_explicit_refresh("Please review again") is True
    assert chatbot_graph.is_explicit_refresh("search again for backend jobs") is True
    assert chatbot_graph.is_explicit_refresh("refresh the results") is True
    assert chatbot_graph.is_explicit_refresh("rerun the match") is True
    assert (
        chatbot_graph.is_explicit_refresh("What should I improve based on my CV?")
        is False
    )


def test_review_fingerprint_changes_with_mode() -> None:
    state = _review_state(message="review")
    general = chatbot_graph.action_fingerprint(
        "review_cv",
        {**state, "selection": {"review_mode": "general"}},
    )
    scored = chatbot_graph.action_fingerprint(
        "review_cv",
        {**state, "selection": {"review_mode": "scored"}},
    )
    assert general
    assert scored
    assert general != scored


def test_record_completed_action_stores_reusable_review() -> None:
    state = _review_state(
        message="review",
        review=None,
    )
    state["selection"] = {"review_mode": "general"}
    update = {"cv": {"review": _review()}}

    result = chatbot_graph.record_completed_action(
        state,
        "review_cv",
        update,
        emit_result=True,
    )

    stored = result["action_results"]["review_cv"]
    assert stored["fingerprint"] == chatbot_graph.action_fingerprint(
        "review_cv",
        {**state, "selection": {"review_mode": "general"}},
    )
    assert stored["snapshot"]["cv"]["review"]["status"] == "complete"
    assert stored["executed_at"]


def test_record_completed_action_skips_failed_review() -> None:
    state = _review_state(message="review")
    state["selection"] = {"review_mode": "general"}
    update = {
        "cv": {
            "review": {
                "status": "unavailable",
                "mode": "general",
                "feedback": [],
                "validation_errors": ["CV review failed"],
            }
        }
    }

    result = chatbot_graph.record_completed_action(
        state,
        "review_cv",
        update,
        emit_result=True,
    )

    assert "review_cv" not in (result.get("action_results") or {})


def test_route_message_reuses_compatible_review_without_follow_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = _review()
    state = _review_state(
        message="What should I improve based on my CV?", review=review
    )
    fingerprint = chatbot_graph.action_fingerprint(
        "review_cv",
        {**state, "selection": {"review_mode": "general"}},
    )
    state["action_results"] = {
        "review_cv": _action_entry(fingerprint, {"cv": {"review": review}})
    }
    _fake_router(monkeypatch, route="review_cv", is_follow_up=False)

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "respond"
    assert result["cv"]["review"]["feedback"][0]["title"] == "Summary"
    assert "tool" not in str(result.get("messages") or []).lower()


def test_route_message_reruns_review_when_mode_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = _review()
    state = _review_state(message="Score this CV out of 100", review=review)
    fingerprint = chatbot_graph.action_fingerprint(
        "review_cv",
        {**state, "selection": {"review_mode": "general"}},
    )
    state["action_results"] = {
        "review_cv": _action_entry(fingerprint, {"cv": {"review": review}})
    }
    _fake_router(monkeypatch, route="review_cv", review_mode="scored")

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "review_cv"
    assert result["selection"]["review_mode"] == "scored"


def test_route_message_reruns_review_on_explicit_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = _review()
    state = _review_state(message="Please review again", review=review)
    fingerprint = chatbot_graph.action_fingerprint(
        "review_cv",
        {**state, "selection": {"review_mode": "general"}},
    )
    state["action_results"] = {
        "review_cv": _action_entry(fingerprint, {"cv": {"review": review}})
    }
    _fake_router(monkeypatch, route="review_cv")

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "review_cv"


def test_route_message_reuses_identical_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        _extracted_cv("1", "a.pdf"),
        _extracted_cv("2", "b.pdf"),
    ]
    comparison = {
        "overview": "A is stronger",
        "candidates": [
            {
                "filename": "a.pdf",
                "strengths": ["Python"],
                "weaknesses": ["Leadership"],
                "summary": "Backend",
            },
            {
                "filename": "b.pdf",
                "strengths": ["Go"],
                "weaknesses": ["Depth"],
                "summary": "Intern",
            },
        ],
        "recommendation": "Choose a.pdf",
    }
    state = _review_state(
        message="Compare them again at a high level",
        documents=documents,
    )
    state["cv"]["comparison"] = comparison
    fingerprint = chatbot_graph.action_fingerprint("compare_cvs", state)
    state["action_results"] = {
        "compare_cvs": _action_entry(fingerprint, {"cv": {"comparison": comparison}})
    }
    _fake_router(monkeypatch, route="compare_cvs")

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "respond"
    assert result["cv"]["comparison"]["recommendation"] == "Choose a.pdf"


def test_route_message_reuses_identical_pasted_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pasted = "Backend Engineer at Acme. Required: Python, PostgreSQL."
    job = _job("Backend Engineer", company="Acme", description=pasted)
    state = {
        "messages": [{"role": "user", "content": pasted}],
        "cv": {"documents": [_extracted_cv()], "needs_extraction": False},
        "router": {"completed_actions": []},
        "selection": {},
        "jobs": {"results": [job]},
        "action_results": {},
    }
    fingerprint = chatbot_graph.action_fingerprint(
        "extract_job",
        {**state, "selection": {"job_input_text": pasted, "job_source": "pasted"}},
    )
    state["action_results"] = {
        "extract_job": _action_entry(
            fingerprint,
            {
                "jobs": {
                    "results": [job],
                    "active_job_keys": [chatbot_graph.job_selection_key(job, 0)],
                }
            },
        )
    }
    _fake_router(monkeypatch, route="extract_job", job_source="pasted")

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "respond"
    assert result["jobs"]["active_job_keys"] == [
        chatbot_graph.job_selection_key(job, 0)
    ]


def test_route_message_reuses_fresh_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = chatbot_graph.ScrapeRequest(keywords=["backend"], sites=["linkedin"])
    job = _job("Backend", url="https://example.com/backend")
    state = {
        "messages": [{"role": "user", "content": "Search backend jobs on linkedin"}],
        "cv": {"documents": [_extracted_cv()], "needs_extraction": False},
        "router": {"completed_actions": []},
        "selection": {},
        "jobs": {
            "results": [job],
            "scrape_request": request.model_dump(exclude_none=True),
        },
        "action_results": {},
    }
    fingerprint = chatbot_graph.action_fingerprint(
        "search_jobs",
        {
            **state,
            "jobs": {
                **state["jobs"],
                "scrape_request": request.model_dump(exclude_none=True),
            },
        },
    )
    state["action_results"] = {
        "search_jobs": _action_entry(
            fingerprint,
            {
                "jobs": {
                    "results": [job],
                    "active_job_keys": ["url:https://example.com/backend"],
                    "scrape_total": 1,
                    "scrape_truncated": False,
                }
            },
        )
    }
    _fake_router(monkeypatch, route="search_jobs", scrape_request=request)

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "respond"
    assert result["jobs"]["active_job_keys"] == ["url:https://example.com/backend"]


def test_route_message_reruns_expired_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = chatbot_graph.ScrapeRequest(keywords=["backend"])
    job = _job("Backend", url="https://example.com/backend")
    state = {
        "messages": [{"role": "user", "content": "Search backend jobs"}],
        "cv": {"documents": [_extracted_cv()], "needs_extraction": False},
        "router": {"completed_actions": []},
        "selection": {},
        "jobs": {"results": [job]},
        "action_results": {},
    }
    fingerprint = chatbot_graph.action_fingerprint(
        "search_jobs",
        {
            **state,
            "jobs": {"scrape_request": request.model_dump(exclude_none=True)},
        },
    )
    state["action_results"] = {
        "search_jobs": _action_entry(
            fingerprint,
            {
                "jobs": {
                    "results": [job],
                    "active_job_keys": ["url:https://example.com/backend"],
                }
            },
            executed_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
    }
    _fake_router(monkeypatch, route="search_jobs", scrape_request=request)

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "search_jobs"


def test_route_message_reruns_changed_search_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_request = chatbot_graph.ScrapeRequest(keywords=["backend"])
    new_request = chatbot_graph.ScrapeRequest(keywords=["frontend"])
    job = _job("Backend", url="https://example.com/backend")
    state = {
        "messages": [{"role": "user", "content": "Search frontend jobs"}],
        "cv": {"documents": [_extracted_cv()], "needs_extraction": False},
        "router": {"completed_actions": []},
        "selection": {},
        "jobs": {"results": [job]},
        "action_results": {},
    }
    fingerprint = chatbot_graph.action_fingerprint(
        "search_jobs",
        {
            **state,
            "jobs": {"scrape_request": old_request.model_dump(exclude_none=True)},
        },
    )
    state["action_results"] = {
        "search_jobs": _action_entry(
            fingerprint,
            {
                "jobs": {
                    "results": [job],
                    "active_job_keys": ["url:https://example.com/backend"],
                }
            },
        )
    }
    _fake_router(monkeypatch, route="search_jobs", scrape_request=new_request)

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "search_jobs"


def test_route_message_reuses_identical_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job("Backend", url="https://example.com/backend", content_hash="abc")
    matches = [
        {
            "cv_id": "1",
            "cv_filename": "a.pdf",
            "job_key": "url:https://example.com/backend",
            "job_card": job["job_card"],
            "score": {
                "normalized_score": 88,
                "score_coverage": 0.8,
                "decision": "ready",
                "fit_verdict": "yes",
            },
        }
    ]
    state = {
        "messages": [{"role": "user", "content": "Are they a match with my CV?"}],
        "cv": {"documents": [_extracted_cv()], "needs_extraction": False},
        "router": {"completed_actions": []},
        "selection": {},
        "jobs": {"results": [job], "matches": matches},
        "action_results": {},
    }
    fingerprint = chatbot_graph.action_fingerprint(
        "match_jobs",
        {**state, "selection": {"job_source": "existing", "score_requested": True}},
    )
    state["action_results"] = {
        "match_jobs": _action_entry(fingerprint, {"jobs": {"matches": matches}})
    }
    _fake_router(
        monkeypatch,
        route="match_jobs",
        job_source="existing",
        score_requested=True,
    )

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "respond"
    assert result["jobs"]["matches"][0]["score"]["normalized_score"] == 88


def test_route_message_reruns_match_when_job_hash_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_job = _job("Backend", url="https://example.com/backend", content_hash="old")
    new_job = _job("Backend", url="https://example.com/backend", content_hash="new")
    matches = [
        {
            "cv_id": "1",
            "job_key": "url:https://example.com/backend",
            "score": {
                "normalized_score": 88,
                "decision": "ready",
                "fit_verdict": "yes",
            },
        }
    ]
    old_state = {
        "cv": {"documents": [_extracted_cv()], "needs_extraction": False},
        "selection": {"job_source": "existing", "score_requested": True},
        "jobs": {"results": [old_job], "matches": matches},
    }
    state = {
        "messages": [{"role": "user", "content": "Match my CV to this job"}],
        "cv": {"documents": [_extracted_cv()], "needs_extraction": False},
        "router": {"completed_actions": []},
        "selection": {},
        "jobs": {"results": [new_job], "matches": matches},
        "action_results": {
            "match_jobs": _action_entry(
                chatbot_graph.action_fingerprint("match_jobs", old_state),
                {"jobs": {"matches": matches}},
            )
        },
    }
    _fake_router(
        monkeypatch,
        route="match_jobs",
        job_source="existing",
        score_requested=True,
    )

    result = asyncio.run(chatbot_graph.route_message(state, chatbot_graph.ChatModel()))

    assert result["router"]["route"] == "match_jobs"


def test_search_result_is_fresh_within_ttl() -> None:
    fresh = _action_entry("abc", {"jobs": {"results": [{}]}})
    expired = _action_entry(
        "abc",
        {"jobs": {"results": [{}]}},
        executed_at=datetime.now(timezone.utc) - timedelta(hours=25),
    )
    assert chatbot_graph.search_result_is_fresh(fresh) is True
    assert chatbot_graph.search_result_is_fresh(expired) is False


def test_ingest_preserves_matches_and_action_results() -> None:
    state = {
        "messages": [{"role": "user", "content": "thanks"}],
        "cv": {"documents": [_extracted_cv()]},
        "jobs": {
            "matches": [{"cv_id": "1", "job_key": "url:https://example.com/a"}],
            "results": [_job("Backend", url="https://example.com/a")],
        },
        "action_results": {"review_cv": {"fingerprint": "keep-me"}},
    }

    result = chatbot_graph.ingest_input(state)

    assert "matches" not in (result.get("jobs") or {})
    assert "action_results" not in result


def test_compact_scrape_uses_job_selection_key_for_urlless_jobs() -> None:
    card = {
        "title": "Backend",
        "company": "Acme",
        "url": "",
        "description": "",
    }
    compact = chatbot_graph.compact_scrape_response({"jobs": [card, dict(card)]})
    assert len(compact["cards"]) == 1
    item = {"job_card": card}
    assert chatbot_graph.job_selection_key(item, 0) == chatbot_graph.job_selection_key(
        {"job_card": compact["cards"][0]},
        0,
    )


def test_match_with_changed_search_query_rescrapes() -> None:
    old_job = _job("Backend", url="https://example.com/old")
    state = {
        "cv": {"documents": [_extracted_cv()], "needs_extraction": False},
        "router": {"route": "match_jobs", "completed_actions": []},
        "selection": {"job_source": "search", "score_requested": True},
        "jobs": {
            "results": [old_job],
            "scrape_request": {"keywords": ["frontend"]},
            "active_job_keys": ["url:https://example.com/old"],
        },
        "action_results": {
            "search_jobs": _action_entry(
                chatbot_graph.action_fingerprint(
                    "search_jobs",
                    {
                        "jobs": {"scrape_request": {"keywords": ["backend"]}},
                    },
                ),
                {"jobs": {"results": [old_job]}},
            )
        },
    }

    assert chatbot_graph.route_into_job_subagent(state) == "scrape_jobs"


def test_match_with_fresh_matching_search_skips_rescrape() -> None:
    job = _job("Backend", url="https://example.com/backend")
    request = {"keywords": ["backend"]}
    state = {
        "cv": {"documents": [_extracted_cv()], "needs_extraction": False},
        "router": {"route": "match_jobs", "completed_actions": []},
        "selection": {"job_source": "search", "score_requested": True},
        "jobs": {
            "results": [job],
            "scrape_request": request,
            "active_job_keys": ["url:https://example.com/backend"],
        },
        "action_results": {
            "search_jobs": _action_entry(
                chatbot_graph.action_fingerprint(
                    "search_jobs",
                    {"jobs": {"scrape_request": request}},
                ),
                {"jobs": {"results": [job]}},
            )
        },
    }

    assert chatbot_graph.route_into_job_subagent(state) == "match_jobs"


def test_resolve_selected_jobs_prefers_active_search_keys() -> None:
    old_job = _job("Old", url="https://example.com/old")
    new_job = _job("New", url="https://example.com/new")
    state = {
        "selection": {},
        "jobs": {
            "results": [old_job, new_job],
            "active_job_keys": ["url:https://example.com/new"],
        },
    }

    selected = chatbot_graph.resolve_selected_jobs(state)

    assert [item["job_card"]["title"] for item in selected] == ["New"]
