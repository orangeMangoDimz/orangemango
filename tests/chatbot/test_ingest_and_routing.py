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


def test_ingest_appends_new_cv_and_preserves_existing(monkeypatch: pytest.MonkeyPatch) -> None:
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
            "comparison": {"overview": "keep me", "candidates": [], "recommendation": "a"},
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
