from __future__ import annotations

import asyncio
from typing import Any

import pytest

from studio.chatbot import graph as chatbot_graph


def _job(title: str, *, url: str = "", company: str = "", description: str = "") -> dict[str, Any]:
    return {
        "job_card": {
            "title": title,
            "company": company,
            "url": url,
            "description": description,
        },
        "validation_status": "valid",
        "matching_features": {"role": title},
    }


def _cv(
    *,
    doc_id: str,
    filename: str,
) -> dict[str, Any]:
    return {
        "id": doc_id,
        "filename": filename,
        "cv_text": f"text for {filename}",
        "cv_result": {"extract": {"name": filename}},
        "cv_features": {"role_tags": ["Backend Engineer"]},
        "cv_review": None,
    }


def test_merge_job_results_appends_and_dedupes_by_url() -> None:
    existing = [_job("Old", url="https://example.com/a")]
    incoming = [
        _job("Updated A", url="https://example.com/a"),
        _job("New", url="https://example.com/b"),
    ]

    merged = chatbot_graph.merge_job_results(existing, incoming)

    assert len(merged) == 2
    assert merged[0]["job_card"]["title"] == "Updated A"
    assert merged[1]["job_card"]["title"] == "New"


def test_resolve_selected_cvs_returns_all_without_selection() -> None:
    state = {
        "cv": {
            "documents": [
                _cv(doc_id="1", filename="a.pdf"),
                _cv(doc_id="2", filename="b.pdf"),
            ]
        },
        "selection": {},
    }

    selected = chatbot_graph.resolve_selected_cvs(state)

    assert [doc["id"] for doc in selected] == ["1", "2"]


def test_resolve_selected_cvs_filters_named_cv() -> None:
    state = {
        "cv": {
            "documents": [
                _cv(doc_id="1", filename="a.pdf"),
                _cv(doc_id="2", filename="b.pdf"),
            ]
        },
        "selection": {"selected_cv_id": "2"},
    }

    selected = chatbot_graph.resolve_selected_cvs(state)

    assert [doc["id"] for doc in selected] == ["2"]


def test_calculate_job_matches_builds_cv_job_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "cv": {
            "documents": [
                _cv(doc_id="cv-a", filename="a.pdf"),
                _cv(doc_id="cv-b", filename="b.pdf"),
            ]
        },
        "selection": {},
        "jobs": {
            "results": [
                _job("Backend", url="https://example.com/backend"),
                _job("Frontend", url="https://example.com/frontend"),
            ]
        },
        "errors": [],
    }

    async def fake_ainvoke(payload: dict[str, Any]) -> dict[str, Any]:
        cv_name = payload["cv_result"]["extract"]["name"]
        job_title = payload["job_result"]["job_card"]["title"]
        score = 90 if cv_name == "a.pdf" and job_title == "Backend" else 40
        return {
            "score": {
                "normalized_score": score,
                "decision": "strong_fit" if score >= 80 else "weak_fit",
                "review_reasons": [f"{cv_name}->{job_title}"],
            }
        }

    class FakeGraph:
        async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            return await fake_ainvoke(payload)

    monkeypatch.setattr(chatbot_graph, "matching_score_graph", FakeGraph())

    result = asyncio.run(chatbot_graph.calculate_job_matches(state))
    matches = result["jobs"]["matches"]

    assert len(matches) == 4
    assert matches[0]["cv_id"] == "cv-a"
    assert matches[0]["cv_filename"] == "a.pdf"
    assert matches[0]["job_card"]["title"] == "Backend"
    assert matches[0]["job_key"] == "url:https://example.com/backend"
    assert {item["cv_id"] for item in matches} == {"cv-a", "cv-b"}
    assert {item["job_card"]["title"] for item in matches} == {"Backend", "Frontend"}
