from __future__ import annotations

from typing import Any, Literal

DetailLevel = Literal["summary", "full"]

COUNT_KEYS: tuple[str, ...] = ("likely", "possible", "unlikely", "insufficient")
HIGH_IMPACT_DIMENSIONS: frozenset[str] = frozenset(
    {"role_match", "required_skills", "seniority", "experience"}
)
MAX_PUBLIC_DETAILS: int = 3
LOW_PARTIAL_RATIO: float = 0.5


def _score(item: dict[str, Any]) -> dict[str, Any]:
    value: Any = item.get("score")
    return dict(value) if isinstance(value, dict) else {}


def _card(item: dict[str, Any]) -> dict[str, Any]:
    value: Any = item.get("job_card")
    return dict(value) if isinstance(value, dict) else {}


def _dimensions(score: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(entry)
        for entry in (score.get("dimensions") or [])
        if isinstance(entry, dict)
    ]


def _dimension_evidence(score: dict[str, Any]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for dimension in _dimensions(score):
        evidence.append(
            {
                "dimension": dimension.get("dimension"),
                "status": dimension.get("status"),
                "ratio": dimension.get("ratio"),
                "evidence": [
                    dict(item)
                    for item in (dimension.get("evidence") or [])
                    if isinstance(item, dict)
                ],
            }
        )
    return evidence


def _has_major_gap(score: dict[str, Any]) -> bool:
    for dimension in _dimensions(score):
        name: str = str(dimension.get("dimension") or "").strip().casefold()
        status: str = str(dimension.get("status") or "").strip().casefold()
        ratio_value: Any = dimension.get("ratio")
        try:
            ratio: float = float(ratio_value) if ratio_value is not None else 0.0
        except (TypeError, ValueError):
            ratio = 0.0
        if name in HIGH_IMPACT_DIMENSIONS and (
            status == "mismatched"
            or (
                status == "partial"
                and name in {"required_skills", "seniority", "experience"}
                and ratio < LOW_PARTIAL_RATIO
            )
        ):
            return True
    return False


def _assessment_key(item: dict[str, Any]) -> str:
    score: dict[str, Any] = _score(item)
    fit_verdict: str = str(score.get("fit_verdict") or "uncertain").strip().casefold()
    dimensions: list[dict[str, Any]] = _dimensions(score)
    has_supporting_evidence: bool = any(
        str(entry.get("status") or "").strip().casefold() in {"matched", "partial"}
        for entry in dimensions
    )
    if fit_verdict == "yes":
        return "likely"
    if fit_verdict == "no":
        return "unlikely"
    if fit_verdict == "unknown":
        return "insufficient"
    if _has_major_gap(score):
        return "unlikely"
    if has_supporting_evidence:
        return "possible"
    return "insufficient"


def _normalized_score(item: dict[str, Any]) -> float | None:
    score: dict[str, Any] = _score(item)
    raw: Any = score.get("normalized_score")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _best_assessment_for_group(rows: list[dict[str, Any]]) -> str:
    keys: list[str] = [_assessment_key(item) for item in rows]
    if "likely" in keys:
        return "likely"
    if "possible" in keys:
        return "possible"
    if "unlikely" in keys:
        return "unlikely"
    return "insufficient"


def _match_identity(item: dict[str, Any], index: int) -> str:
    key: str = str(item.get("job_key") or "").strip()
    if key:
        return key
    card: dict[str, Any] = _card(item)
    fallback_url: str = str(card.get("url") or "").strip()
    return fallback_url or f"match:{index}"


def _grouped_match_items(matches: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, item in enumerate(matches):
        identity: str = _match_identity(item, index)
        grouped.setdefault(identity, []).append(item)
    return list(grouped.values())


def _project_row(
    item: dict[str, Any],
    *,
    show_score: bool,
    detail_level: DetailLevel,
) -> dict[str, Any]:
    score: dict[str, Any] = _score(item)
    card: dict[str, Any] = _card(item)
    dimensions: list[dict[str, Any]] = _dimension_evidence(score)
    assessment_key: str = _assessment_key(item)
    row: dict[str, Any] = {
        "title": card.get("title") or "Untitled job",
        "location": card.get("location") or "",
        "posted_date": card.get("posted_date") or "",
        "salary": card.get("salary") or "",
        "url": card.get("url") or "",
        "assessment_key": assessment_key,
    }
    if show_score and _normalized_score(item) is not None:
        row["score"] = _normalized_score(item)
    if detail_level == "full":
        row["evidence"] = dimensions[:MAX_PUBLIC_DETAILS]
        row["reason_code"] = str(score.get("verdict_reason_code") or "").strip() or None
    return row


def _project_group(
    rows: list[dict[str, Any]],
    *,
    show_score: bool,
    detail_level: DetailLevel,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], float | None]:
    if not rows:
        return "insufficient", [], {}, None
    projected: list[dict[str, Any]] = [
        _project_row(row, show_score=show_score, detail_level=detail_level)
        for row in rows
    ]
    key: str = _best_assessment_for_group(rows)
    score_values: list[float] = [
        value
        for value in (
            _normalized_score(row) for row in rows
        )
        if value is not None
    ]
    max_score: float | None = max(score_values) if score_values else None
    # Use the best-scored row as representative when we need a single job object.
    representative: dict[str, Any] = max(
        projected,
        key=lambda row: (
            row.get("score") is not None,
            row.get("score") or -1.0,
        ),
    )
    if detail_level == "full" and len(projected) > 1:
        # Keep compact evidence when multiple CVs were compared for one job.
        representative["cv_count"] = len(rows)
    return key, projected, representative, max_score


def build_public_match_summary(
    matches: list[dict[str, Any]],
    *,
    show_score: bool = False,
    detail_level: DetailLevel = "summary",
) -> dict[str, Any]:
    del show_score, detail_level
    counts: dict[str, int] = {key: 0 for key in COUNT_KEYS}
    for rows in _grouped_match_items(matches):
        key, _, _, _ = _project_group(rows, show_score=False, detail_level="summary")
        if key in counts:
            counts[key] += 1
    total: int = sum(counts.values())
    assessed: int = total - counts["insufficient"]
    if assessed == 0:
        overall_fit = "insufficient"
    elif counts["likely"] == assessed:
        overall_fit = "likely"
    elif counts["unlikely"] == assessed:
        overall_fit = "unlikely"
    elif counts["likely"]:
        overall_fit = "mixed"
    elif counts["possible"] and counts["unlikely"] > counts["possible"]:
        overall_fit = "mostly_not_a_match"
    elif counts["possible"]:
        overall_fit = "possible"
    else:
        overall_fit = "mixed"
    return {
        "overall_fit": overall_fit,
        "has_likely_match": counts["likely"] > 0,
        "has_possible_match": counts["possible"] > 0,
        "counts": counts,
        "total": total,
    }


def build_public_match_recommendation(
    matches: list[dict[str, Any]],
    *,
    show_score: bool = False,
    detail_level: DetailLevel = "summary",
) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    for order, rows in enumerate(_grouped_match_items(matches)):
        key, _, representative, max_score = _project_group(
            rows,
            show_score=show_score,
            detail_level=detail_level,
        )
        if key not in {"likely", "possible"}:
            continue
        ranked.append(
            {
                "order": order,
                "assessment_key": key,
                "match": representative,
                "score": max_score,
            }
        )
    if not ranked:
        return {"status": "unavailable"}
    ranked.sort(
        key=lambda item: (
            0 if item["assessment_key"] == "likely" else 1,
            -(item["score"] if isinstance(item["score"], (int, float)) else -1.0),
            item["order"],
        )
    )
    chosen: dict[str, Any] = ranked[0]
    payload: dict[str, Any] = {
        "match": chosen["match"],
        "assessment_key": chosen["assessment_key"],
    }
    if show_score and isinstance(chosen.get("score"), (int, float)):
        payload["score"] = chosen["score"]
    return payload


def build_public_match_selected(
    selected: list[dict[str, Any]],
    *,
    selected_key: str,
    show_score: bool = False,
    detail_level: DetailLevel = "summary",
) -> dict[str, Any]:
    if not selected:
        return {"status": "unavailable", "selected_key": selected_key}
    key, _, representative, max_score = _project_group(
        selected,
        show_score=show_score,
        detail_level=detail_level,
    )
    if key == "unlikely" and detail_level == "summary":
        representative["status"] = "unfavorable"
    payload: dict[str, Any] = {
        "match": representative,
        "selected_key": selected_key,
        "assessment_key": key,
    }
    if show_score and isinstance(max_score, (int, float)):
        payload["score"] = max_score
    return payload


def build_public_match_assessment(
    matches: list[dict[str, Any]],
    *,
    show_score: bool = False,
    detail_level: DetailLevel = "summary",
) -> dict[str, Any]:
    return build_public_match_summary(
        matches,
        show_score=show_score,
        detail_level=detail_level,
    )
