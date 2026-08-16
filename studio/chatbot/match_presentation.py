from __future__ import annotations

from typing import Any, Literal
from urllib.parse import quote, urlparse

DetailLevel = Literal["summary", "full"]

ASSESSMENT_LABELS: dict[str, str] = {
    "likely": "Likely match",
    "possible": "Possible match",
    "unlikely": "Unlikely match",
    "insufficient": "Not enough information",
}
COUNT_KEYS: tuple[str, ...] = ("likely", "possible", "unlikely", "insufficient")
HIGH_IMPACT_DIMENSIONS: frozenset[str] = frozenset(
    {"role_match", "required_skills", "seniority", "experience"}
)
MAX_PUBLIC_DETAILS: int = 3
LOW_PARTIAL_RATIO: float = 0.5
PUBLIC_REASON_MESSAGES: dict[str, str] = {
    "INSUFFICIENT_COVERAGE": (
        "The listing does not include enough detail to confirm the fit."
    ),
    "NO_NORMALIZED_SCORE": "There is not enough comparable information to judge fit.",
    "CONFIRMED_MATCH": "The available evidence aligns well with your CV.",
    "CONFIRMED_NON_MATCH": "The available evidence shows a meaningful mismatch.",
    "ASSESSMENT_UNAVAILABLE": "The fit could not be assessed for this job.",
}


def _score(item: dict[str, Any]) -> dict[str, Any]:
    value: Any = item.get("score")
    return dict(value) if isinstance(value, dict) else {}


def _card(item: dict[str, Any]) -> dict[str, Any]:
    value: Any = item.get("job_card")
    return dict(value) if isinstance(value, dict) else {}


def _dimensions(score: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item) for item in (score.get("dimensions") or []) if isinstance(item, dict)
    ]


def _first_evidence(dimension: dict[str, Any]) -> dict[str, Any]:
    values: list[dict[str, Any]] = [
        item for item in (dimension.get("evidence") or []) if isinstance(item, dict)
    ]
    return values[0] if values else {}


def _display_value(value: Any) -> str:
    text: str = str(value or "").strip()
    return text.title() if text else ""


def _year_value(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    number: float = float(value)
    return str(int(number)) if number.is_integer() else str(round(number, 1))


def _role_strength(dimension: dict[str, Any]) -> str:
    evidence: dict[str, Any] = _first_evidence(dimension)
    role: str = _display_value(
        evidence.get("requirement") or evidence.get("matched_value")
    )
    if role:
        return f"The {role} role aligns with your background."
    return "The role aligns with your stated experience."


def _skill_names(
    score: dict[str, Any],
    field: str,
    groups: tuple[str, ...],
) -> list[str]:
    skills: Any = score.get("skills")
    if not isinstance(skills, dict):
        return []
    category: Any = skills.get(field)
    if not isinstance(category, dict):
        return []
    names: list[str] = []
    for key in groups:
        for value in category.get(key) or []:
            text: str = str(value or "").strip()
            if text and text not in names:
                names.append(text)
    return names


def _skills_strength(score: dict[str, Any], field: str) -> str:
    names: list[str] = _skill_names(score, field, ("matched", "partial"))[:2]
    if names:
        return f"Your CV shows relevant skills in {', '.join(names)}."
    return "Your CV shows skills relevant to the role."


def _seniority_strength(dimension: dict[str, Any]) -> str:
    evidence: dict[str, Any] = _first_evidence(dimension)
    candidate: str = _display_value(evidence.get("candidate"))
    requirement: str = _display_value(evidence.get("requirement"))
    if candidate and requirement:
        return f"Your {candidate} level is close to the role's {requirement} level."
    return "Your experience level appears compatible with the role."


def _experience_strength(dimension: dict[str, Any]) -> str:
    evidence: dict[str, Any] = _first_evidence(dimension)
    candidate: str = _year_value(evidence.get("candidate"))
    minimum: str = _year_value(evidence.get("minimum"))
    if candidate and minimum:
        candidate_unit: str = "year" if candidate == "1" else "years"
        if dimension.get("status") == "partial":
            verb: str = "comes" if candidate == "1" else "come"
            return (
                f"Your {candidate} {candidate_unit} of experience {verb} close to "
                f"the role's {minimum}-year requirement."
            )
        verb = "meets" if candidate == "1" else "meet"
        return (
            f"Your {candidate} {candidate_unit} of experience {verb} the role's "
            f"{minimum}-year requirement."
        )
    return "Your experience appears compatible with the role."


def _strength_for_dimension(
    score: dict[str, Any],
    dimension: dict[str, Any],
) -> str | None:
    name: str = str(dimension.get("dimension") or "")
    status: str = str(dimension.get("status") or "")
    if status not in {"matched", "partial"}:
        return None
    ratio: float = float(dimension.get("ratio") or 0.0)
    if (
        status == "partial"
        and name in {"seniority", "experience"}
        and ratio < LOW_PARTIAL_RATIO
    ):
        return None
    if name == "role_match":
        return _role_strength(dimension)
    if name == "required_skills":
        return _skills_strength(score, "required")
    if name == "preferred_skills":
        return _skills_strength(score, "preferred")
    if name == "seniority":
        return _seniority_strength(dimension)
    if name == "experience":
        return _experience_strength(dimension)
    if name == "location" and float(dimension.get("ratio") or 0.0) >= LOW_PARTIAL_RATIO:
        return "The location appears compatible with your CV."
    return None


def _seniority_gap(dimension: dict[str, Any]) -> str:
    evidence: dict[str, Any] = _first_evidence(dimension)
    candidate: str = _display_value(evidence.get("candidate"))
    requirement: str = _display_value(evidence.get("requirement"))
    if candidate and requirement:
        return (
            f"The role targets a {requirement} level while your CV is "
            f"currently closer to {candidate}."
        )
    return "The role's seniority does not align with your current experience."


def _experience_gap(dimension: dict[str, Any]) -> str:
    evidence: dict[str, Any] = _first_evidence(dimension)
    candidate: str = _year_value(evidence.get("candidate"))
    minimum: str = _year_value(evidence.get("minimum"))
    if candidate and minimum:
        return (
            f"The role asks for about {minimum} years of experience while "
            f"your CV shows about {candidate}."
        )
    return "The role asks for more experience than your CV currently shows."


def _gap_for_dimension(
    score: dict[str, Any],
    dimension: dict[str, Any],
) -> str | None:
    name: str = str(dimension.get("dimension") or "")
    status: str = str(dimension.get("status") or "")
    ratio: float = float(dimension.get("ratio") or 0.0)
    has_missing_skills: bool = name == "required_skills" and bool(
        _skill_names(score, "required", ("missing",))
    )
    is_gap: bool = status == "mismatched" or (
        status == "partial"
        and (
            (name in {"seniority", "experience"} and ratio < LOW_PARTIAL_RATIO)
            or has_missing_skills
        )
    )
    if not is_gap:
        return None
    if name == "role_match":
        return "The role is outside the main experience shown in your CV."
    if name == "required_skills":
        missing: list[str] = _skill_names(score, "required", ("missing",))[:2]
        if missing:
            return f"Your CV does not show the required skills: {', '.join(missing)}."
        return "Your CV does not show the role's required skills."
    if name == "seniority":
        return _seniority_gap(dimension)
    if name == "experience":
        return _experience_gap(dimension)
    if name == "location":
        return "The role's location does not appear to align with your CV."
    return None


def _unknown_for_dimension(dimension: dict[str, Any]) -> str | None:
    status: str = str(dimension.get("status") or "")
    if status not in {"unknown", "ambiguous", "not_specified"}:
        return None
    name: str = str(dimension.get("dimension") or "")
    messages: dict[str, str] = {
        "role_match": "The listing does not clearly describe the role.",
        "required_skills": "The listing does not provide required skills.",
        "preferred_skills": "The listing does not provide preferred skills.",
        "seniority": "The listing does not state the seniority level.",
        "experience": "The listing does not state the required experience.",
        "location": "The location fit cannot be confirmed.",
        "work_type": "The work arrangement is not stated.",
        "employment_type": "The employment type is not stated.",
        "salary": "The salary fit cannot be confirmed.",
    }
    return messages.get(name)


def _public_evidence(
    score: dict[str, Any],
) -> tuple[list[str], list[str], list[str], bool]:
    strengths: list[str] = []
    gaps: list[str] = []
    unknowns: list[str] = []
    has_major_gap: bool = False
    for dimension in _dimensions(score):
        strength: str | None = _strength_for_dimension(score, dimension)
        gap: str | None = _gap_for_dimension(score, dimension)
        unknown: str | None = _unknown_for_dimension(dimension)
        if strength and strength not in strengths:
            strengths.append(strength)
        if gap and gap not in gaps:
            gaps.append(gap)
            name: str = str(dimension.get("dimension") or "")
            status: str = str(dimension.get("status") or "")
            ratio: float = float(dimension.get("ratio") or 0.0)
            if name in HIGH_IMPACT_DIMENSIONS and (
                status == "mismatched"
                or (
                    status == "partial"
                    and name in {"required_skills", "seniority", "experience"}
                    and ratio < LOW_PARTIAL_RATIO
                )
            ):
                has_major_gap = True
        if unknown and unknown not in unknowns:
            unknowns.append(unknown)
    return strengths, gaps, unknowns, has_major_gap


def _assessment_key(
    verdict: str,
    strengths: list[str],
    has_major_gap: bool,
) -> str:
    if verdict == "yes":
        return "likely"
    if verdict == "no":
        return "unlikely"
    if verdict == "unknown":
        return "insufficient"
    if has_major_gap:
        return "unlikely"
    if strengths:
        return "possible"
    return "insufficient"


def _lower_first(value: str) -> str:
    if not value:
        return ""
    return value[0].lower() + value[1:]


def _combine(first: str, second: str) -> str:
    lead: str = first.rstrip(".")
    follow: str = _lower_first(second.rstrip("."))
    return f"{lead}, but {follow}."


def _summary_reason(
    key: str,
    strengths: list[str],
    gaps: list[str],
    unknowns: list[str],
) -> str:
    if key == "likely":
        return (
            strengths[0] if strengths else "The available evidence aligns with your CV."
        )
    if key == "unlikely":
        return (
            gaps[0] if gaps else "The available evidence shows a meaningful mismatch."
        )
    if key == "possible":
        if strengths and gaps:
            return _combine(strengths[0], gaps[0])
        if strengths and unknowns:
            return _combine(strengths[0], unknowns[0])
        return strengths[0] if strengths else "The role has some relevant overlap."
    if unknowns:
        return unknowns[0]
    return "The listing does not include enough detail to judge the fit."


def _public_reason(code: Any) -> str | None:
    return PUBLIC_REASON_MESSAGES.get(str(code or "").strip())


def _public_match_row(
    item: dict[str, Any],
    *,
    show_score: bool,
    detail_level: DetailLevel,
) -> tuple[str, dict[str, Any]]:
    score: dict[str, Any] = _score(item)
    card: dict[str, Any] = _card(item)
    strengths, gaps, unknowns, has_major_gap = _public_evidence(score)
    reason_code: str = str(score.get("verdict_reason_code") or "")
    mapped_reason: str | None = _public_reason(reason_code)
    if (
        reason_code in {"INSUFFICIENT_COVERAGE", "NO_NORMALIZED_SCORE"}
        and mapped_reason
        and mapped_reason not in unknowns
    ):
        unknowns.append(mapped_reason)
    key: str = _assessment_key(
        str(score.get("fit_verdict") or "uncertain"),
        strengths,
        has_major_gap,
    )
    why: str = (
        _summary_reason(key, strengths, gaps, unknowns)
        if strengths or gaps or unknowns
        else mapped_reason or _summary_reason(key, strengths, gaps, unknowns)
    )
    row: dict[str, Any] = {
        "title": card.get("title") or "Untitled job",
        "company": card.get("company") or "",
        "url": card.get("url") or "",
        "assessment": ASSESSMENT_LABELS[key],
        "why": why,
    }
    if show_score and score.get("normalized_score") is not None:
        row["score"] = score.get("normalized_score")
    if detail_level == "full":
        public_strengths: list[str] = strengths or (
            [why] if key in {"likely", "possible"} else []
        )
        public_gaps: list[str] = gaps or ([why] if key == "unlikely" else [])
        public_unknowns: list[str] = unknowns or (
            [why] if key == "insufficient" else []
        )
        row["strengths"] = public_strengths[:MAX_PUBLIC_DETAILS]
        row["gaps"] = public_gaps[:MAX_PUBLIC_DETAILS]
        row["unknowns"] = public_unknowns[:MAX_PUBLIC_DETAILS]
    return key, row


def _match_identity(item: dict[str, Any], index: int) -> str:
    key: str = str(item.get("job_key") or "").strip()
    if key:
        return key
    url: str = str(_card(item).get("url") or "").strip()
    return url or f"match:{index}"


def _grouped_match_items(
    matches: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, item in enumerate(matches):
        identity: str = _match_identity(item, index)
        grouped[identity] = [*grouped.get(identity, []), item]
    return list(grouped.values())


def _group_assessment_key(keys: list[str]) -> str:
    return keys[0] if len(set(keys)) == 1 else "possible"


def _attributed_value(label: str, value: Any) -> str:
    return f"{label}: {value}"


def _group_public_rows(
    items: list[dict[str, Any]],
    rows: list[tuple[str, dict[str, Any]]],
    *,
    show_score: bool,
    detail_level: DetailLevel,
) -> tuple[str, dict[str, Any]]:
    if len(rows) == 1:
        return rows[0]
    labels: list[str] = [
        str(item.get("cv_filename") or f"CV {index + 1}")
        for index, item in enumerate(items)
    ]
    keys: list[str] = [key for key, _ in rows]
    first: dict[str, Any] = rows[0][1]
    grouped: dict[str, Any] = {
        "title": first.get("title"),
        "company": first.get("company"),
        "url": first.get("url"),
        "assessment": "; ".join(
            _attributed_value(label, row.get("assessment"))
            for label, (_, row) in zip(labels, rows, strict=True)
        ),
        "why": "; ".join(
            _attributed_value(label, row.get("why"))
            for label, (_, row) in zip(labels, rows, strict=True)
        ),
    }
    if show_score:
        scores: list[str] = [
            _attributed_value(label, row.get("score"))
            for label, (_, row) in zip(labels, rows, strict=True)
            if row.get("score") is not None
        ]
        if scores:
            grouped["score"] = "; ".join(scores)
    if detail_level == "full":
        for field in ("strengths", "gaps", "unknowns"):
            grouped[field] = [
                _attributed_value(label, detail)
                for label, (_, row) in zip(labels, rows, strict=True)
                for detail in row.get(field) or []
            ][: MAX_PUBLIC_DETAILS * len(rows)]
    return _group_assessment_key(keys), grouped


def build_public_match_assessment(
    matches: list[dict[str, Any]],
    *,
    show_score: bool = False,
    detail_level: DetailLevel = "summary",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {key: 0 for key in COUNT_KEYS}
    for items in _grouped_match_items(matches):
        projected: list[tuple[str, dict[str, Any]]] = [
            _public_match_row(
                item,
                show_score=show_score,
                detail_level=detail_level,
            )
            for item in items
        ]
        key, row = _group_public_rows(
            items,
            projected,
            show_score=show_score,
            detail_level=detail_level,
        )
        counts[key] += 1
        rows.append(row)
    return {"counts": counts, "matches": rows, "detail_level": detail_level}


def _table_cell(value: Any) -> str:
    raw: str = "" if value is None else str(value)
    text: str = " ".join(raw.split())
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "\\`")
        .replace("*", "\\*")
        .replace("_", "\\_")
        .replace("~", "\\~")
    )


def _link(value: Any) -> str:
    url: str = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    safe_url: str = quote(url, safe=":/?#@!$&'*+,;=%")
    return f"[View]({safe_url})"


def render_match_table(
    assessment: dict[str, Any],
    *,
    show_score: bool = False,
) -> str:
    rows: list[dict[str, Any]] = [
        item for item in (assessment.get("matches") or []) if isinstance(item, dict)
    ]
    full_detail: bool = assessment.get("detail_level") == "full"
    headers: list[str] = ["Job", "Company", "Assessment"]
    if full_detail:
        headers.extend(["Strengths", "Gaps", "Missing information"])
    else:
        headers.append("Why")
    if show_score:
        headers.append("Score (0-100)")
    headers.append("Link")
    lines: list[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values: list[Any] = [
            row.get("title"),
            row.get("company"),
            row.get("assessment"),
        ]
        if full_detail:
            values.extend(
                [
                    "; ".join(row.get("strengths") or []) or "None confirmed",
                    "; ".join(row.get("gaps") or []) or "None confirmed",
                    "; ".join(row.get("unknowns") or []) or "None",
                ]
            )
        else:
            values.append(row.get("why"))
        if show_score:
            values.append(row.get("score"))
        values.append(_link(row.get("url")))
        cells: list[str] = [_table_cell(value) for value in values[:-1]]
        cells.append(str(values[-1]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_search_table(jobs: list[dict[str, Any]]) -> str:
    headers: list[str] = ["Job", "Company", "Location", "Pay", "Link"]
    lines: list[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for job in jobs:
        values: list[Any] = [
            job.get("title"),
            job.get("company"),
            job.get("location"),
            job.get("salary"),
            _link(job.get("url")),
        ]
        cells: list[str] = [_table_cell(value) for value in values[:-1]]
        cells.append(str(values[-1]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
