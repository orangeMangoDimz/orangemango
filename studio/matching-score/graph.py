from __future__ import annotations


import re

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph


WEIGHTS = {
    "role_match": 25,
    "required_skills": 30,
    "preferred_skills": 10,
    "seniority": 10,
    "experience": 10,
    "location": 5,
    "work_type": 5,
    "employment_type": 3,
    "salary": 2,
}


SKILL_ALIASES = {
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "react.js": "React",
    "reactjs": "React",
    "react": "React",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "aws": "Amazon Web Services",
    "amazon web services": "Amazon Web Services",
    "gcp": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "google cloud platform": "Google Cloud Platform",
    "golang": "Go",
    "go": "Go",
    "sql querying": "SQL",
    "sql": "SQL",
    "express js": "Express.js",
    "express": "Express.js",
    "express.js": "Express.js",
}


MatchStatus = Literal[
    "matched",
    "partial",
    "mismatched",
    "unknown",
    "ambiguous",
    "not_specified",
]


SKILL_EQUIVALENTS: dict[str, set[str]] = {
    "relational databases": {"PostgreSQL", "MySQL", "SQL Server", "SQL"},
    "backend programming fundamentals": {
        "OOP",
        "Golang",
        "Go",
        "Node.js",
        "Java",
        "Express.js",
    },
    "debugging and troubleshooting": {
        "debugging",
        "troubleshooting",
        "Problem Solving",
    },
}


SKILL_REQUIREMENT_TERMS: dict[str, tuple[str, list[str]]] = {
    "json and http": ("all", ["JSON", "HTTP"]),
    "rest api and graphql": ("all", ["REST API", "GraphQL"]),
}


SENIORITY_ORDER = [
    "intern",
    "entry",
    "junior",
    "mid",
    "senior",
    "lead",
    "manager",
    "director",
]


OR_SPLIT = re.compile(r"\s+(?:or|atau)\s+", re.IGNORECASE)

UNKNOWN_VALUES = {None, "", "unknown", "unspecified"}


def normalize_skill_name(name: str) -> str:

    key = re.sub(r"\s+", " ", name.strip().lower())

    return SKILL_ALIASES.get(key, name.strip())


def normalize_skill_set(names: list[str]) -> set[str]:

    return {normalize_skill_name(n) for n in names if n and str(n).strip()}


def is_unknown(value: Any) -> bool:

    if value is None:
        return True

    if isinstance(value, str) and value.strip().lower() in UNKNOWN_VALUES:
        return True

    return False


def make_match_detail(
    status: MatchStatus,
    ratio: float | None,
    *,
    evidence: list[dict[str, Any]] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:

    detail: dict[str, Any] = {
        "status": status,
        "ratio": round(ratio, 4) if ratio is not None else None,
        "evidence": evidence or [],
    }

    if reason:
        detail["reason"] = reason

    return detail


def canonical_skill_set(user_skills: set[str]) -> set[str]:

    return normalize_skill_set(list(user_skills))


def skill_requirement_detail(
    requirement: str,
    user_skills: set[str],
) -> dict[str, Any]:

    raw = requirement.strip()

    if not raw:
        return make_match_detail(
            "ambiguous",
            None,
            reason="empty skill requirement",
        )

    canonical_user_skills = canonical_skill_set(user_skills)

    alternatives = [p.strip() for p in OR_SPLIT.split(raw) if p.strip()]

    if len(alternatives) > 1:
        alternative_details = [
            skill_requirement_detail(part, canonical_user_skills)
            for part in alternatives
        ]

        if any(detail["status"] == "matched" for detail in alternative_details):
            matched = next(
                detail
                for detail in alternative_details
                if detail["status"] == "matched"
            )

            return make_match_detail(
                "matched",
                1.0,
                evidence=matched["evidence"],
                reason="one OR alternative matched",
            )

        if any(
            detail["status"] in {"unknown", "ambiguous"}
            for detail in alternative_details
        ):
            return make_match_detail(
                "unknown",
                None,
                reason="no OR alternative could be confirmed",
            )

        return make_match_detail("mismatched", 0.0)

    key = re.sub(r"\s+", " ", raw.casefold())

    composite = SKILL_REQUIREMENT_TERMS.get(key)

    if composite:
        mode, terms = composite

        term_details = [
            skill_requirement_detail(term, canonical_user_skills) for term in terms
        ]

        matched_terms = [
            term
            for term, detail in zip(terms, term_details)
            if detail["status"] == "matched"
        ]

        evidence = [item for detail in term_details for item in detail["evidence"]]

        if mode == "all" and len(matched_terms) == len(terms):
            return make_match_detail("matched", 1.0, evidence=evidence)

        if not canonical_user_skills:
            return make_match_detail(
                "unknown", None, reason="candidate skills are missing"
            )

        if matched_terms:
            return make_match_detail(
                "partial",
                len(matched_terms) / len(terms),
                evidence=evidence,
            )

        return make_match_detail("mismatched", 0.0)

    normalized = normalize_skill_name(raw)

    normalized_key = normalized.casefold()

    if normalized_key in {skill.casefold() for skill in canonical_user_skills}:
        matched_by = "alias" if normalized_key != key else "exact"

        return make_match_detail(
            "matched",
            1.0,
            evidence=[
                {
                    "requirement": requirement,
                    "matched_value": normalized,
                    "matched_by": matched_by,
                }
            ],
        )

    equivalents = SKILL_EQUIVALENTS.get(key)

    if equivalents:
        canonical_equivalents = canonical_skill_set(equivalents)

        matches = canonical_equivalents & canonical_user_skills

        if matches:
            matched_value = sorted(matches)[0]

            return make_match_detail(
                "matched",
                1.0,
                evidence=[
                    {
                        "requirement": requirement,
                        "matched_value": matched_value,
                        "matched_by": "equivalent",
                    }
                ],
            )

    if not canonical_user_skills:
        return make_match_detail("unknown", None, reason="candidate skills are missing")

    return make_match_detail("mismatched", 0.0)


def skill_satisfied(requirement: str, user_skills: set[str]) -> bool:

    return skill_requirement_detail(requirement, user_skills)["status"] == "matched"


def match_skills(requirements: list[str], user_skills: set[str]) -> dict[str, Any]:

    cleaned_requirements = [
        str(req).strip() for req in requirements if str(req).strip()
    ]

    if not cleaned_requirements:
        return {
            **make_match_detail(
                "not_specified", None, reason="no skill requirements provided"
            ),
            "matched": [],
            "partial": [],
            "missing": [],
            "unknown": [],
            "details": [],
        }

    details = [
        {
            "requirement": requirement,
            **skill_requirement_detail(requirement, user_skills),
        }
        for requirement in cleaned_requirements
    ]

    matched = [item["requirement"] for item in details if item["status"] == "matched"]

    partial = [item["requirement"] for item in details if item["status"] == "partial"]

    unknown = [
        item["requirement"]
        for item in details
        if item["status"] in {"unknown", "ambiguous"}
    ]

    missing = [
        item["requirement"]
        for item in details
        if item["status"] in {"partial", "mismatched", "unknown", "ambiguous"}
    ]

    ratios = [item["ratio"] for item in details]

    evidence = [item for detail in details for item in detail["evidence"]]

    if unknown:
        status: MatchStatus = "unknown" if not matched and not partial else "partial"

        ratio = None

    elif len(matched) == len(details):
        status = "matched"

        ratio = 1.0

    elif matched or partial:
        status = "partial"

        ratio = sum(value or 0.0 for value in ratios) / len(ratios)

    else:
        status = "mismatched"

        ratio = 0.0

    return {
        **make_match_detail(status, ratio, evidence=evidence),
        "matched": matched,
        "partial": partial,
        "missing": missing,
        "unknown": unknown,
        "details": details,
    }


def role_match_detail(cv_roles: list[str], job_roles: list[str]) -> dict[str, Any]:

    if not job_roles:
        return make_match_detail(
            "not_specified", None, reason="no role requirements provided"
        )

    cv_norm = {str(role).casefold() for role in cv_roles if role}

    if not cv_norm:
        return make_match_detail("unknown", None, reason="candidate roles are missing")

    evidence = []

    hits = 0

    for role in job_roles:
        job_key = str(role).casefold()

        matched_role = next(
            (
                candidate
                for candidate in cv_norm
                if job_key in candidate or candidate in job_key
            ),
            None,
        )

        if matched_role:
            hits += 1

            evidence.append(
                {
                    "requirement": role,
                    "matched_value": matched_role,
                    "matched_by": "role_text",
                }
            )

    ratio = hits / len(job_roles)

    status: MatchStatus = (
        "matched" if hits == len(job_roles) else "partial" if hits else "mismatched"
    )

    return make_match_detail(status, ratio, evidence=evidence)


def role_ratio(cv_roles: list[str], job_roles: list[str]) -> float:

    detail = role_match_detail(cv_roles, job_roles)

    return detail["ratio"] if detail["ratio"] is not None else 0.0


def seniority_match_detail(
    cv_seniority: str | None,
    job_seniority: str | None,
) -> dict[str, Any]:

    if is_unknown(job_seniority):
        return make_match_detail(
            "not_specified", None, reason="job seniority is not specified"
        )

    if is_unknown(cv_seniority):
        return make_match_detail(
            "unknown", None, reason="candidate seniority is missing"
        )

    cv = str(cv_seniority).casefold()

    job = str(job_seniority).casefold()

    if cv not in SENIORITY_ORDER or job not in SENIORITY_ORDER:
        return make_match_detail(
            "ambiguous", None, reason="unrecognized seniority value"
        )

    gap = SENIORITY_ORDER.index(job) - SENIORITY_ORDER.index(cv)

    ratio = 1.0 if gap <= 0 else 0.6 if gap == 1 else 0.3 if gap == 2 else 0.0

    status: MatchStatus = (
        "matched" if ratio == 1.0 else "partial" if ratio else "mismatched"
    )

    return make_match_detail(
        status,
        ratio,
        evidence=[
            {
                "candidate": cv_seniority,
                "requirement": job_seniority,
                "matched_by": "seniority_order",
            }
        ],
    )


def seniority_ratio(cv_seniority: str | None, job_seniority: str | None) -> float:

    detail = seniority_match_detail(cv_seniority, job_seniority)

    return detail["ratio"] if detail["ratio"] is not None else 0.0


def experience_match_detail(
    cv_years: float | None,
    job_min: float | None,
    job_max: float | None,
) -> dict[str, Any]:

    if job_min is None and job_max is None:
        return make_match_detail(
            "not_specified", None, reason="job experience range is not specified"
        )

    if cv_years is None:
        return make_match_detail(
            "unknown", None, reason="candidate experience is missing"
        )

    try:
        if job_min is not None and cv_years < job_min:
            ratio = 1.0 if job_min <= 0 else max(0.0, cv_years / job_min)

            status: MatchStatus = "matched" if ratio == 1.0 else "partial"

        elif job_max is not None and cv_years > job_max:
            ratio = 1.0

            status = "matched"

        else:
            ratio = 1.0

            status = "matched"

    except TypeError:
        return make_match_detail(
            "ambiguous", None, reason="experience values are not numeric"
        )

    return make_match_detail(
        status,
        ratio,
        evidence=[
            {
                "candidate": cv_years,
                "minimum": job_min,
                "maximum": job_max,
                "matched_by": "experience_range",
            }
        ],
    )


def experience_ratio(
    cv_years: float | None,
    job_min: float | None,
    job_max: float | None,
) -> float:

    detail = experience_match_detail(cv_years, job_min, job_max)

    return detail["ratio"] if detail["ratio"] is not None else 0.0


def location_ratio(cv_features: dict[str, Any], job_features: dict[str, Any]) -> float:

    city = job_features.get("city")

    region = job_features.get("region")

    country = job_features.get("country")

    if is_unknown(city) and is_unknown(region) and is_unknown(country):
        return 1.0

    candidates = []

    if cv_features.get("current_location"):
        candidates.append(str(cv_features["current_location"]))

    candidates.extend(str(x) for x in (cv_features.get("preferred_locations") or []))

    if not candidates:
        return 0.0

    blob = " ".join(candidates).casefold()

    checks = [x for x in (city, region, country) if not is_unknown(x)]

    if not checks:
        return 1.0

    def part_hits(part: str) -> bool:

        text = str(part).casefold()

        if text in blob:
            return True

        tokens = [t for t in re.split(r"[\s,]+", text) if len(t) > 2]

        return any(token in blob for token in tokens)

    hits = sum(1 for part in checks if part_hits(part))

    return hits / len(checks) if hits else 0.0


def enum_match_detail(
    cv_values: list[str] | str | None,
    job_value: str | None,
) -> dict[str, Any]:

    if is_unknown(job_value):
        return make_match_detail(
            "not_specified", None, reason="job enum is not specified"
        )

    job_key = str(job_value).casefold().replace("-", "_").replace(" ", "_")

    values = [cv_values] if isinstance(cv_values, str) else list(cv_values or [])

    if not values:
        return make_match_detail(
            "unknown", None, reason="candidate enum preference is missing"
        )

    for value in values:
        cv_key = str(value).casefold().replace("-", "_").replace(" ", "_")

        if job_key == cv_key:
            return make_match_detail(
                "matched",
                1.0,
                evidence=[
                    {
                        "candidate": value,
                        "requirement": job_value,
                        "matched_by": "enum_exact",
                    }
                ],
            )

    return make_match_detail("mismatched", 0.0)


def enum_ratio(cv_values: list[str] | str | None, job_value: str | None) -> float:

    detail = enum_match_detail(cv_values, job_value)

    return detail["ratio"] if detail["ratio"] is not None else 0.0


def salary_match_detail(
    cv_features: dict[str, Any], job_features: dict[str, Any]
) -> dict[str, Any]:

    job_min = job_features.get("salary_minimum")

    job_max = job_features.get("salary_maximum")

    job_currency = job_features.get("salary_currency")

    job_period = job_features.get("salary_period")

    if (
        job_min is None
        and job_max is None
        and is_unknown(job_currency)
        and is_unknown(job_period)
    ):
        return make_match_detail(
            "not_specified", None, reason="job salary is not specified"
        )

    expected = cv_features.get("expected_salary") or {}

    cv_min = expected.get("minimum")

    cv_max = expected.get("maximum")

    if cv_min is None and cv_max is None:
        return make_match_detail(
            "unknown", None, reason="candidate salary expectation is missing"
        )

    if job_currency and expected.get("currency"):
        if str(job_currency).casefold() != str(expected["currency"]).casefold():
            return make_match_detail(
                "mismatched", 0.0, reason="salary currencies differ"
            )

    cv_ask = cv_min if cv_min is not None else cv_max

    if cv_ask is None:
        return make_match_detail(
            "ambiguous", None, reason="candidate salary is incomplete"
        )

    if job_max is not None and cv_ask > job_max:
        return make_match_detail(
            "mismatched", 0.0, reason="candidate salary exceeds job maximum"
        )

    if job_min is not None and cv_max is not None and cv_max < job_min:
        return make_match_detail(
            "mismatched", 0.0, reason="candidate salary is below job minimum"
        )

    return make_match_detail(
        "matched",
        1.0,
        evidence=[
            {
                "candidate_minimum": cv_min,
                "candidate_maximum": cv_max,
                "job_minimum": job_min,
                "job_maximum": job_max,
                "matched_by": "salary_range",
            }
        ],
    )


def salary_ratio(cv_features: dict[str, Any], job_features: dict[str, Any]) -> float:

    detail = salary_match_detail(cv_features, job_features)

    return detail["ratio"] if detail["ratio"] is not None else 0.0


def score_match(
    cv_result: dict[str, Any], job_result: dict[str, Any]
) -> dict[str, Any]:

    for label, extraction in (("CV", cv_result), ("job", job_result)):
        if extraction.get("validation_status") not in (None, "valid"):
            raise ValueError(
                f"{label} extraction is invalid: {extraction.get('validation_errors') or extraction.get('warnings')}"
            )

        if not isinstance(extraction.get("matching_features"), dict):
            raise ValueError(f"{label} extraction has no validated matching_features")

    cv = cv_result.get("matching_features") or {}

    job = job_result.get("matching_features") or {}

    user_skills = normalize_skill_set(cv.get("skill_names") or [])

    required = list(job.get("required_skill_names") or [])

    preferred = list(job.get("preferred_skill_names") or [])

    required_match = match_skills(required, user_skills)

    preferred_match = match_skills(preferred, user_skills)

    dimension_details: dict[str, dict[str, Any]] = {
        "role_match": role_match_detail(
            cv.get("role_tags") or [], job.get("role_tags") or []
        ),
        "required_skills": required_match,
        "preferred_skills": preferred_match,
        "seniority": seniority_match_detail(cv.get("seniority"), job.get("seniority")),
        "experience": experience_match_detail(
            cv.get("years_of_experience"),
            job.get("minimum_years_of_experience"),
            job.get("maximum_years_of_experience"),
        ),
        # Location is intentionally kept on the existing scoring behavior for now.
        "location": {"ratio": location_ratio(cv, job), "evidence": []},
        "work_type": enum_match_detail(
            cv.get("preferred_work_types") or [], job.get("work_type")
        ),
        "employment_type": enum_match_detail(
            cv.get("preferred_employment_types") or [],
            job.get("employment_type"),
        ),
        "salary": salary_match_detail(cv, job),
    }

    dimensions = []

    total = 0.0

    applicable_weight = 0

    for name, weight in WEIGHTS.items():
        detail = dimension_details[name]

        ratio = detail.get("ratio")

        status = detail.get("status", "legacy")

        is_applicable = name == "location" or status != "not_specified"

        points = weight * (ratio if ratio is not None else 0.0)

        if is_applicable:
            applicable_weight += weight

            total += points

        dimension = {
            "dimension": name,
            "weight": weight,
            "status": status,
            "ratio": round(ratio, 4) if ratio is not None else None,
            "points": round(points, 4),
            "applicable": is_applicable,
            "evidence": detail.get("evidence", []),
        }

        if detail.get("reason"):
            dimension["reason"] = detail["reason"]

        dimensions.append(dimension)

    confirmation_required = cv_result.get("confirmation_required") or []

    warnings = list(
        (cv_result.get("warnings") or []) + (job_result.get("warnings") or [])
    )

    candidate_ambiguous = list(cv.get("ambiguous_fields") or [])

    job_ambiguous = list(job.get("ambiguous_fields") or [])

    hard_requirements = list(job.get("hard_requirements") or [])

    eligibility_constraints = list(job.get("eligibility_constraints") or [])

    review_reasons = [
        f"{dimension['dimension']}: {dimension['status']}"
        for dimension in dimensions
        if dimension["dimension"] != "location"
        and dimension["status"] in {"unknown", "ambiguous"}
    ]

    review_reasons.extend(
        f"candidate ambiguous field: {field}" for field in candidate_ambiguous
    )

    review_reasons.extend(f"job ambiguous field: {field}" for field in job_ambiguous)

    if hard_requirements:
        review_reasons.append("unsupported hard requirements")

    if eligibility_constraints:
        review_reasons.append("unsupported eligibility constraints")

    review_reasons.extend(
        f"confirmation required: {field}" for field in confirmation_required
    )

    review_reasons.extend(f"warning: {warning}" for warning in warnings)

    decision = "needs_review" if review_reasons else "ready"

    normalized_score = (total / applicable_weight * 100) if applicable_weight else None

    return {
        "total_score": round(total, 2),
        "normalized_score": round(normalized_score, 2)
        if normalized_score is not None
        else None,
        "max_score": 100,
        "applicable_weight": applicable_weight,
        "score_coverage": round(applicable_weight / sum(WEIGHTS.values()), 4),
        "decision": decision,
        "review_reasons": review_reasons,
        "dimensions": dimensions,
        "skills": {
            "user_skills": sorted(user_skills),
            "required": required_match,
            "preferred": preferred_match,
        },
        "extraction_evidence": {
            "candidate_skills": list(cv.get("skill_evidence") or []),
            "required_job_skills": list(job.get("required_skill_evidence") or []),
            "preferred_job_skills": list(job.get("preferred_skill_evidence") or []),
            "candidate_fields": list(cv.get("field_evidence") or []),
            "job_fields": list(job.get("field_evidence") or []),
        },
        "hard_requirements": hard_requirements,
        "eligibility_constraints": eligibility_constraints,
        "confirmation_required": confirmation_required,
        "warnings": warnings,
    }


class MatchingState(TypedDict, total=False):
    cv_result: dict[str, Any]
    job_result: dict[str, Any]
    score: dict[str, Any]


def score_node(state: MatchingState) -> dict[str, Any]:
    return {"score": score_match(state["cv_result"], state["job_result"])}


builder = StateGraph(MatchingState)
builder.add_node("score", score_node)
builder.add_edge(START, "score")
builder.add_edge("score", END)
graph = builder.compile()
