from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_file() -> str:
    path = PROJECT_ROOT / ".env"
    if path.exists():
        return str(path)
    return str(PROJECT_ROOT.parent / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_env_file(), extra="ignore")

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "orangemango"


settings = Settings()

os.environ["OPENAI_API_KEY"] = settings.openai_api_key
if settings.langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_TRACING"] = str(settings.langsmith_tracing).lower()
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project


import uuid
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator

from app.services.text_normalization import collapse_whitespace

Seniority = Literal[
    "intern",
    "entry",
    "junior",
    "mid",
    "senior",
    "lead",
    "manager",
    "director",
    "unknown",
]
EducationLevel = Literal[
    "high_school", "diploma", "bachelor", "master", "doctorate", "unspecified"
]
SalaryPeriod = Literal["hourly", "daily", "monthly", "yearly"]

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
    "express js": "Express.js",
    "express": "Express.js",
    "express.js": "Express.js",
}
SENIORITY_VALUES = {
    "intern",
    "entry",
    "junior",
    "mid",
    "senior",
    "lead",
    "manager",
    "director",
    "unknown",
}
EDUCATION_VALUES = {
    "high_school",
    "diploma",
    "bachelor",
    "master",
    "doctorate",
    "unspecified",
}
PERIOD_VALUES = {"hourly", "daily", "monthly", "yearly"}


class ExpectedSalary(BaseModel):
    minimum: float | None = Field(default=None, ge=0)
    maximum: float | None = Field(default=None, ge=0)
    currency: str | None = None
    period: SalaryPeriod | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "ExpectedSalary":
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("expected salary minimum cannot exceed maximum")
        return self


class FieldEvidence(BaseModel):
    field: str
    value: str | None = None
    inferred: bool = False
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: str | None = None

    @model_validator(mode="after")
    def inferred_values_require_evidence(self) -> "FieldEvidence":
        if self.inferred and (self.confidence is None or not self.evidence):
            raise ValueError("inferred fields require confidence and evidence")
        return self


class MatchingFeatures(BaseModel):
    role_tags: list[str] = Field(default_factory=list)
    skill_names: list[str] = Field(default_factory=list)
    seniority: Seniority = "unknown"
    years_of_experience: float | None = Field(default=None, ge=0)
    current_location: str | None = None
    preferred_locations: list[str] = Field(default_factory=list)
    preferred_work_types: list[str] = Field(default_factory=list)
    preferred_employment_types: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    education_level: EducationLevel = "unspecified"
    languages: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    expected_salary: ExpectedSalary = Field(default_factory=ExpectedSalary)
    skill_evidence: list[dict[str, Any]] = Field(default_factory=list)
    field_evidence: list[dict[str, Any]] = Field(default_factory=list)
    ambiguous_fields: list[str] = Field(default_factory=list)


class SkillEvidence(BaseModel):
    name: str
    inferred: bool = False
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: str | None = None

    @model_validator(mode="after")
    def inferred_skills_require_evidence(self) -> "SkillEvidence":
        if self.inferred and (self.confidence is None or not self.evidence):
            raise ValueError("inferred skills require confidence and evidence")
        return self


class WorkExperience(BaseModel):
    job_title: str | None = None
    normalized_role: str | None = None
    company: str | None = None
    industry: str | None = None
    location: str | None = None
    work_type: str | None = None
    employment_type: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    duration_months: int | None = Field(default=None, ge=0)
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    skills_used: list[str] = Field(default_factory=list)
    raw_snippet: str | None = None


class SuggestedPreference(BaseModel):
    field: str
    value: str | list[str] | float | int | bool | None = None
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(min_length=1)


class CvExtract(BaseModel):
    role_tags: list[str] = Field(default_factory=list)
    skills: list[SkillEvidence] = Field(default_factory=list)
    seniority: Seniority = "unknown"
    years_of_experience: float | None = Field(default=None, ge=0)
    current_location: str | None = None
    preferred_locations: list[str] = Field(default_factory=list)
    preferred_work_types: list[str] = Field(default_factory=list)
    preferred_employment_types: list[str] = Field(default_factory=list)
    industries: list[str] = Field(default_factory=list)
    education_level: EducationLevel = "unspecified"
    languages: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    expected_salary: ExpectedSalary = Field(default_factory=ExpectedSalary)
    work_experiences: list[WorkExperience] = Field(default_factory=list)
    suggested_preferences: list[SuggestedPreference] = Field(default_factory=list)
    field_evidence: list[FieldEvidence] = Field(default_factory=list)
    ambiguous_fields: list[str] = Field(default_factory=list)
    confirmation_required: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CvState(TypedDict, total=False):
    cv_text: str
    extract: dict[str, Any]
    matching_features: dict[str, Any] | None
    suggested_preferences: list[dict[str, Any]]
    confirmation_required: list[str]
    warnings: list[str]
    validation_status: Literal["valid", "invalid"]
    validation_errors: list[str]


EXTRACTION_PROMPT = """You are a CV data extraction agent.

Your task is to extract structured candidate profile data from the provided CV. The extracted data will be used by a rule-based service to match the candidate against job postings.

Rules:
1. Extract only information supported by the CV.
2. Never invent missing information.
3. Use null, unknown, or an empty array when unavailable.
4. Normalize role names, skill names, locations, dates, employment types, and education levels.
5. Separate explicitly stated information from inferred information.
6. Every inferred value must include confidence from 0.0 to 1.0 and short evidence.
7. Do not assume desired role, salary, location, or work arrangement unless explicitly stated.
8. Suggested preferences must be confirmed by the user before scoring.
9. Preserve raw snippets for audit and confirmation.
10. Return valid structured data only.
11. For every non-empty field that can affect matching, add field_evidence with the exact supporting text.
12. Put values inferred from context, rather than explicitly stated, in ambiguous_fields and mark their field_evidence as inferred.

Normalize common skill aliases, for example:
- js -> JavaScript
- ts -> TypeScript
- node or node.js -> Node.js
- react.js -> React
- postgres -> PostgreSQL
- mongo -> MongoDB
- k8s -> Kubernetes
- aws -> Amazon Web Services
- gcp -> Google Cloud Platform
- golang -> Go
- sql querying -> SQL
- express js -> Express.js

Do not merge non-equivalent skills such as Java and JavaScript, React and React Native, or PostgreSQL and MySQL.

For preferred_* and expected_salary: only fill when the CV explicitly states them.
Put uncertain preference fields into confirmation_required.
You may add suggested_preferences with confidence and evidence, but do not treat them as confirmed.
"""


def normalize_skill_name(name: str) -> str:
    key = collapse_whitespace(name).lower()
    return SKILL_ALIASES.get(key, name.strip())


def unique_preserve(items: list[str]) -> list[str]:
    # Keep first occurrence; drop later duplicates (case-insensitive).
    # ["React", "js", "react", "JS"] -> ["React", "js"]
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        marker = cleaned.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        result.append(cleaned)
    return result


MAX_PDF_BYTES = 10 * 1024 * 1024
MATCHING_EVIDENCE_FIELDS = (
    "role_tags",
    "seniority",
    "years_of_experience",
    "current_location",
    "preferred_locations",
    "preferred_work_types",
    "preferred_employment_types",
    "expected_salary",
)


def validate_pdf_upload(
    filename: str,
    content: bytes | bytearray | memoryview,
    max_bytes: int = MAX_PDF_BYTES,
) -> None:
    normalized_filename = filename.replace("\\", "/")
    safe_name = Path(normalized_filename).name
    if (
        not filename
        or normalized_filename != filename
        or safe_name != normalized_filename
        or safe_name in {".", ".."}
    ):
        raise ValueError("invalid PDF filename")
    if Path(safe_name).suffix.casefold() != ".pdf":
        raise ValueError("only .pdf uploads are supported")
    payload = bytes(content)
    if len(payload) > max_bytes:
        raise ValueError(f"PDF exceeds the {max_bytes} byte limit")
    if not payload.startswith(b"%PDF-"):
        raise ValueError("uploaded content is not a PDF")


def safe_pdf_path(upload_dir: Path, filename: str) -> Path:
    safe_name = Path(filename.replace("\\", "/")).name
    return upload_dir / f"{uuid.uuid4().hex}_{safe_name}"


def mark_unproven_fields(
    extract: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], list[str]]:
    field_evidence = [
        dict(item) for item in extract.get("field_evidence") or [] if item.get("field")
    ]
    evidence_by_field = {item["field"]: item for item in field_evidence}
    ambiguous = unique_preserve(extract.get("ambiguous_fields") or [])

    for field in MATCHING_EVIDENCE_FIELDS:
        value = extract.get(field)
        if field == "expected_salary":
            value_present = any(
                (value or {}).get(key) not in (None, "", "unknown", "unspecified")
                for key in ("minimum", "maximum", "currency", "period")
            )
        else:
            value_present = value not in (None, "", [], "unknown")
        if not value_present:
            continue
        evidence = evidence_by_field.get(field)
        if evidence is None or evidence.get("inferred"):
            if field not in ambiguous:
                ambiguous.append(field)
            warnings.append(f"{field} needs explicit evidence before scoring")

    extract["field_evidence"] = field_evidence
    extract["ambiguous_fields"] = unique_preserve(ambiguous)
    return extract, warnings


def extract_node(state: CvState) -> dict[str, Any]:
    llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    structured = llm.with_structured_output(CvExtract)
    result = structured.invoke(
        [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": f"CV INPUT:\n{state['cv_text']}"},
        ]
    )
    payload = result.model_dump()
    return {
        "extract": payload,
        "suggested_preferences": payload.get("suggested_preferences", []),
        "confirmation_required": payload.get("confirmation_required", []),
        "warnings": payload.get("warnings", []),
    }


def normalize_node(state: CvState) -> dict[str, Any]:
    extract = dict(state.get("extract") or {})
    warnings = list(state.get("warnings") or [])

    skills = []
    for skill in extract.get("skills") or []:
        item = dict(skill)
        item["name"] = normalize_skill_name(item.get("name") or "")
        if item["name"]:
            skills.append(item)
    extract["skills"] = skills

    experiences = []
    for exp in extract.get("work_experiences") or []:
        item = dict(exp)
        item["skills_used"] = unique_preserve(
            [normalize_skill_name(s) for s in item.get("skills_used") or []]
        )
        experiences.append(item)
    extract["work_experiences"] = experiences

    seniority = (extract.get("seniority") or "unknown").lower()
    if seniority not in SENIORITY_VALUES:
        warnings.append(
            f"Invalid seniority '{extract.get('seniority')}' reset to unknown"
        )
        seniority = "unknown"
    extract["seniority"] = seniority

    education = (extract.get("education_level") or "unspecified").lower()
    if education not in EDUCATION_VALUES:
        warnings.append(
            f"Invalid education_level '{extract.get('education_level')}' reset to unspecified"
        )
        education = "unspecified"
    extract["education_level"] = education

    salary = dict(extract.get("expected_salary") or {})
    period = salary.get("period")
    if period is not None and period not in PERIOD_VALUES:
        warnings.append(f"Invalid salary period '{period}' reset to null")
        salary["period"] = None
    extract["expected_salary"] = salary

    extract["role_tags"] = unique_preserve(extract.get("role_tags") or [])
    extract["preferred_locations"] = unique_preserve(
        extract.get("preferred_locations") or []
    )
    extract["preferred_work_types"] = unique_preserve(
        extract.get("preferred_work_types") or []
    )
    extract["preferred_employment_types"] = unique_preserve(
        extract.get("preferred_employment_types") or []
    )
    extract["industries"] = unique_preserve(extract.get("industries") or [])
    extract["languages"] = unique_preserve(extract.get("languages") or [])
    extract["certifications"] = unique_preserve(extract.get("certifications") or [])
    extract, warnings = mark_unproven_fields(extract, warnings)

    return {"extract": extract, "warnings": warnings}


def suggest_preferences_node(state: CvState) -> dict[str, Any]:
    # Soft-fill missing prefs from CV history; mark them as needing confirmation.
    # preferred_locations=[] + current_location="Jakarta" -> suggest ["Jakarta"]
    # preferred_work_types=[] + past work_type="remote" -> suggest ["remote"]
    # preferred_employment_types=[] + past employment_type="full-time" -> suggest ["full-time"]

    extract = dict(state.get("extract") or {})
    suggestions = [dict(s) for s in state.get("suggested_preferences") or []]
    confirmation = list(state.get("confirmation_required") or [])

    existing_fields = {s.get("field") for s in suggestions}

    if not extract.get("preferred_locations") and extract.get("current_location"):
        if "preferred_locations" not in existing_fields:
            suggestions.append(
                {
                    "field": "preferred_locations",
                    "value": [extract["current_location"]],
                    "confidence": 0.45,
                    "evidence": "Inferred from current_location; not stated as preference",
                }
            )
        if "preferred_locations" not in confirmation:
            confirmation.append("preferred_locations")

    if not extract.get("preferred_work_types"):
        work_types = unique_preserve(
            [
                exp.get("work_type")
                for exp in extract.get("work_experiences") or []
                if exp.get("work_type")
            ]
        )
        if work_types and "preferred_work_types" not in existing_fields:
            suggestions.append(
                {
                    "field": "preferred_work_types",
                    "value": work_types,
                    "confidence": 0.4,
                    "evidence": "Inferred from past work_type values; not stated as preference",
                }
            )
            if "preferred_work_types" not in confirmation:
                confirmation.append("preferred_work_types")

    if not extract.get("preferred_employment_types"):
        employment_types = unique_preserve(
            [
                exp.get("employment_type")
                for exp in extract.get("work_experiences") or []
                if exp.get("employment_type")
            ]
        )
        if employment_types and "preferred_employment_types" not in existing_fields:
            suggestions.append(
                {
                    "field": "preferred_employment_types",
                    "value": employment_types,
                    "confidence": 0.4,
                    "evidence": "Inferred from past employment_type values; not stated as preference",
                }
            )
            if "preferred_employment_types" not in confirmation:
                confirmation.append("preferred_employment_types")

    salary = extract.get("expected_salary") or {}
    salary_empty = all(
        salary.get(key) is None for key in ("minimum", "maximum", "currency", "period")
    )
    if salary_empty and "expected_salary" not in confirmation:
        confirmation.append("expected_salary")

    extract["suggested_preferences"] = suggestions
    extract["confirmation_required"] = unique_preserve(confirmation)

    return {
        "extract": extract,
        "suggested_preferences": suggestions,
        "confirmation_required": extract["confirmation_required"],
    }


def build_matching_features_node(state: CvState) -> dict[str, Any]:
    # Shrink full extract down to the matching_features JSON used for job matching.
    # extract.skills=[{name:"React",...}] -> matching_features.skill_names=["React"]
    extract = state.get("extract") or {}
    normalized_skills = []
    skill_evidence = []
    for skill in extract.get("skills") or []:
        item = dict(skill)
        item["name"] = normalize_skill_name(item.get("name") or "")
        if not item["name"]:
            continue
        normalized_skills.append(item["name"])
        skill_evidence.append(
            {
                "name": item["name"],
                "inferred": bool(item.get("inferred", False)),
                "confidence": item.get("confidence"),
                "evidence": item.get("evidence"),
            }
        )
    skill_names = unique_preserve(normalized_skills)
    field_evidence = [dict(item) for item in extract.get("field_evidence") or []]
    features = MatchingFeatures(
        role_tags=unique_preserve(extract.get("role_tags") or []),
        skill_names=skill_names,
        seniority=extract.get("seniority") or "unknown",
        years_of_experience=extract.get("years_of_experience"),
        current_location=extract.get("current_location"),
        preferred_locations=unique_preserve(extract.get("preferred_locations") or []),
        preferred_work_types=unique_preserve(extract.get("preferred_work_types") or []),
        preferred_employment_types=unique_preserve(
            extract.get("preferred_employment_types") or []
        ),
        industries=unique_preserve(extract.get("industries") or []),
        education_level=extract.get("education_level") or "unspecified",
        languages=unique_preserve(extract.get("languages") or []),
        certifications=unique_preserve(extract.get("certifications") or []),
        expected_salary=ExpectedSalary(**(extract.get("expected_salary") or {})),
        skill_evidence=skill_evidence,
        field_evidence=field_evidence,
        ambiguous_fields=unique_preserve(extract.get("ambiguous_fields") or []),
    )
    return {"matching_features": features.model_dump()}


def validate_node(state: CvState) -> dict[str, Any]:
    warnings = list(state.get("warnings") or [])
    try:
        features = MatchingFeatures.model_validate(state.get("matching_features") or {})
        payload = features.model_dump()
    except Exception as exc:
        error = f"matching_features validation failed: {exc}"
        warnings.append(error)
        return {
            "matching_features": None,
            "warnings": warnings,
            "validation_status": "invalid",
            "validation_errors": [error],
        }
    return {
        "matching_features": payload,
        "warnings": warnings,
        "validation_status": "valid",
        "validation_errors": [],
    }


builder = StateGraph(CvState)
builder.add_node("extract", extract_node)
builder.add_node("normalize", normalize_node)
builder.add_node("suggest_preferences", suggest_preferences_node)
builder.add_node("build_matching_features", build_matching_features_node)
builder.add_node("validate", validate_node)

builder.add_edge(START, "extract")
builder.add_edge("extract", "normalize")
builder.add_edge("normalize", "suggest_preferences")
builder.add_edge("suggest_preferences", "build_matching_features")
builder.add_edge("build_matching_features", "validate")
builder.add_edge("validate", END)

graph = builder.compile()
