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


import hashlib
import json
import re
from typing import Any, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator

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
WorkType = Literal["remote", "hybrid", "onsite", "unknown"]
EmploymentType = Literal[
    "full_time",
    "part_time",
    "contract",
    "internship",
    "freelance",
    "temporary",
    "unknown",
]
SalaryPeriod = Literal["hourly", "daily", "monthly", "yearly", "unknown"]

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
WORK_TYPE_VALUES = {"remote", "hybrid", "onsite", "unknown"}
EMPLOYMENT_TYPE_VALUES = {
    "full_time",
    "part_time",
    "contract",
    "internship",
    "freelance",
    "temporary",
    "unknown",
}
PERIOD_VALUES = {"hourly", "daily", "monthly", "yearly", "unknown"}


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


class NormalizedLocation(BaseModel):
    city: str | None = None
    region: str | None = None
    country: str | None = None
    remote_scope: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: str | None = None


class SalaryExtract(BaseModel):
    minimum: float | None = Field(default=None, ge=0)
    maximum: float | None = Field(default=None, ge=0)
    currency: str | None = None
    period: SalaryPeriod | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: str | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "SalaryExtract":
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("salary minimum cannot exceed maximum")
        return self


class ExperienceRange(BaseModel):
    minimum_years: float | None = Field(default=None, ge=0)
    maximum_years: float | None = Field(default=None, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: str | None = None

    @model_validator(mode="after")
    def validate_range(self) -> "ExperienceRange":
        if (
            self.minimum_years is not None
            and self.maximum_years is not None
            and self.minimum_years > self.maximum_years
        ):
            raise ValueError("experience minimum cannot exceed maximum")
        return self


class HardRequirement(BaseModel):
    description: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: str | None = None


class ExtractionMetadata(BaseModel):
    source: str | None = None
    source_job_id: str | None = None
    job_url: str | None = None
    content_hash: str | None = None
    model: str | None = None


class JobExtract(BaseModel):
    source: str | None = None
    source_job_id: str | None = None
    job_url: str | None = None
    content_hash: str | None = None
    raw_title: str | None = None
    normalized_title: str | None = None
    company: str | None = None
    industry: str | None = None
    role_tags: list[str] = Field(default_factory=list)
    primary_role: str | None = None
    seniority: Seniority = "unknown"
    location: NormalizedLocation = Field(default_factory=NormalizedLocation)
    work_type: WorkType = "unknown"
    employment_type: EmploymentType = "unknown"
    experience_range: ExperienceRange = Field(default_factory=ExperienceRange)
    required_skills: list[SkillEvidence] = Field(default_factory=list)
    preferred_skills: list[SkillEvidence] = Field(default_factory=list)
    mentioned_skills: list[SkillEvidence] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    education_level: EducationLevel = "unspecified"
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    salary: SalaryExtract = Field(default_factory=SalaryExtract)
    eligibility_constraints: list[str] = Field(default_factory=list)
    hard_requirements: list[HardRequirement] = Field(default_factory=list)
    field_evidence: list[FieldEvidence] = Field(default_factory=list)
    ambiguous_fields: list[str] = Field(default_factory=list)
    raw_content: str | None = None
    warnings: list[str] = Field(default_factory=list)


class MatchingFeatures(BaseModel):
    source: str | None = None
    source_job_id: str | None = None
    job_url: str | None = None
    content_hash: str | None = None
    role_tags: list[str] = Field(default_factory=list)
    required_skill_names: list[str] = Field(default_factory=list)
    preferred_skill_names: list[str] = Field(default_factory=list)
    seniority: Seniority = "unknown"
    minimum_years_of_experience: float | None = None
    maximum_years_of_experience: float | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None
    work_type: WorkType = "unknown"
    employment_type: EmploymentType = "unknown"
    industries: list[str] = Field(default_factory=list)
    education_level: EducationLevel = "unspecified"
    languages: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    salary_minimum: float | None = None
    salary_maximum: float | None = None
    salary_currency: str | None = None
    salary_period: SalaryPeriod = "unknown"
    required_skill_evidence: list[dict[str, Any]] = Field(default_factory=list)
    preferred_skill_evidence: list[dict[str, Any]] = Field(default_factory=list)
    field_evidence: list[dict[str, Any]] = Field(default_factory=list)
    ambiguous_fields: list[str] = Field(default_factory=list)
    hard_requirements: list[dict[str, Any]] = Field(default_factory=list)
    eligibility_constraints: list[str] = Field(default_factory=list)


class JobState(TypedDict, total=False):
    scraped_job: dict[str, Any]
    extract: dict[str, Any]
    matching_features: dict[str, Any] | None
    warnings: list[str]
    validation_status: Literal["valid", "invalid"]
    validation_errors: list[str]


EXTRACTION_PROMPT = """You are a job posting data extraction agent.

Your task is to convert raw scraped job posting data into structured JSON used by a rule-based matching and scoring service.

Rules:
1. Extract only information supported by the posting.
2. Never invent missing requirements.
3. Use null, unknown, or an empty array when unavailable.
4. Normalize role names, skills, locations, employment types, work arrangements, salary periods, and education levels.
5. Separate required qualifications, preferred qualifications, and merely mentioned technologies.
6. Every inferred value must include confidence from 0.0 to 1.0 and supporting evidence.
7. Do not classify a skill as required unless the posting clearly indicates it.
8. Preserve raw scraped text for audit and debugging.
9. Do not assume a remote role is worldwide.
10. Return valid structured data only.
11. For every non-empty field that can affect matching, add field_evidence with the exact supporting text.
12. Mark values inferred from titles or context as inferred and add them to ambiguous_fields; do not present them as explicit requirements.
13. Preserve every hard requirement and eligibility constraint for downstream review, even if the scorer cannot evaluate it yet.

Normalize common aliases using the same normalization rules as the CV extractor.

Required wording examples:
- must have
- required
- mandatory
- minimum qualification
- at least
- should have
- wajib
- harus
- minimal

Preferred wording examples:
- preferred
- nice to have
- bonus
- advantage
- plus
- ideally
- optional
- diutamakan
- menjadi nilai tambah

When wording is unclear, place the skill under mentioned, not required.

Extract:
- source, source_job_id, job_url, and content hash
- raw and normalized title
- company and industry
- role tags and primary role
- seniority
- normalized location and remote scope
- work type and employment type
- experience range
- required, preferred, and mentioned skills
- responsibilities
- education, certifications, and languages
- salary
- eligibility constraints
- hard requirements
- raw content
- extraction warnings

Add a hard requirement only when failure to meet it would likely make the candidate ineligible. Preferred qualifications must never become hard requirements.
"""


def normalize_skill_name(name: str) -> str:
    key = re.sub(r"\s+", " ", name.strip().lower())
    return SKILL_ALIASES.get(key, name.strip())


def unique_preserve(items: list[str]) -> list[str]:
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


def normalize_skill_list(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for skill in skills:
        item = dict(skill)
        item["name"] = normalize_skill_name(item.get("name") or "")
        if item["name"]:
            result.append(item)
    return result


JOB_MATCHING_EVIDENCE_FIELDS = (
    "role_tags",
    "seniority",
    "work_type",
    "employment_type",
    "experience_range",
    "location",
    "salary",
)


def validate_scraped_job_input(scraped: dict[str, Any]) -> None:
    if not isinstance(scraped, dict):
        raise ValueError("scraped job input must be an object")
    if scraped.get("ok") is False or scraped.get("errors"):
        raise ValueError(f"scrape failed: {scraped.get('errors') or 'unknown error'}")
    if not scraped.get("job"):
        raise ValueError("scraped job input has no job payload")


def mark_unproven_fields(
    extract: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, Any], list[str]]:
    field_evidence = [
        dict(item) for item in extract.get("field_evidence") or [] if item.get("field")
    ]
    evidence_by_field = {item["field"]: item for item in field_evidence}
    ambiguous = unique_preserve(extract.get("ambiguous_fields") or [])

    nested_values = {
        "experience_range": extract.get("experience_range") or {},
        "location": extract.get("location") or {},
        "salary": extract.get("salary") or {},
    }
    for field in JOB_MATCHING_EVIDENCE_FIELDS:
        value = nested_values.get(field, extract.get(field))
        if isinstance(value, dict):
            value_present = any(
                item not in (None, "", "unknown", "unspecified", [], {})
                for key, item in value.items()
                if key not in {"confidence", "evidence"}
            )
        else:
            value_present = value not in (None, "", [], "unknown", "unspecified")
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


def content_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_job_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"/job/(\d+)", url)
    if match:
        return match.group(1)
    return None


def extract_node(state: JobState) -> dict[str, Any]:
    scraped = state.get("scraped_job") or {}
    validate_scraped_job_input(scraped)
    llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    structured = llm.with_structured_output(JobExtract)
    result = structured.invoke(
        [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": f"SCRAPED JOB INPUT:\n{json.dumps(scraped, ensure_ascii=False)}",
            },
        ]
    )
    payload = result.model_dump()
    job = scraped.get("job") or {}
    job_url = payload.get("job_url") or job.get("url") or scraped.get("url")
    payload["job_url"] = job_url
    payload["source"] = payload.get("source") or scraped.get("site")
    payload["source_job_id"] = payload.get("source_job_id") or source_job_id_from_url(
        job_url
    )
    payload["content_hash"] = payload.get("content_hash") or content_hash(scraped)
    if not payload.get("raw_content"):
        payload["raw_content"] = json.dumps(scraped, ensure_ascii=False)
    return {
        "extract": payload,
        "warnings": payload.get("warnings") or [],
    }


def normalize_node(state: JobState) -> dict[str, Any]:
    extract = dict(state.get("extract") or {})
    warnings = list(state.get("warnings") or [])

    extract["required_skills"] = normalize_skill_list(
        extract.get("required_skills") or []
    )
    extract["preferred_skills"] = normalize_skill_list(
        extract.get("preferred_skills") or []
    )
    extract["mentioned_skills"] = normalize_skill_list(
        extract.get("mentioned_skills") or []
    )

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

    work_type = (extract.get("work_type") or "unknown").lower()
    if work_type not in WORK_TYPE_VALUES:
        warnings.append(
            f"Invalid work_type '{extract.get('work_type')}' reset to unknown"
        )
        work_type = "unknown"
    extract["work_type"] = work_type

    employment_type = (extract.get("employment_type") or "unknown").lower()
    if employment_type not in EMPLOYMENT_TYPE_VALUES:
        warnings.append(
            f"Invalid employment_type '{extract.get('employment_type')}' reset to unknown"
        )
        employment_type = "unknown"
    extract["employment_type"] = employment_type

    salary = dict(extract.get("salary") or {})
    period = salary.get("period")
    if period is not None and period not in PERIOD_VALUES:
        warnings.append(f"Invalid salary period '{period}' reset to unknown")
        salary["period"] = "unknown"
    extract["salary"] = salary

    extract["role_tags"] = unique_preserve(extract.get("role_tags") or [])
    extract["responsibilities"] = unique_preserve(extract.get("responsibilities") or [])
    extract["languages"] = unique_preserve(extract.get("languages") or [])
    extract["certifications"] = unique_preserve(extract.get("certifications") or [])
    extract["eligibility_constraints"] = unique_preserve(
        extract.get("eligibility_constraints") or []
    )
    extract, warnings = mark_unproven_fields(extract, warnings)

    return {"extract": extract, "warnings": warnings}


def build_matching_features_node(state: JobState) -> dict[str, Any]:
    extract = state.get("extract") or {}
    location = extract.get("location") or {}
    experience = extract.get("experience_range") or {}
    salary = extract.get("salary") or {}
    industry = extract.get("industry")
    industries = unique_preserve([industry] if industry else [])
    required_skills = normalize_skill_list(extract.get("required_skills") or [])
    preferred_skills = normalize_skill_list(extract.get("preferred_skills") or [])
    required_skill_names = unique_preserve([skill["name"] for skill in required_skills])
    preferred_skill_names = unique_preserve(
        [skill["name"] for skill in preferred_skills]
    )

    features = MatchingFeatures(
        source=extract.get("source"),
        source_job_id=extract.get("source_job_id"),
        job_url=extract.get("job_url"),
        content_hash=extract.get("content_hash"),
        role_tags=unique_preserve(extract.get("role_tags") or []),
        required_skill_names=required_skill_names,
        preferred_skill_names=preferred_skill_names,
        seniority=extract.get("seniority") or "unknown",
        minimum_years_of_experience=experience.get("minimum_years"),
        maximum_years_of_experience=experience.get("maximum_years"),
        city=location.get("city"),
        region=location.get("region"),
        country=location.get("country"),
        work_type=extract.get("work_type") or "unknown",
        employment_type=extract.get("employment_type") or "unknown",
        industries=industries,
        education_level=extract.get("education_level") or "unspecified",
        languages=unique_preserve(extract.get("languages") or []),
        certifications=unique_preserve(extract.get("certifications") or []),
        salary_minimum=salary.get("minimum"),
        salary_maximum=salary.get("maximum"),
        salary_currency=salary.get("currency"),
        salary_period=salary.get("period") or "unknown",
        required_skill_evidence=required_skills,
        preferred_skill_evidence=preferred_skills,
        field_evidence=[dict(item) for item in extract.get("field_evidence") or []],
        ambiguous_fields=unique_preserve(extract.get("ambiguous_fields") or []),
        hard_requirements=[
            dict(item) for item in extract.get("hard_requirements") or []
        ],
        eligibility_constraints=unique_preserve(
            extract.get("eligibility_constraints") or []
        ),
    )
    return {"matching_features": features.model_dump()}


def validate_node(state: JobState) -> dict[str, Any]:
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


builder = StateGraph(JobState)
builder.add_node("extract", extract_node)
builder.add_node("normalize", normalize_node)
builder.add_node("build_matching_features", build_matching_features_node)
builder.add_node("validate", validate_node)

builder.add_edge(START, "extract")
builder.add_edge("extract", "normalize")
builder.add_edge("normalize", "build_matching_features")
builder.add_edge("build_matching_features", "validate")
builder.add_edge("validate", END)

graph = builder.compile()
