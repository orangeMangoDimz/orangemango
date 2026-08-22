"""Structured output schemas for the chatbot graph's language-model calls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config.const.chatbot import (
    MAX_CV_DOCUMENTS,
)
from app.models.chatbot.literals import (
    CvAction,
    CvTargetScope,
    GoalName,
    JobAction,
    JobResponse,
    JobSource,
    JobTargetScope,
    JobTask,
    ParentTarget,
    ReviewMode,
    RoleSource,
    RouteName,
)


class ScrapeRequest(BaseModel):
    keywords: list[str] = Field(default_factory=list, max_length=5)
    sites: list[str] = Field(default_factory=list, max_length=5)
    max_age_hours: int | None = Field(default=None, ge=1, le=720)


class PlanStep(BaseModel):
    node: str = Field(min_length=1, max_length=80)
    expected: str = Field(min_length=1, max_length=300)
    reason: str = Field(min_length=1, max_length=300)


class ParentPlanStep(PlanStep):
    node: ParentTarget


class CvPlanStep(PlanStep):
    node: CvAction


class JobPlanStep(PlanStep):
    node: JobAction


class ParentIntent(BaseModel):
    query: str = Field(min_length=1, max_length=12000)
    goal: str = Field(min_length=1, max_length=300)
    cv_ids: list[str] = Field(default_factory=list, max_length=MAX_CV_DOCUMENTS)
    job_ids: list[str] = Field(default_factory=list, max_length=20)


class ParentPlan(BaseModel):
    """End-to-end high-level plan owned by the parent graph."""

    intent: ParentIntent
    plan: list[ParentPlanStep] = Field(min_length=1, max_length=3)


class CvPlan(BaseModel):
    """End-to-end CV-domain plan selected inside the CV subagent."""

    plan: list[CvPlanStep] = Field(min_length=1, max_length=3)
    cv_ids: list[str] = Field(default_factory=list, max_length=MAX_CV_DOCUMENTS)
    review_mode: ReviewMode = "general"
    review_focus: str | None = Field(default=None, max_length=200)
    target_role: str | None = Field(default=None, max_length=160)


class JobPlan(BaseModel):
    """End-to-end job-domain plan selected inside the job subagent."""

    plan: list[JobPlanStep] = Field(min_length=1, max_length=3)
    source: JobSource = "none"
    response: JobResponse = "none"
    refresh: bool = False
    cv_ids: list[str] = Field(default_factory=list, max_length=MAX_CV_DOCUMENTS)
    job_ids: list[str] = Field(default_factory=list, max_length=20)
    search: ScrapeRequest = Field(default_factory=ScrapeRequest)
    pasted_content: str | None = None
    show_score: bool = False


class RouteDecision(BaseModel):
    route: RouteName
    reason: str = Field(min_length=1, max_length=300)
    job_task: JobTask = Field(
        default="none",
        description="Mapped task intent for job behavior: search, match, extract, or cancel.",
    )
    job_response: JobResponse = Field(
        default="none",
        description=(
            "Mapped response shape for this request: list, summary, recommendation, "
            "explanation, or details."
        ),
    )
    job_refresh: bool = Field(
        default=False,
        description=(
            "True when the user is refreshing the current goal/search state. "
            "This applies only to search task behavior."
        ),
    )
    job_source: JobSource = Field(
        default="none",
        description=(
            "Where the job input comes from: existing loaded jobs, a web search, "
            "or a job description pasted in the latest user message."
        ),
    )
    score_requested: bool = Field(
        default=False,
        description=(
            "True only when the user explicitly requests a numeric match score, "
            "rating, grade, or percentage."
        ),
    )
    assessment_requested: bool = Field(
        default=False,
        description=(
            "True when the user wants CV-to-job fit assessed. This is independent "
            "from requesting a numeric score."
        ),
    )
    role_constraints: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Explicit role, technology, or location constraints for a new search. "
            "Empty when reusing the active goal."
        ),
    )
    role_evidence: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "Evidence supporting the selected role constraints. For explicit roles "
            "this is a quote from the latest message; for CV-derived roles this "
            "describes the structured CV profile evidence."
        ),
    )
    role_source: RoleSource = Field(
        default="none",
        description=(
            "Whether the role came from the latest message, an active goal, or "
            "inference from an extracted CV profile."
        ),
    )
    role_candidates: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=5,
        description="Candidate roles inferred from the structured CV profile.",
    )
    job_target_scope: JobTargetScope = Field(
        default="none",
        description=(
            "one for one catalog job, all for every current job or match, and none "
            "when no catalog target is involved."
        ),
    )
    decision_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence that the structured job intent is unambiguous.",
    )
    review_target_role: str | None = Field(
        default=None,
        max_length=160,
        description=(
            "The explicit target role for a CV-quality review, or null when the "
            "user did not provide one."
        ),
    )
    review_mode: ReviewMode = Field(
        default="general",
        description=(
            "For CV reviews: scored only when the user explicitly asks for a "
            "numerical score, rating, or grade; focused when they name a section "
            "or aspect; otherwise general."
        ),
    )
    review_focus: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "A concise description of the CV section or aspect the user wants "
            "reviewed, or null for a broad review."
        ),
    )
    review_mode_reason: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "A concise explanation of why this review mode fits the user's "
            "request, or null when the route is not review_cv."
        ),
    )
    needs_cv_text: bool = Field(
        default=False,
        description=(
            "True when answering requires the raw uploaded CV text "
            "(ambiguity, wording, section review, quote experience)."
        ),
    )
    needs_cv_features: bool = Field(
        default=False,
        description=(
            "True when answering depends on extracted CV profile features "
            "(role or job-type recommendations from uploaded CVs, CV Q&A)."
        ),
    )
    is_follow_up: bool = Field(
        default=False,
        description=(
            "True when the latest message continues, clarifies, or reformats an "
            "already-available result (existing CV review, comparison, jobs, or "
            "matches) without requesting a new tool action."
        ),
    )
    selected_cv_id: str | None = Field(
        default=None,
        max_length=80,
        description=(
            "For review_cv: the id from the cvs catalog for the CV the user "
            "named, or null to use the first extracted CV. For match_jobs: the "
            "id of one CV when named, or null to match every extracted CV."
        ),
    )
    selected_job_keys: list[str] | None = Field(
        default=None,
        max_length=20,
        description=(
            "Exact keys from the jobs or matches catalog when the user identifies "
            "specific jobs. Never invent keys."
        ),
    )
    scrape_request: ScrapeRequest = Field(default_factory=ScrapeRequest)


class RoleCandidate(BaseModel):
    role: str = Field(min_length=1, max_length=120)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = Field(min_length=1, max_length=300)


class JobRequestDecision(BaseModel):
    task: JobTask = Field(
        default="none",
        description=(
            "Requested job operation: search finds jobs, match assesses jobs against "
            "CVs, extract reads a pasted job, and cancel stops the active job goal."
        ),
    )
    source: JobSource = Field(
        default="none",
        description=(
            "Job data source: search for a new search, existing for retained jobs or "
            "matches, and pasted for job text in the latest message."
        ),
    )
    response: JobResponse = Field(
        default="none",
        description=(
            "Requested result scope: list for search results, summary for overall fit, "
            "recommendation for one best fit, and explanation or details for one "
            "selected job."
        ),
    )
    refresh: bool = Field(
        default=False,
        description=(
            "True only when the user explicitly asks to refresh or rerun an existing "
            "job goal; false for an initial search."
        ),
    )
    input: str | None = Field(
        default=None,
        description="Pasted job text when task is extract; otherwise null.",
    )
    scrape: ScrapeRequest = Field(
        default_factory=ScrapeRequest,
        description="Search criteria supplied or inferred for a search task.",
    )


class RequestDecision(BaseModel):
    """Combined request intent and catalog-target resolution output."""

    goal: GoalName = Field(
        description="The current user goal. Use job for every job-related request."
    )
    reason: str = Field(min_length=1, max_length=300)
    decision_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    job: JobRequestDecision = Field(default_factory=JobRequestDecision)
    score_requested: bool = False
    assessment_requested: bool = False
    role_constraints: list[str] = Field(default_factory=list, max_length=5)
    role_evidence: str | None = Field(default=None, max_length=300)
    role_source: RoleSource = "none"
    role_candidates: list[RoleCandidate] = Field(default_factory=list, max_length=5)
    review_target_role: str | None = Field(default=None, max_length=160)
    review_mode: ReviewMode = "general"
    review_focus: str | None = Field(default=None, max_length=200)
    review_mode_reason: str | None = Field(default=None, max_length=300)
    needs_cv_text: bool = False
    needs_cv_features: bool = False
    is_follow_up: bool = False
    cv_target_scope: CvTargetScope = "none"
    selected_cv_ids: list[str] = Field(
        default_factory=list,
        max_length=MAX_CV_DOCUMENTS,
    )
    job_target_scope: JobTargetScope = "none"
    selected_job_keys: list[str] | None = Field(default=None, max_length=20)
    unresolved_references: list[str] = Field(default_factory=list, max_length=8)
    targets_ambiguous: bool = False


class WorkflowPlan(BaseModel):
    """Stage 3 output: exactly one next workflow action."""

    action: RouteName = Field(
        description=(
            "Exactly one next backend action. Select only a missing executable step; "
            "use respond when prerequisites are missing, targets are ambiguous, or "
            "reusable state already satisfies the request."
        )
    )
    reason: str = Field(min_length=1, max_length=300)


class CvComparisonCandidate(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    strengths: list[str] = Field(min_length=1, max_length=5)
    weaknesses: list[str] = Field(min_length=1, max_length=5)
    summary: str = Field(min_length=1, max_length=500)


class CvComparisonResult(BaseModel):
    overview: str = Field(min_length=1, max_length=800)
    candidates: list[CvComparisonCandidate] = Field(min_length=2, max_length=5)
    skill_matrix: list[str] = Field(default_factory=list, max_length=12)
    recommendation: str = Field(min_length=1, max_length=800)
