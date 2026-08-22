from __future__ import annotations

"""Conversational CV and job-search graph for LangGraph Studio.

Accepts one or more PDFs via Studio chat file blocks on the user message, or via
graph-mode pending_cv_upload / pending_cv_uploads::

    {
        "messages": [
            {"role": "user", "content": "Compare these CVs"}
        ],
        "pending_cv_uploads": [
            {
                "filename": "cv_a.pdf",
                "content_base64": "<base64-encoded PDF bytes>"
            },
            {
                "filename": "cv_b.pdf",
                "content_base64": "<base64-encoded PDF bytes>"
            }
        ]
    }

File blocks are removed from persisted messages so Studio chat does not hit
the empty file-data render error. PDF bytes are decoded in memory only.
"""

import asyncio
import base64
import binascii
import hashlib
import importlib.util
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any, Annotated, Literal, TypedDict, cast

from app.models.chat_model import ChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.config.const.chat import MAX_CV_FILE_BYTES
from app.services.cv_document import extract_pdf_text, validate_pdf_upload
from app.services.text_normalization import casefolded_text
from studio.chatbot import match_presentation


def _load_graph(path: Path, module_name: str) -> ModuleType:
    spec: ModuleSpec | None = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load child graph module from {path}")

    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "graph"):
        raise ImportError(f"Child graph module does not export graph: {path}")

    return module


STUDIO_ROOT: Path = Path(__file__).resolve().parents[1]
CV_GRAPH_PATH: Path = STUDIO_ROOT / "cv-extraction" / "graph.py"
JOB_GRAPH_PATH: Path = STUDIO_ROOT / "job-extraction" / "graph.py"
MATCHING_SCORE_GRAPH_PATH: Path = STUDIO_ROOT / "matching-score" / "graph.py"
CV_REVIEW_GRAPH_PATH: Path = STUDIO_ROOT / "cv-review" / "graph.py"

cv_module: ModuleType = _load_graph(CV_GRAPH_PATH, "orangemango_chatbot_cv_extraction")
job_module: ModuleType = _load_graph(
    JOB_GRAPH_PATH, "orangemango_chatbot_job_extraction"
)
matching_score_module: ModuleType = _load_graph(
    MATCHING_SCORE_GRAPH_PATH,
    "orangemango_chatbot_matching_score",
)
cv_review_module: ModuleType = _load_graph(
    CV_REVIEW_GRAPH_PATH, "orangemango_chatbot_cv_review"
)

cv_extraction_graph: Any = cv_module.graph
job_extraction_graph: Any = job_module.graph
matching_score_graph: Any = matching_score_module.graph


JOB_SUBAGENT_MCP_URL: str = "http://localhost:8080/mcp"
JOB_SUBAGENT_MCP_CLIENT: MultiServerMCPClient = MultiServerMCPClient(
    {"job_scraper": {"transport": "http", "url": JOB_SUBAGENT_MCP_URL}}
)
_JOB_SUBAGENT_SCRAPE_TOOL: Any | None = None

MAX_PDF_BYTES: int = MAX_CV_FILE_BYTES
MAX_FIELD_CHARS: int = 800
MAX_REQUIREMENTS: int = 12
MAX_REQUIREMENT_CHARS: int = 360
MAX_ROUTER_CHARS: int = 12000
MAX_CONTEXT_MESSAGES: int = 8
MAX_ROUTER_HISTORY_MESSAGES: int = 6
MAX_ROUTER_HISTORY_CHARS: int = 400
DEFAULT_CONTEXT_WINDOW_TOKENS: int = 32768
DEFAULT_CONTEXT_OUTPUT_RESERVE_TOKENS: int = 4096
DEFAULT_CONTEXT_PROMPT_RESERVE_TOKENS: int = 8192
CONTEXT_SUMMARY_TRIGGER_RATIO: float = 0.75
MAX_CONVERSATION_SUMMARY_CHARS: int = 2400
MAX_CONVERSATION_MEMORY_ITEMS: int = 8


def _positive_int_env(name: str, default: int) -> int:
    value: str = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed: int = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def context_input_budget() -> int:
    context_window: int = _positive_int_env(
        "OPENAI_CONTEXT_WINDOW_TOKENS",
        DEFAULT_CONTEXT_WINDOW_TOKENS,
    )
    output_reserve: int = _positive_int_env(
        "OPENAI_CONTEXT_OUTPUT_RESERVE_TOKENS",
        DEFAULT_CONTEXT_OUTPUT_RESERVE_TOKENS,
    )
    prompt_reserve: int = _positive_int_env(
        "OPENAI_CONTEXT_PROMPT_RESERVE_TOKENS",
        DEFAULT_CONTEXT_PROMPT_RESERVE_TOKENS,
    )
    return max(1024, context_window - output_reserve - prompt_reserve)


async def load_scrape_jobs_tool() -> Any:
    global _JOB_SUBAGENT_SCRAPE_TOOL
    if _JOB_SUBAGENT_SCRAPE_TOOL is None:
        tools: list[Any] = await JOB_SUBAGENT_MCP_CLIENT.get_tools()
        tools_by_name: dict[str, Any] = {tool.name: tool for tool in tools}
        if "scrape_jobs" not in tools_by_name:
            raise RuntimeError(
                "The MCP server did not expose scrape_jobs. "
                f"Available tools: {sorted(tools_by_name)}"
            )
        _JOB_SUBAGENT_SCRAPE_TOOL = tools_by_name["scrape_jobs"]
    return _JOB_SUBAGENT_SCRAPE_TOOL


def _decode_upload_content(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Uploaded CV content must be base64-encoded PDF bytes")

    encoded: str = value.strip()
    if encoded.startswith("data:"):
        try:
            encoded = encoded.split(",", 1)[1]
        except IndexError as exc:
            raise ValueError("Uploaded CV data URL is malformed") from exc
    encoded = "".join(encoded.split())
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Uploaded CV content is not valid base64") from exc


def _block_as_dict(block: Any) -> dict[str, Any] | None:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        dumped: Any = block.model_dump()
        return dumped if isinstance(dumped, dict) else None
    if hasattr(block, "dict"):
        dumped: Any = block.dict()
        return dumped if isinstance(dumped, dict) else None
    return None


def _is_file_block(block: dict[str, Any]) -> bool:
    block_type: str = str(block.get("type") or "").casefold()
    if block_type == "file":
        return True
    mime: str = str(block.get("mime_type") or block.get("mimeType") or "").casefold()
    return mime == "application/pdf"


def _nested_file_dict(block: dict[str, Any]) -> dict[str, Any] | None:
    nested: Any = block.get("file")
    return nested if isinstance(nested, dict) else None


def _file_block_filename(block: dict[str, Any]) -> str:
    nested: dict[str, Any] = _nested_file_dict(block) or {}
    metadata: Any = block.get("metadata")
    metadata_name: str = ""
    if isinstance(metadata, dict):
        metadata_name = str(
            metadata.get("filename")
            or metadata.get("name")
            or metadata.get("title")
            or ""
        )
    extras: Any = block.get("extras")
    if not metadata_name and isinstance(extras, dict):
        nested_meta: Any = extras.get("metadata")
        if isinstance(nested_meta, dict):
            metadata_name = str(
                nested_meta.get("filename")
                or nested_meta.get("name")
                or nested_meta.get("title")
                or ""
            )
    return str(
        block.get("filename")
        or block.get("name")
        or nested.get("filename")
        or nested.get("name")
        or metadata_name
        or "cv.pdf"
    )


def _payload_from_mapping(mapping: dict[str, Any]) -> Any:
    for key in (
        "file_data",
        "content_base64",
        "data",
        "base64",
        "content",
        "url",
    ):
        value: Any = mapping.get(key)
        if value not in (None, ""):
            return value
    source: Any = mapping.get("source")
    if isinstance(source, dict):
        return _payload_from_mapping(source)
    return None


def _file_block_payload(block: dict[str, Any]) -> Any:
    payload: Any = _payload_from_mapping(block)
    if payload not in (None, ""):
        return payload
    nested: dict[str, Any] | None = _nested_file_dict(block)
    if nested is not None:
        return _payload_from_mapping(nested)
    return None


def upload_from_file_block(block: Any) -> dict[str, Any] | None:
    parsed: dict[str, Any] | None = _block_as_dict(block)
    if parsed is None or not _is_file_block(parsed):
        return None
    payload: Any = _file_block_payload(parsed)
    if payload in (None, ""):
        return None
    filename: str = _file_block_filename(parsed)
    safe_name: str = Path(str(filename).replace("\\", "/")).name or "cv.pdf"
    if Path(safe_name).suffix.casefold() != ".pdf":
        safe_name = f"{safe_name}.pdf" if safe_name else "cv.pdf"
    return {"filename": safe_name, "content_base64": payload}


def message_content_blocks(message: Any) -> list[Any]:
    content: Any = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    )
    if isinstance(content, list):
        return content
    return []


def extract_uploads_from_message(message: Any) -> list[dict[str, Any]]:
    uploads: list[dict[str, Any]] = []
    for block in message_content_blocks(message):
        upload: dict[str, Any] | None = upload_from_file_block(block)
        if upload is not None:
            uploads.append(upload)
    return uploads


def extract_upload_from_message(message: Any) -> dict[str, Any] | None:
    uploads: list[dict[str, Any]] = extract_uploads_from_message(message)
    return uploads[0] if uploads else None


def read_uploaded_cv(uploaded_file: Any) -> str:
    if not isinstance(uploaded_file, dict):
        raise ValueError(
            "pending_cv_upload must contain filename and base64 PDF content"
        )

    filename: str = str(
        uploaded_file.get("filename") or uploaded_file.get("name") or ""
    )
    safe_name: str = Path(filename.replace("\\", "/")).name
    if (
        not filename
        or safe_name != filename
        or Path(safe_name).suffix.casefold() != ".pdf"
    ):
        raise ValueError("Only .pdf CV uploads are supported")

    content: Any = uploaded_file.get("content_base64")
    if content is None:
        content = uploaded_file.get("content")
    if content is None:
        content = uploaded_file.get("data")
    if content is None:
        content = uploaded_file.get("base64")
    payload: bytes = _decode_upload_content(content)
    validate_pdf_upload(
        filename=filename,
        content_type="application/pdf",
        content=payload,
    )
    return extract_pdf_text(payload)


def cv_document_from_upload(uploaded_file: Any) -> dict[str, Any]:
    filename: str = str(
        uploaded_file.get("filename") or uploaded_file.get("name") or ""
    )
    safe_name: str = Path(filename.replace("\\", "/")).name or "cv.pdf"
    return {
        "id": str(uuid.uuid4()),
        "filename": safe_name,
        "cv_text": read_uploaded_cv(uploaded_file),
        "cv_result": None,
        "cv_features": None,
        "cv_review": None,
    }


def with_message_content(message: Any, content: str | list[dict[str, Any]]) -> Any:
    if isinstance(message, dict):
        updated: dict[str, Any] = {**message, "content": content}
        additional: dict[str, Any] = dict(updated.get("additional_kwargs") or {})
        additional.pop("pending_cv_upload", None)
        additional.pop("pending_cv_uploads", None)
        if additional:
            updated["additional_kwargs"] = additional
        elif "additional_kwargs" in updated:
            updated = {
                key: value
                for key, value in updated.items()
                if key != "additional_kwargs"
            }
        return updated

    additional: dict[str, Any] = dict(getattr(message, "additional_kwargs", None) or {})
    additional.pop("pending_cv_upload", None)
    additional.pop("pending_cv_uploads", None)
    if hasattr(message, "model_copy"):
        return message.model_copy(
            update={"content": content, "additional_kwargs": additional}
        )

    updated: dict[str, Any] = {"role": "user", "content": content}
    message_id: Any = getattr(message, "id", None)
    if message_id:
        updated["id"] = message_id
    if additional:
        updated["additional_kwargs"] = additional
    return updated


def sanitize_file_message(
    message: Any,
    *,
    stash_upload: bool = False,
) -> Any | None:
    content: list[Any] = message_content_blocks(message)
    if not content:
        return None

    file_blocks: list[Any] = [
        block
        for block in content
        if (parsed := _block_as_dict(block)) is not None and _is_file_block(parsed)
    ]
    if not file_blocks:
        return None

    text_parts: list[str] = [
        str(parsed.get("text")).strip()
        for block in content
        if (parsed := _block_as_dict(block)) is not None
        and parsed.get("type") == "text"
        and str(parsed.get("text") or "").strip()
    ]
    text_parts.append(PDF_UPLOAD_MARKER)
    sanitized: Any = with_message_content(message, "\n".join(text_parts))

    if not stash_upload:
        return sanitized

    uploads: list[dict[str, Any]] = extract_uploads_from_message(message)
    if not uploads:
        first_file: dict[str, Any] = _block_as_dict(file_blocks[0]) or {}
        uploads = [
            {
                "filename": _file_block_filename(first_file),
                "content_base64": "",
                "missing_bytes": True,
            }
        ]

    if isinstance(sanitized, dict):
        additional: dict[str, Any] = dict(sanitized.get("additional_kwargs") or {})
        additional["pending_cv_uploads"] = uploads
        additional["pending_cv_upload"] = uploads[0]
        return {**sanitized, "additional_kwargs": additional}

    additional: dict[str, Any] = dict(
        getattr(sanitized, "additional_kwargs", None) or {}
    )
    additional["pending_cv_uploads"] = uploads
    additional["pending_cv_upload"] = uploads[0]
    if hasattr(sanitized, "model_copy"):
        return sanitized.model_copy(update={"additional_kwargs": additional})
    return sanitized


def add_chat_messages(
    left: list[Any] | None,
    right: list[Any] | None,
) -> list[Any]:
    sanitized_right: list[Any] = []
    for message in right or []:
        sanitized: Any | None = sanitize_file_message(message, stash_upload=True)
        sanitized_right.append(message if sanitized is None else sanitized)
    return add_messages(left or [], sanitized_right)


RouteName = Literal[
    "respond",
    "extract_cv",
    "review_cv",
    "compare_cvs",
    "extract_job",
    "search_jobs",
    "match_jobs",
]
AgentAction = Literal[
    "extract_cv",
    "review_cv",
    "compare_cvs",
    "extract_job",
    "search_jobs",
    "match_jobs",
]
JobSource = Literal["none", "existing", "search", "pasted"]
RoleSource = Literal["none", "explicit", "active_goal", "cv_inferred"]
ReviewMode = Literal["general", "scored", "focused"]
JobTask = Literal["none", "search", "match", "extract", "cancel"]
JobResponse = Literal[
    "none",
    "list",
    "summary",
    "recommendation",
    "explanation",
    "details",
]
JobTargetScope = Literal["none", "one", "all"]
CvTargetScope = Literal["none", "one", "all"]
GoalName = Literal[
    "review_cv",
    "compare_cvs",
    "job",
    "general_question",
    "extract_cv",
]

AGENT_ACTIONS: frozenset[str] = frozenset(
    {
        "extract_cv",
        "review_cv",
        "compare_cvs",
        "extract_job",
        "search_jobs",
        "match_jobs",
    }
)
USER_FACING_ACTIONS: frozenset[str] = frozenset(
    {
        "review_cv",
        "compare_cvs",
        "extract_job",
        "search_jobs",
        "match_jobs",
    }
)
EXECUTION_DESTINATIONS: dict[str, tuple[str, str]] = {
    "extract_cv": ("cv_subagent", "extract_cv"),
    "review_cv": ("cv_subagent", "review_cv"),
    "compare_cvs": ("cv_subagent", "compare_cvs"),
    "extract_job": ("job_subagent", "extract_pasted_job"),
    "search_jobs": ("job_subagent", "scrape_jobs"),
    "match_jobs": ("job_subagent", "match_jobs"),
}
EXECUTION_NODE_USES: dict[tuple[str, str], list[str]] = {
    ("chatbot", "validate_plan"): [
        "routing.request",
        "routing.targets",
        "routing.plan",
        "jobs.scrape_request",
        "action_results",
    ],
    ("cv_subagent", "extract_cv"): [
        "cv.documents",
        "routing.plan.planned_stages",
    ],
    ("cv_subagent", "review_cv"): [
        "cv.documents",
        "routing.targets.cv",
        "routing.request.review",
    ],
    ("cv_subagent", "compare_cvs"): [
        "cv.documents",
        "routing.targets.cv",
    ],
    ("job_subagent", "scrape_jobs"): [
        "jobs.scrape_request",
        "jobs.results",
        "routing.plan.planned_stages",
    ],
    ("job_subagent", "extract_pasted_job"): [
        "routing.request.job.input",
        "jobs.results",
    ],
    ("job_subagent", "match_jobs"): [
        "cv.documents",
        "jobs.results",
        "routing.targets",
    ],
}
CV_FEATURE_INTENTS: frozenset[str] = frozenset(
    {
        "extract_cv",
        "review_cv",
        "compare_cvs",
        "match_jobs",
    }
)
MAX_AGENT_ACTIONS: int = 4
MAX_CV_DOCUMENTS: int = 5
MAX_ACTION_RESULT_CHARS: int = 6000
SEARCH_RESULT_TTL_SECONDS: int = 24 * 60 * 60
MIN_JOB_INTENT_CONFIDENCE: float = 0.75
PDF_UPLOAD_MARKER: str = "[PDF CV uploaded separately]"


class ScrapeRequest(BaseModel):
    keywords: list[str] = Field(default_factory=list, max_length=5)
    sites: list[str] = Field(default_factory=list, max_length=5)
    max_age_hours: int | None = Field(default=None, ge=1, le=720)


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


class ConversationMemory(BaseModel):
    """Durable conversational context; domain state remains authoritative."""

    summary: str = Field(default="", max_length=MAX_CONVERSATION_SUMMARY_CHARS)
    user_preferences: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONVERSATION_MEMORY_ITEMS,
    )
    decisions: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONVERSATION_MEMORY_ITEMS,
    )
    open_questions: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONVERSATION_MEMORY_ITEMS,
    )
    references: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONVERSATION_MEMORY_ITEMS,
    )


def merge_maps(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    return {**(left or {}), **(right or {})}


class RoutingGoalState(TypedDict, total=False):
    name: GoalName
    reason: str
    confidence: float


class RoutingJobState(TypedDict, total=False):
    task: JobTask
    response: JobResponse
    source: JobSource
    input: str | None
    refresh: bool
    scrape: dict[str, Any]


class RoutingRoleState(TypedDict, total=False):
    constraints: list[str]
    evidence: str | None
    source: RoleSource
    candidates: list[dict[str, Any]]


class RoutingAssessmentState(TypedDict, total=False):
    requested: bool
    detail_level: str


class RoutingScoreState(TypedDict, total=False):
    requested: bool
    visible: bool


class RoutingReviewState(TypedDict, total=False):
    target_role: str | None
    mode: ReviewMode
    focus: str | None
    reason: str | None


class RoutingCvState(TypedDict, total=False):
    text_needed: bool
    features_needed: bool


class RoutingContextState(TypedDict, total=False):
    follow_up: bool


class RoutingRequestState(TypedDict, total=False):
    goal: RoutingGoalState
    job: RoutingJobState
    role: RoutingRoleState
    assessment: RoutingAssessmentState
    score: RoutingScoreState
    review: RoutingReviewState
    cv: RoutingCvState
    context: RoutingContextState


class RoutingCvTargetsState(TypedDict, total=False):
    scope: CvTargetScope
    ids: list[str]


class RoutingJobTargetsState(TypedDict, total=False):
    scope: JobTargetScope
    keys: list[str]


class RoutingTargetsState(TypedDict, total=False):
    cv: RoutingCvTargetsState
    job: RoutingJobTargetsState
    unresolved_references: list[str]
    ambiguous: bool


class RoutingPlanState(TypedDict, total=False):
    action: RouteName
    reason: str
    validation: str
    validation_error: str | None
    planned_stages: list[str]
    policy_reason: str
    active_goal_id: str | None
    completed_actions: list[str]


class RoutingState(TypedDict, total=False):
    request: RoutingRequestState
    targets: RoutingTargetsState
    plan: RoutingPlanState


ExecutionStatus = Literal["completed", "failed", "skipped"]


class ExecutionNoteState(TypedDict):
    summary: str
    reason: str


class ExecutionFromNodeState(TypedDict):
    graph: str
    node: str
    uses: list[str]
    note: ExecutionNoteState
    output: dict[str, Any]


class ExecutionToNodeState(TypedDict):
    graph: str
    node: str
    uses: list[str]
    note: ExecutionNoteState
    args: dict[str, Any]
    result: dict[str, Any]


class ExecutionContextState(TypedDict, total=False):
    args_source: RoleSource
    source_reference: dict[str, Any]


ExecutionStepState = TypedDict(
    "ExecutionStepState",
    {
        "index": int,
        "action": AgentAction,
        "status": ExecutionStatus,
        "from": ExecutionFromNodeState,
        "to": ExecutionToNodeState,
        "context": ExecutionContextState,
        "error": str | None,
    },
)


class ExecutionState(TypedDict):
    steps: list[ExecutionStepState]


def merge_routing_maps(
    left: RoutingState | None,
    right: RoutingState | None,
) -> RoutingState:
    merged: dict[str, Any] = dict(left or {})
    for section, update in (right or {}).items():
        if isinstance(update, dict) and isinstance(merged.get(section), dict):
            merged[section] = {**merged[section], **update}
        else:
            merged[section] = update
    return merged  # type: ignore[return-value]


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_chat_messages]
    pending_cv_upload: dict[str, Any] | None
    pending_cv_uploads: list[dict[str, Any]] | None
    input_error: bool
    cv: Annotated[dict[str, Any], merge_maps]
    routing: Annotated[RoutingState, merge_routing_maps]
    # Transitional top-level target/plan buckets are read-only fallbacks.
    targets: dict[str, Any]
    plan: dict[str, Any]
    # Legacy non-request routing buckets are retained for old helpers.
    router: Annotated[dict[str, Any], merge_maps]
    selection: Annotated[dict[str, Any], merge_maps]
    jobs: Annotated[dict[str, Any], merge_maps]
    action_results: Annotated[dict[str, Any], merge_maps]
    execution: Annotated[ExecutionState, merge_maps]
    conversation_memory: Annotated[dict[str, Any], merge_maps]
    conversation_memory_cursor: int
    response: str | None
    job_list: list[dict[str, Any]]
    errors: list[str]


class StudioInput(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_chat_messages]
    pending_cv_upload: dict[str, Any] | None
    pending_cv_uploads: list[dict[str, Any]] | None


def short_text(value: Any, limit: int = MAX_FIELD_CHARS) -> str:
    text: str = "" if value is None else str(value).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def cv_bucket(state: ConversationState) -> dict[str, Any]:
    value = state.get("cv")
    return dict(value) if isinstance(value, dict) else {}


def routing_bucket(state: ConversationState) -> dict[str, Any]:
    value = state.get("routing")
    return dict(value) if isinstance(value, dict) else {}


def request_bucket(state: ConversationState) -> dict[str, Any]:
    value = routing_bucket(state).get("request")
    return dict(value) if isinstance(value, dict) else {}


REQUEST_VALUE_KEYS: frozenset[str] = frozenset(
    {
        "goal",
        "goal_reason",
        "decision_confidence",
        "job_task",
        "job_response",
        "job_refresh",
        "job_source",
        "job_input_text",
        "score_requested",
        "assessment_requested",
        "show_score",
        "refresh_requested",
        "match_detail_level",
        "role_constraints",
        "role_evidence",
        "role_source",
        "role_candidates",
        "review_target_role",
        "review_mode",
        "review_focus",
        "review_mode_reason",
        "needs_cv_text",
        "needs_cv_features",
        "is_follow_up",
        "scrape_request",
    }
)


def request_values(request: dict[str, Any]) -> dict[str, Any]:
    """Project nested request state into a private runtime view."""
    goal: dict[str, Any] = request.get("goal") or {}
    job: dict[str, Any] = request.get("job") or {}
    role: dict[str, Any] = request.get("role") or {}
    assessment: dict[str, Any] = request.get("assessment") or {}
    score: dict[str, Any] = request.get("score") or {}
    review: dict[str, Any] = request.get("review") or {}
    cv: dict[str, Any] = request.get("cv") or {}
    context: dict[str, Any] = request.get("context") or {}
    return {
        "goal": goal.get("name") or "general_question",
        "goal_reason": goal.get("reason") or "",
        "decision_confidence": goal.get("confidence", 1.0),
        "job_source": job.get("source") or "none",
        "job_task": job.get("task") or "none",
        "job_response": job.get("response") or "none",
        "job_refresh": bool(job.get("refresh")),
        "job_input_text": job.get("input"),
        "score_requested": bool(score.get("requested")),
        "assessment_requested": bool(assessment.get("requested")),
        "show_score": bool(score.get("visible")),
        "refresh_requested": bool(job.get("refresh")),
        "match_detail_level": assessment.get("detail_level") or "summary",
        "role_constraints": list(role.get("constraints") or []),
        "role_evidence": role.get("evidence"),
        "role_source": role.get("source") or "none",
        "role_candidates": list(role.get("candidates") or []),
        "review_target_role": review.get("target_role"),
        "review_mode": review.get("mode") or "general",
        "review_focus": review.get("focus"),
        "review_mode_reason": review.get("reason"),
        "needs_cv_text": bool(cv.get("text_needed")),
        "needs_cv_features": bool(cv.get("features_needed")),
        "is_follow_up": bool(context.get("follow_up")),
        "scrape_request": dict(job.get("scrape") or {}),
    }


def request_values_from_state(state: ConversationState) -> dict[str, Any]:
    return request_values(request_bucket(state))


def request_job_task(state: ConversationState) -> JobTask:
    return cast(JobTask, request_values_from_state(state).get("job_task") or "none")


def request_job_response(state: ConversationState) -> JobResponse:
    return cast(
        JobResponse, request_values_from_state(state).get("job_response") or "none"
    )


def request_show_score(state: ConversationState) -> bool:
    values: dict[str, Any] = request_values_from_state(state)
    return bool(values.get("show_score") or values.get("score_requested"))


def non_request_fields(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if key not in REQUEST_VALUE_KEYS}


def normalize_targets_state(value: dict[str, Any]) -> dict[str, Any]:
    """Read the nested target contract while tolerating pre-migration state."""
    nested_cv: Any = value.get("cv")
    nested_job: Any = value.get("job")
    if isinstance(nested_cv, dict) or isinstance(nested_job, dict):
        return {
            "cv": dict(nested_cv) if isinstance(nested_cv, dict) else {},
            "job": dict(nested_job) if isinstance(nested_job, dict) else {},
            "unresolved_references": list(value.get("unresolved_references") or []),
            "ambiguous": bool(value.get("ambiguous", value.get("targets_ambiguous"))),
        }
    selected_cv_ids: list[str] = [
        str(item).strip()
        for item in (value.get("selected_cv_ids") or [])
        if str(item).strip()
    ]
    selected_job_keys: list[str] = [
        str(item).strip()
        for item in (value.get("selected_job_keys") or [])
        if str(item).strip()
    ]
    return {
        "cv": {
            "scope": value.get("cv_target_scope") or "none",
            "ids": selected_cv_ids,
        },
        "job": {
            "scope": value.get("job_target_scope") or "none",
            "keys": selected_job_keys,
        },
        "unresolved_references": list(value.get("unresolved_references") or []),
        "ambiguous": bool(value.get("targets_ambiguous")),
    }


def targets_bucket(state: ConversationState) -> dict[str, Any]:
    value = routing_bucket(state).get("targets")
    if isinstance(value, dict):
        return normalize_targets_state(value)
    value = state.get("targets")
    if isinstance(value, dict):
        return normalize_targets_state(value)
    value = state.get("selection")
    return normalize_targets_state(value) if isinstance(value, dict) else {}


def plan_bucket(state: ConversationState) -> dict[str, Any]:
    value = routing_bucket(state).get("plan")
    if isinstance(value, dict):
        return dict(value)
    value = state.get("plan")
    if isinstance(value, dict):
        return dict(value)
    value = state.get("router")
    return dict(value) if isinstance(value, dict) else {}


def router_bucket(state: ConversationState) -> dict[str, Any]:
    merged: dict[str, Any] = non_request_fields(state.get("router"))
    merged.update(request_values_from_state(state))
    merged.update(plan_bucket(state))
    plan: dict[str, Any] = plan_bucket(state)
    if plan.get("action") is not None:
        merged["route"] = plan.get("action")
    if plan.get("reason") is not None:
        merged["route_reason"] = plan.get("reason")
    if plan.get("validation") is not None:
        merged["plan_validation"] = plan.get("validation")
    return merged


def selection_bucket(state: ConversationState) -> dict[str, Any]:
    merged: dict[str, Any] = non_request_fields(state.get("selection"))
    merged.update(request_values_from_state(state))
    targets: dict[str, Any] = targets_bucket(state)
    cv_targets: dict[str, Any] = targets.get("cv") or {}
    job_targets: dict[str, Any] = targets.get("job") or {}
    merged.update(
        {
            "cv_target_scope": cv_targets.get("scope") or "none",
            "selected_cv_ids": list(cv_targets.get("ids") or []),
            "selected_cv_id": (
                cv_targets.get("ids")[0]
                if len(cv_targets.get("ids") or []) == 1
                else None
            ),
            "job_target_scope": job_targets.get("scope") or "none",
            "selected_job_keys": list(job_targets.get("keys") or []) or None,
            "unresolved_references": list(targets.get("unresolved_references") or []),
            "targets_ambiguous": bool(targets.get("ambiguous")),
        }
    )
    return merged


def jobs_bucket(state: ConversationState) -> dict[str, Any]:
    value = state.get("jobs")
    return dict(value) if isinstance(value, dict) else {}


def action_results_bucket(state: ConversationState) -> dict[str, Any]:
    value = state.get("action_results")
    return dict(value) if isinstance(value, dict) else {}


def completed_actions(state: ConversationState) -> list[str]:
    """Return the valid agent actions completed during this user message."""
    return [
        action
        for action in plan_bucket(state).get("completed_actions") or []
        if isinstance(action, str) and action in AGENT_ACTIONS
    ]


def execution_bucket(state: ConversationState) -> dict[str, Any]:
    value = state.get("execution")
    return dict(value) if isinstance(value, dict) else {}


def execution_steps(state: ConversationState) -> list[ExecutionStepState]:
    value: Any = execution_bucket(state).get("steps")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _new_action_errors(
    state: ConversationState,
    update: dict[str, Any],
) -> list[str]:
    previous: set[str] = {
        str(item) for item in (state.get("errors") or []) if str(item)
    }
    return [
        str(item)
        for item in (update.get("errors") or [])
        if str(item) and str(item) not in previous
    ]


def _execution_source_node(
    state: ConversationState,
    action: AgentAction,
) -> tuple[str, str, str]:
    actions: set[str] = set(completed_actions(state))
    if action in {"review_cv", "compare_cvs"} and "extract_cv" in actions:
        return "cv_subagent", "extract_cv", "cv_features_ready"
    if action == "match_jobs":
        if "search_jobs" in actions:
            return "job_subagent", "scrape_jobs", "search_results_ready"
        if "extract_job" in actions:
            return "job_subagent", "extract_pasted_job", "job_extraction_ready"
    plan: dict[str, Any] = plan_bucket(state)
    return (
        "chatbot",
        "validate_plan",
        str(plan.get("reason") or "plan_validated"),
    )


def _execution_args(
    action: AgentAction,
    state: ConversationState,
) -> dict[str, Any]:
    request: dict[str, Any] = request_values_from_state(state)
    if action == "extract_cv":
        documents: list[dict[str, Any]] = state_cv_documents(state)
        ids: list[str] = [
            str(document.get("id"))
            for document in documents
            if str(document.get("id") or "").strip()
        ]
        if ids:
            return {"cv_ids": ids}
        return {
            "cv_filenames": [
                str(document.get("filename") or "cv.pdf") for document in documents
            ]
        }
    if action == "review_cv":
        target: dict[str, Any] | None = resolve_selected_cv(state)
        args: dict[str, Any] = {
            "cv_id": target.get("id") if target else None,
            "mode": request.get("review_mode") or "general",
        }
        if request.get("review_focus"):
            args["focus"] = request["review_focus"]
        if request.get("review_target_role"):
            args["target_role"] = request["review_target_role"]
        return {key: value for key, value in args.items() if value not in (None, "")}
    if action == "compare_cvs":
        return {
            "cv_ids": [
                str(document.get("id"))
                for document in resolve_selected_cvs(state)
                if str(document.get("id") or "").strip()
            ]
        }
    if action == "extract_job":
        return {"source": "pasted"}
    if action == "search_jobs":
        raw: Any = jobs_bucket(state).get("scrape_request")
        if not isinstance(raw, dict):
            raw = request.get("scrape_request")
        raw_request: dict[str, Any] = raw if isinstance(raw, dict) else {}
        return {
            key: value
            for key, value in raw_request.items()
            if value not in (None, "", [], {})
        }
    if action == "match_jobs":
        jobs: list[dict[str, Any]] = resolve_selected_jobs(state)
        args = {
            "cv_ids": [
                str(document.get("id"))
                for document in resolve_selected_cvs(state)
                if str(document.get("id") or "").strip()
            ],
            "job_keys": [
                job_selection_key(item, index) for index, item in enumerate(jobs)
            ],
        }
        if request.get("job_source"):
            args["source"] = request["job_source"]
        return args
    return {}


def _execution_context(
    action: AgentAction,
    state: ConversationState,
) -> ExecutionContextState:
    if action != "search_jobs":
        return {}
    request: dict[str, Any] = request_values_from_state(state)
    source: Any = request.get("role_source") or "none"
    if source == "none":
        return {}
    context: ExecutionContextState = {"args_source": source}
    if source == "cv_inferred":
        document: dict[str, Any] | None = resolve_selected_cv(state)
        if document:
            context["source_reference"] = {
                "cv_id": document.get("id"),
                "field": "cv_features.role_tags",
            }
    elif source == "active_goal":
        goal: dict[str, Any] | None = active_job_goal(state)
        if goal:
            context["source_reference"] = {
                "goal_id": goal.get("id"),
                "field": "jobs.active_job_goal.role_constraints",
            }
    else:
        context["source_reference"] = {"field": "routing.request.role.evidence"}
    return context


def _execution_result(
    action: AgentAction,
    state: ConversationState,
    update: dict[str, Any],
    merged_state: ConversationState,
) -> tuple[ExecutionStatus, dict[str, Any], str | None]:
    errors: list[str] = _new_action_errors(state, update)
    jobs_update: dict[str, Any] = (
        dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
    )
    if action == "extract_cv":
        documents = extracted_cv_documents(merged_state)
        result = {
            "cv_count": len(documents),
            "cv_ids": [str(item.get("id")) for item in documents],
            "errors": errors,
        }
        failed = not any(item.get("cv_features") for item in documents)
    elif action == "review_cv":
        review: Any = (update.get("cv") or {}).get("review")
        result = {
            "status": review.get("status") if isinstance(review, dict) else None,
            "feedback_count": len(review.get("feedback") or [])
            if isinstance(review, dict)
            else 0,
            "errors": errors,
        }
        failed = not isinstance(review, dict) or review.get("status") == "unavailable"
    elif action == "compare_cvs":
        comparison: Any = (update.get("cv") or {}).get("comparison")
        result = {
            "candidate_count": len(comparison.get("candidates") or [])
            if isinstance(comparison, dict)
            else 0,
            "errors": errors,
        }
        failed = not isinstance(comparison, dict)
    elif action == "extract_job":
        payload: dict[str, Any] = slim_extract_job_result(jobs_update, state)
        result = {
            "job_count": payload.get("job_count", 0),
            "validation_status": payload.get("validation_status"),
            "errors": errors,
        }
        failed = payload.get("validation_status") != "valid"
    elif action == "search_jobs":
        payload = slim_search_result(jobs_update, state)
        result = {
            "job_count": payload.get("job_count", 0),
            "scrape_total": payload.get("scrape_total", 0),
            "truncated": bool(payload.get("scrape_truncated")),
            "errors": errors,
        }
        failed = any(item.startswith("job_scraping_failed:") for item in errors)
    else:
        matches: list[Any] = [
            item
            for item in (jobs_update.get("matches") or [])
            if isinstance(item, dict)
        ]
        result = {"match_count": len(matches), "errors": errors}
        failed = bool(errors)
    status: ExecutionStatus = "failed" if failed else "completed"
    error: str | None = errors[0] if failed and errors else None
    return status, result, error


def build_execution_step(
    state: ConversationState,
    action: AgentAction,
    update: dict[str, Any],
    merged_state: ConversationState,
) -> ExecutionStepState:
    source_graph, source_node, source_reason = _execution_source_node(state, action)
    destination_graph, destination_node = EXECUTION_DESTINATIONS[action]
    status, result, error = _execution_result(action, state, update, merged_state)
    route: Any = router_bucket(state).get("route") or plan_bucket(state).get("action")
    from_node: ExecutionFromNodeState = {
        "graph": source_graph,
        "node": source_node,
        "uses": list(EXECUTION_NODE_USES.get((source_graph, source_node), [])),
        "note": {
            "summary": "Selected the next executable action.",
            "reason": source_reason,
        },
        "output": {"route": route or "respond", "next_action": action},
    }
    to_node: ExecutionToNodeState = {
        "graph": destination_graph,
        "node": destination_node,
        "uses": list(
            EXECUTION_NODE_USES.get((destination_graph, destination_node), [])
        ),
        "note": {
            "summary": f"Executed {action}.",
            "reason": "action_failed" if status == "failed" else "action_completed",
        },
        "args": _execution_args(action, state),
        "result": result,
    }
    return {
        "index": len(execution_steps(state)) + 1,
        "action": action,
        "status": status,
        "from": from_node,
        "to": to_node,
        "context": _execution_context(action, state),
        "error": error,
    }


def build_skipped_execution_step(
    state: ConversationState,
    action: AgentAction,
) -> ExecutionStepState:
    destination_graph, destination_node = EXECUTION_DESTINATIONS[action]
    if action == "search_jobs":
        result = {
            "reused": True,
            "job_count": len(job_results_for_display(state)),
        }
    else:
        result = {
            "reused": True,
            "match_count": len(jobs_bucket(state).get("matches") or []),
        }
    return {
        "index": len(execution_steps(state)) + 1,
        "action": action,
        "status": "skipped",
        "from": {
            "graph": "chatbot",
            "node": "validate_plan",
            "uses": list(EXECUTION_NODE_USES[("chatbot", "validate_plan")]),
            "note": {
                "summary": "Selected the next executable action.",
                "reason": str(plan_bucket(state).get("reason") or "plan_validated"),
            },
            "output": {
                "route": router_bucket(state).get("route") or "respond",
                "next_action": action,
            },
        },
        "to": {
            "graph": destination_graph,
            "node": destination_node,
            "uses": list(EXECUTION_NODE_USES[(destination_graph, destination_node)]),
            "note": {
                "summary": f"Skipped {action}.",
                "reason": "reused_existing_result",
            },
            "args": _execution_args(action, state),
            "result": result,
        },
        "context": _execution_context(action, state),
        "error": None,
    }


def record_completed_action(
    state: ConversationState,
    action: AgentAction,
    update: dict[str, Any],
    *,
    emit_result: bool = False,
) -> dict[str, Any]:
    """Attach one executed action to a node update without duplicate entries."""
    actions: list[str] = completed_actions(state)
    if action not in actions:
        actions.append(action)
    routing_update: Any = update.get("routing")
    plan_update: Any = (
        routing_update.get("plan")
        if isinstance(routing_update, dict)
        else update.get("plan") or update.get("router")
    )
    nested: dict[str, Any] = dict(plan_update) if isinstance(plan_update, dict) else {}
    rest: dict[str, Any] = {
        key: value
        for key, value in update.items()
        if key not in {"router", "plan", "routing"}
    }
    if emit_result and action in USER_FACING_ACTIONS:
        payload: dict[str, Any] | None = slim_action_result(action, update, state)
        if payload is not None:
            result_messages: list[AnyMessage] = build_action_result_messages(
                action, payload
            )
            existing_messages: Any = rest.get("messages")
            if isinstance(existing_messages, list) and existing_messages:
                rest["messages"] = [*existing_messages, *result_messages]
            else:
                rest["messages"] = result_messages
    recorded_routing: dict[str, Any] = (
        dict(routing_update) if isinstance(routing_update, dict) else {}
    )
    recorded_routing["plan"] = {**nested, "completed_actions": actions}
    recorded: dict[str, Any] = {**rest, "routing": recorded_routing}
    merged_state: ConversationState = {
        **state,
        **{key: value for key, value in rest.items() if key != "messages"},
    }
    if isinstance(update.get("cv"), dict):
        merged_state["cv"] = {**cv_bucket(state), **update["cv"]}
    if isinstance(update.get("jobs"), dict):
        merged_state["jobs"] = {**jobs_bucket(state), **update["jobs"]}
    if isinstance(update.get("routing"), dict):
        merged_state["routing"] = merge_routing_maps(
            routing_bucket(state),
            update["routing"],
        )
    if isinstance(update.get("targets"), dict):
        merged_state["targets"] = {**targets_bucket(state), **update["targets"]}
    if isinstance(update.get("selection"), dict):
        merged_state["targets"] = {
            **targets_bucket(state),
            **update["selection"],
        }
    step: ExecutionStepState = build_execution_step(
        state,
        action,
        update,
        merged_state,
    )
    recorded["execution"] = {
        "steps": [*execution_steps(state), step],
    }
    snapshot: dict[str, Any] | None = reusable_action_snapshot(action, update, state)
    if snapshot is None:
        return recorded
    fingerprint: str | None = action_fingerprint(action, merged_state)
    if not fingerprint:
        return recorded
    return {
        **recorded,
        "action_results": {
            action: {
                "fingerprint": fingerprint,
                "executed_at": datetime.now(timezone.utc).isoformat(),
                "snapshot": snapshot,
            }
        },
    }


def slim_review_result(review: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(review, dict):
        return None
    feedback: list[dict[str, Any]] = []
    for item in review.get("feedback") or []:
        if not isinstance(item, dict):
            continue
        feedback.append(
            {
                "title": item.get("title"),
                "observation": item.get("observation"),
                "recommendation": item.get("recommendation"),
            }
        )
    return {
        "status": review.get("status"),
        "mode": review.get("mode"),
        "focus": review.get("focus"),
        "target_role": review.get("target_role"),
        "overall_score": review.get("overall_score"),
        "feedback": feedback,
    }


def slim_comparison_result(comparison: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(comparison, dict):
        return None
    candidates: list[dict[str, Any]] = []
    for item in comparison.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        candidates.append(
            {
                "filename": item.get("filename"),
                "strengths": item.get("strengths") or [],
                "weaknesses": item.get("weaknesses") or item.get("gaps") or [],
                "summary": item.get("summary"),
            }
        )
    return {
        "overview": comparison.get("overview"),
        "candidates": candidates,
        "recommendation": comparison.get("recommendation"),
    }


def slim_job_card(
    card: dict[str, Any] | None,
    *,
    include_description: bool = True,
) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {}
    fields: tuple[str, ...] = (
        "company",
        "title",
        "location",
        "posted_date",
        "salary",
        "url",
    )
    if include_description:
        fields = (*fields, "description")
    return {
        key: card.get(key)
        for key in fields
        if card.get(key) is not None and card.get(key) != "" and card.get(key) != []
    }


def slim_search_result(
    jobs_update: dict[str, Any], state: ConversationState
) -> dict[str, Any]:
    results: list[dict[str, Any]] = [
        item
        for item in (
            jobs_update.get("results") or jobs_bucket(state).get("results") or []
        )
        if isinstance(item, dict)
    ]
    active_keys: Any = jobs_update.get("active_job_keys")
    if not isinstance(active_keys, list) or not active_keys:
        active_keys = jobs_bucket(state).get("active_job_keys")
    if isinstance(active_keys, list) and active_keys:
        wanted: set[str] = {str(key).strip() for key in active_keys if str(key).strip()}
        results = [
            item
            for index, item in enumerate(results)
            if job_selection_key(item, index) in wanted
        ]
    jobs: list[dict[str, Any]] = []
    for item in results:
        raw_card: Any = (
            item.get("job_card") if isinstance(item.get("job_card"), dict) else None
        )
        card: dict[str, Any] = {
            field: raw_card.get(field)
            for field in ("title", "location", "posted_date", "salary", "url")
            if isinstance(raw_card, dict) and raw_card.get(field) not in (None, "", [])
        }
        if card:
            jobs.append(card)
    return {
        "job_count": len(jobs),
        "scrape_total": jobs_update.get(
            "scrape_total", jobs_bucket(state).get("scrape_total", 0)
        ),
        "scrape_truncated": bool(
            jobs_update.get(
                "scrape_truncated",
                jobs_bucket(state).get("scrape_truncated", False),
            )
        ),
        "jobs": jobs,
    }


def slim_match_result(
    jobs_update: dict[str, Any],
    *,
    show_score: bool = True,
) -> dict[str, Any]:
    assessment: dict[str, Any] = build_match_assessment(
        [item for item in (jobs_update.get("matches") or []) if isinstance(item, dict)],
        show_score=show_score,
    )
    return {
        "match_count": assessment["match_count"],
        "match_assessment": assessment,
        "matches": assessment["matches"],
    }


def project_match_item(
    item: dict[str, Any],
    *,
    show_score: bool = True,
) -> dict[str, Any] | None:
    score: Any = item.get("score")
    if not isinstance(score, dict):
        score = {}
    fit_verdict: str | None = score.get("fit_verdict")
    verdict_reason_code: str | None = score.get("verdict_reason_code")
    if fit_verdict not in {"yes", "no", "uncertain", "unknown"}:
        fit_verdict, verdict_reason_code = matching_score_module.classify_fit_verdict(
            normalized_score=score.get("normalized_score"),
            score_coverage=score.get("score_coverage"),
            decision=score.get("decision"),
        )
    card: dict[str, Any] = slim_job_card(
        item.get("job_card") if isinstance(item.get("job_card"), dict) else None
    )
    projected: dict[str, Any] = {
        "cv_id": item.get("cv_id"),
        "cv_filename": item.get("cv_filename"),
        "job_key": item.get("job_key"),
        "job": card,
        "title": card.get("title") or item.get("title"),
        "company": card.get("company") or item.get("company"),
        "url": card.get("url") or item.get("url"),
        "fit_verdict": fit_verdict,
        "verdict_reason_code": verdict_reason_code,
        "review_reason_codes": list(score.get("review_reason_codes") or [])[:5],
    }
    if not show_score:
        return projected
    projected["decision"] = score.get("decision")
    projected["score_coverage"] = score.get("score_coverage")
    normalized: Any = score.get("normalized_score")
    if fit_verdict == "uncertain":
        if normalized is not None:
            projected["provisional_score"] = normalized
    elif normalized is not None:
        projected["score"] = normalized
    return projected


def build_match_assessment(
    matches: list[dict[str, Any]],
    *,
    show_score: bool = True,
) -> dict[str, Any]:
    projected: list[dict[str, Any]] = []
    yes_count = 0
    no_count = 0
    uncertain_count = 0
    for item in matches:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] | None = project_match_item(item, show_score=show_score)
        if row is None:
            continue
        projected.append(row)
        verdict: str = str(row.get("fit_verdict") or "uncertain")
        if verdict == "yes":
            yes_count += 1
        elif verdict == "no":
            no_count += 1
        else:
            uncertain_count += 1

    aggregate: str
    if not projected:
        aggregate = "uncertain"
    elif uncertain_count == len(projected):
        aggregate = "uncertain"
    elif yes_count == len(projected):
        aggregate = "yes"
    elif no_count == len(projected):
        aggregate = "no"
    elif yes_count > 0:
        aggregate = "some"
    else:
        aggregate = "uncertain"

    return {
        "verdict": aggregate,
        "match_count": len(projected),
        "yes_count": yes_count,
        "no_count": no_count,
        "uncertain_count": uncertain_count,
        "matches": projected,
    }


def slim_extract_job_result(
    jobs_update: dict[str, Any],
    state: ConversationState,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = [
        item for item in (jobs_update.get("results") or []) if isinstance(item, dict)
    ]
    if not results:
        results = [
            item
            for item in (jobs_bucket(state).get("results") or [])
            if isinstance(item, dict)
        ]
    latest: dict[str, Any] = results[-1] if results else {}
    return {
        "job_count": len(results),
        "validation_status": latest.get("validation_status"),
        "job": slim_job_card(
            latest.get("job_card") if isinstance(latest.get("job_card"), dict) else None
        ),
    }


def slim_action_result(
    action: AgentAction,
    update: dict[str, Any],
    state: ConversationState,
) -> dict[str, Any] | None:
    errors: list[str] = [
        str(item)
        for item in (update.get("errors") or state.get("errors") or [])[-3:]
        if item
    ]
    if action == "review_cv":
        review: Any = (update.get("cv") or {}).get("review")
        slim: dict[str, Any] | None = slim_review_result(
            review if isinstance(review, dict) else None
        )
        if slim is None:
            return {
                "ok": False,
                "action": action,
                "errors": errors or ["cv_review_failed"],
            }
        return {
            "ok": slim.get("status") != "unavailable",
            "action": action,
            "review": slim,
            "errors": errors,
        }
    if action == "compare_cvs":
        comparison: Any = (update.get("cv") or {}).get("comparison")
        slim_comparison: dict[str, Any] | None = slim_comparison_result(
            comparison if isinstance(comparison, dict) else None
        )
        if slim_comparison is None:
            return {
                "ok": False,
                "action": action,
                "errors": errors or ["cv_comparison_failed"],
            }
        return {
            "ok": True,
            "action": action,
            "comparison": slim_comparison,
            "errors": errors,
        }
    if action == "search_jobs":
        jobs_update: dict[str, Any] = (
            dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
        )
        return {
            "ok": True,
            "action": action,
            **slim_search_result(jobs_update, state),
            "errors": errors,
        }
    if action == "match_jobs":
        jobs_update = (
            dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
        )
        return {
            "ok": bool(jobs_update.get("matches")),
            "action": action,
            **slim_match_result(
                jobs_update,
                show_score=(
                    bool(selection_bucket(state).get("show_score"))
                    if "show_score" in selection_bucket(state)
                    else True
                ),
            ),
            "errors": errors,
        }
    if action == "extract_job":
        jobs_update = (
            dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
        )
        payload: dict[str, Any] = slim_extract_job_result(jobs_update, state)
        return {
            "ok": payload.get("validation_status") == "valid",
            "action": action,
            **payload,
            "errors": errors,
        }
    return None


def build_action_result_messages(
    action: AgentAction,
    payload: dict[str, Any],
) -> list[AnyMessage]:
    tool_call_id: str = f"{action}:{uuid.uuid4()}"
    content: str = short_text(
        json.dumps(payload, ensure_ascii=False),
        MAX_ACTION_RESULT_CHARS,
    )
    return [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": tool_call_id,
                    "name": action,
                    "args": {},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=action,
        ),
    ]


def state_cv_documents(state: ConversationState) -> list[dict[str, Any]]:
    documents: Any = cv_bucket(state).get("documents")
    if isinstance(documents, list) and documents:
        return [dict(item) for item in documents if isinstance(item, dict)]
    return []


def cv_needs_extraction_update(documents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cv": {
            "needs_extraction": any(
                (doc.get("cv_text") or "").strip() and not doc.get("cv_features")
                for doc in documents
            )
        }
    }


def cvs_need_extraction(state: ConversationState) -> bool:
    documents: list[dict[str, Any]] = state_cv_documents(state)
    return bool(
        cv_bucket(state).get("needs_extraction")
        or any(
            (doc.get("cv_text") or "").strip() and not doc.get("cv_features")
            for doc in documents
        )
    )


def intent_requires_cv_features(state: ConversationState) -> bool:
    router: dict[str, Any] = router_bucket(state)
    route: RouteName = router.get("route") or "respond"
    if route in CV_FEATURE_INTENTS:
        return True
    return route == "respond" and bool(router.get("needs_cv_features"))


def cv_feature_summary(features: dict[str, Any] | None) -> dict[str, Any]:
    features: dict[str, Any] = features or {}
    return {
        key: features.get(key)
        for key in (
            "role_tags",
            "skill_names",
            "seniority",
            "years_of_experience",
            "current_location",
        )
        if features.get(key) not in (None, [], "")
    }


def extracted_cv_documents(state: ConversationState) -> list[dict[str, Any]]:
    """Return uploaded CVs that have usable extracted features."""
    return [
        document
        for document in state_cv_documents(state)
        if (document.get("cv_text") or "").strip() and document.get("cv_features")
    ]


def job_selection_key(item: dict[str, Any], index: int) -> str:
    card: Any = item.get("job_card") if isinstance(item, dict) else None
    if isinstance(card, dict):
        url: str = str(card.get("url") or "").strip()
        if url:
            return "url:" + url.casefold()
        title: str = str(card.get("title") or "").strip().casefold()
        company: str = str(card.get("company") or "").strip().casefold()
        if title or company:
            return f"title:{title}|company:{company}"
        description: str = str(card.get("description") or "").strip()
        if description:
            digest: str = hashlib.sha1(
                description.encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:16]
            return f"desc:{digest}"
    return f"idx:{index}"


def merge_job_results(
    existing: list[dict[str, Any]] | None,
    incoming: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for source in (existing or [], incoming or []):
        for index, item in enumerate(source):
            if not isinstance(item, dict):
                continue
            key: str = job_selection_key(item, index)
            if key not in by_key:
                order.append(key)
            by_key[key] = item
    return [by_key[key] for key in order]


def resolve_selected_cv(state: ConversationState) -> dict[str, Any] | None:
    documents: list[dict[str, Any]] = resolve_selected_cvs(state)
    return documents[0] if documents else None


def resolve_selected_cvs(state: ConversationState) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = extracted_cv_documents(state)
    if not documents:
        return []
    selected_ids: Any = selection_bucket(state).get("selected_cv_ids")
    if isinstance(selected_ids, list) and selected_ids:
        wanted: set[str] = {
            str(item).strip() for item in selected_ids if str(item).strip()
        }
        selected: list[dict[str, Any]] = [
            document
            for document in documents
            if str(document.get("id") or "") in wanted
        ]
        return selected or documents
    selected_id: str = str(selection_bucket(state).get("selected_cv_id") or "").strip()
    if not selected_id:
        return documents
    selected: list[dict[str, Any]] = [
        document
        for document in documents
        if str(document.get("id") or "") == selected_id
    ]
    return selected or documents


def resolve_selected_jobs(state: ConversationState) -> list[dict[str, Any]]:
    job_results: Any = jobs_bucket(state).get("results") or []
    results: list[dict[str, Any]] = [
        item
        for item in job_results
        if isinstance(item, dict) and item.get("validation_status") == "valid"
    ]
    raw_keys: Any = selection_bucket(state).get("selected_job_keys")
    if isinstance(raw_keys, list) and raw_keys:
        wanted: set[str] = {str(key).strip() for key in raw_keys if str(key).strip()}
        if wanted:
            selected: list[dict[str, Any]] = []
            for index, item in enumerate(job_results):
                if (
                    not isinstance(item, dict)
                    or item.get("validation_status") != "valid"
                ):
                    continue
                if job_selection_key(item, index) in wanted:
                    selected.append(item)
            return selected or results
    active_keys: Any = jobs_bucket(state).get("active_job_keys")
    if isinstance(active_keys, list) and active_keys:
        wanted = {str(key).strip() for key in active_keys if str(key).strip()}
        if wanted:
            selected = []
            for index, item in enumerate(job_results):
                if (
                    not isinstance(item, dict)
                    or item.get("validation_status") != "valid"
                ):
                    continue
                if job_selection_key(item, index) in wanted:
                    selected.append(item)
            return selected
    return results


def canonical_json_hash(value: Any) -> str:
    encoded: bytes = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_fingerprint_text(value: Any) -> str:
    return casefolded_text(value)


def is_cv_upload_turn(state: ConversationState) -> bool:
    return PDF_UPLOAD_MARKER in last_user_text(state)


def default_request_fields() -> dict[str, Any]:
    return {
        "goal": {"name": "general_question", "reason": "", "confidence": 1.0},
        "job": {
            "task": "none",
            "response": "none",
            "source": "none",
            "input": None,
            "refresh": False,
            "scrape": {},
        },
        "role": {
            "constraints": [],
            "evidence": None,
            "source": "none",
            "candidates": [],
        },
        "assessment": {"requested": False, "detail_level": "summary"},
        "score": {"requested": False, "visible": False},
        "review": {
            "target_role": None,
            "mode": "general",
            "focus": None,
            "reason": None,
        },
        "cv": {"text_needed": False, "features_needed": False},
        "context": {"follow_up": False},
    }


def default_target_fields() -> dict[str, Any]:
    return {
        "cv": {"scope": "none", "ids": []},
        "job": {"scope": "none", "keys": []},
        "unresolved_references": [],
        "ambiguous": False,
    }


def default_selection_fields() -> dict[str, Any]:
    """Compatibility view for legacy helpers; active state uses request/targets."""
    return {**default_request_fields(), **default_target_fields()}


def cv_version(document: dict[str, Any] | None) -> str:
    if not isinstance(document, dict):
        return ""
    text: str = (document.get("cv_text") or "").strip()
    if text:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    features: Any = document.get("cv_features")
    if isinstance(features, dict) and features:
        return canonical_json_hash(features)
    return str(document.get("id") or "")


def job_content_version(item: dict[str, Any]) -> str:
    features: Any = item.get("matching_features")
    if isinstance(features, dict):
        content_hash: Any = features.get("content_hash")
        if content_hash:
            return str(content_hash)
    card: Any = item.get("job_card") if isinstance(item.get("job_card"), dict) else item
    return canonical_json_hash(card or {})


def normalize_scrape_request(request: dict[str, Any] | None) -> dict[str, Any]:
    raw: dict[str, Any] = request if isinstance(request, dict) else {}
    keywords: list[str] = sorted(
        {
            normalize_fingerprint_text(item)
            for item in (raw.get("keywords") or [])
            if normalize_fingerprint_text(item)
        }
    )
    sites: list[str] = sorted(
        {
            normalize_fingerprint_text(item)
            for item in (raw.get("sites") or [])
            if normalize_fingerprint_text(item)
        }
    )
    return {
        "keywords": keywords,
        "sites": sites,
        "max_age_hours": raw.get("max_age_hours"),
    }


def parse_executed_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed: datetime = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def search_result_is_fresh(
    entry: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> bool:
    if not isinstance(entry, dict):
        return False
    executed: datetime | None = parse_executed_at(entry.get("executed_at"))
    if executed is None:
        return False
    current: datetime = now or datetime.now(timezone.utc)
    return (current - executed).total_seconds() <= SEARCH_RESULT_TTL_SECONDS


def stored_action_result(
    state: ConversationState,
    action: str,
) -> dict[str, Any] | None:
    entry: Any = action_results_bucket(state).get(action)
    return dict(entry) if isinstance(entry, dict) else None


def action_fingerprint(action: str, state: ConversationState) -> str | None:
    selection: dict[str, Any] = selection_bucket(state)
    if action == "review_cv":
        target: dict[str, Any] | None = resolve_selected_cv(state)
        if target is None:
            documents: list[dict[str, Any]] = extracted_cv_documents(
                state
            ) or state_cv_documents(state)
            target = documents[0] if documents else None
        if target is None:
            return None
        return canonical_json_hash(
            {
                "action": "review_cv",
                "cv_id": str(target.get("id") or ""),
                "cv_version": cv_version(target),
                "mode": selection.get("review_mode") or "general",
                "focus": normalize_fingerprint_text(selection.get("review_focus")),
                "target_role": normalize_fingerprint_text(
                    selection.get("review_target_role")
                ),
            }
        )
    if action == "compare_cvs":
        documents = extracted_cv_documents(state)
        if len(documents) < 2:
            return None
        return canonical_json_hash(
            {
                "action": "compare_cvs",
                "cvs": [
                    {"id": str(doc.get("id") or ""), "version": cv_version(doc)}
                    for doc in documents
                ],
            }
        )
    if action == "extract_job":
        text: str = normalize_fingerprint_text(
            selection.get("job_input_text") or last_user_text(state)
        )
        if not text:
            return None
        return canonical_json_hash({"action": "extract_job", "text": text})
    if action == "search_jobs":
        request: dict[str, Any] = normalize_scrape_request(
            jobs_bucket(state).get("scrape_request")
        )
        return canonical_json_hash({"action": "search_jobs", **request})
    if action == "match_jobs":
        cvs: list[dict[str, Any]] = resolve_selected_cvs(state)
        jobs: list[dict[str, Any]] = resolve_selected_jobs(state)
        if not cvs or not jobs:
            return None
        payload: dict[str, Any] = {
            "action": "match_jobs",
            "cvs": [
                {"id": str(doc.get("id") or ""), "version": cv_version(doc)}
                for doc in cvs
            ],
            "jobs": [
                {
                    "key": job_selection_key(item, index),
                    "version": job_content_version(item),
                }
                for index, item in enumerate(jobs)
            ],
        }
        if selection.get("job_source") == "search":
            payload["search"] = normalize_scrape_request(
                jobs_bucket(state).get("scrape_request")
            )
        return canonical_json_hash(payload)
    return None


def action_result_is_reusable(
    state: ConversationState,
    action: str,
    fingerprint: str | None,
    *,
    now: datetime | None = None,
) -> bool:
    if not fingerprint:
        return False
    entry: dict[str, Any] | None = stored_action_result(state, action)
    if entry is None or entry.get("fingerprint") != fingerprint:
        return False
    snapshot: Any = entry.get("snapshot")
    if not isinstance(snapshot, dict) or not snapshot:
        return False
    if action == "search_jobs" and not search_result_is_fresh(entry, now=now):
        return False
    return True


def current_search_is_reusable(state: ConversationState) -> bool:
    return action_result_is_reusable(
        state,
        "search_jobs",
        action_fingerprint("search_jobs", state),
    )


def normalize_role_constraints(values: Any) -> list[str]:
    seen: set[str] = set()
    constraints: list[str] = []
    for item in values or []:
        text: str = normalize_fingerprint_text(item)
        if text and text not in seen:
            seen.add(text)
            constraints.append(text)
    return constraints


def display_role_constraints(constraints: list[str]) -> list[str]:
    return [item.title() for item in constraints]


def first_contiguous_phrase(text: str, candidates: Any) -> str | None:
    """Return the first candidate as written in text, ignoring case."""
    folded_text: str = text.casefold()
    for candidate in candidates or []:
        phrase: str = str(candidate or "").strip()
        if not phrase:
            continue
        start: int = folded_text.find(phrase.casefold())
        if start >= 0:
            return text[start : start + len(phrase)]
    return None


def normalize_job_role_fields(
    latest: str,
    *,
    job_task: str,
    role_constraints: Any,
    role_evidence: Any,
    role_source: Any,
    scrape_keywords: Any,
) -> tuple[list[str], str | None, str]:
    """Recover explicit role fields when the structured router omits them."""
    constraints: list[str] = normalize_role_constraints(role_constraints)
    evidence: str = str(role_evidence or "").strip()
    source: str = str(role_source or "none")
    if job_task != "search":
        return constraints, evidence or None, source

    keywords: list[str] = normalize_role_constraints(scrape_keywords)
    if not constraints:
        constraints = keywords
    if not evidence:
        evidence = first_contiguous_phrase(latest, keywords or constraints) or ""
    if source == "none" and evidence:
        source = "explicit"
    return constraints, evidence or None, source


def role_constraints_from_cv(document: dict[str, Any] | None) -> list[str]:
    if not isinstance(document, dict):
        return []
    features: Any = document.get("cv_features")
    if not isinstance(features, dict):
        return []
    return normalize_role_constraints(features.get("role_tags") or [])


def normalize_role_candidates(values: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values or []:
        if isinstance(item, RoleCandidate):
            raw_role: Any = item.role
            raw_confidence: Any = item.confidence
            raw_evidence: Any = item.evidence
        elif isinstance(item, dict):
            raw_role = item.get("role")
            raw_confidence = item.get("confidence", 1.0)
            raw_evidence = item.get("evidence")
        else:
            continue
        role: str = normalize_fingerprint_text(raw_role)
        evidence: str = short_text(raw_evidence or "", 300)
        if not role or role in seen or not evidence:
            continue
        try:
            confidence: float = max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        seen.add(role)
        candidates.append(
            {
                "role": role,
                "confidence": confidence,
                "evidence": evidence,
            }
        )
    return sorted(candidates, key=lambda item: item["confidence"], reverse=True)[:5]


def build_active_job_goal(
    *,
    source: str,
    role_constraints: list[str],
    cv_id: str | None,
    cv_version: str | None,
    originating_turn: str,
    invalidated: bool = False,
    invalidation_reason: str | None = None,
) -> dict[str, Any]:
    constraints: list[str] = normalize_role_constraints(role_constraints)
    identity: dict[str, Any] = {
        "source": source,
        "role_constraints": constraints,
        "cv_id": cv_id or None,
        "cv_version": cv_version or None,
    }
    return {
        "id": canonical_json_hash(identity),
        "source": source,
        "role_constraints": constraints,
        "cv_id": cv_id or None,
        "cv_version": cv_version or None,
        "originating_turn": originating_turn,
        "invalidated": invalidated,
        "invalidation_reason": invalidation_reason,
    }


def active_job_goal(state: ConversationState) -> dict[str, Any] | None:
    value: Any = jobs_bucket(state).get("active_job_goal")
    return dict(value) if isinstance(value, dict) else None


def pending_match_request(state: ConversationState) -> dict[str, Any] | None:
    value: Any = jobs_bucket(state).get("pending_match")
    return dict(value) if isinstance(value, dict) else None


def unambiguous_extracted_cv(state: ConversationState) -> dict[str, Any] | None:
    documents: list[dict[str, Any]] = extracted_cv_documents(state)
    selected_id: str = str(selection_bucket(state).get("selected_cv_id") or "").strip()
    if selected_id:
        matched: list[dict[str, Any]] = [
            document
            for document in documents
            if str(document.get("id") or "") == selected_id
        ]
        return matched[0] if len(matched) == 1 else None
    if len(documents) == 1:
        return documents[0]
    return None


def active_job_goal_is_usable(state: ConversationState) -> bool:
    goal: dict[str, Any] | None = active_job_goal(state)
    if goal is None or goal.get("invalidated") or not goal.get("role_constraints"):
        return False
    document: dict[str, Any] | None = unambiguous_extracted_cv(state)
    if document is None:
        return False
    goal_cv_id: str = str(goal.get("cv_id") or "").strip()
    if goal_cv_id and goal_cv_id != str(document.get("id") or ""):
        return False
    goal_version: str = str(goal.get("cv_version") or "")
    if goal_version and goal_version != cv_version(document):
        return False
    return True


def planned_job_stages(
    state: ConversationState,
    *,
    route: RouteName,
    selection: dict[str, Any],
    jobs_update: dict[str, Any],
) -> list[str]:
    gated: ConversationState = {
        **state,
        "routing": {
            "request": request_state_fields(selection),
            "targets": target_state_fields(selection),
        },
        "jobs": {**jobs_bucket(state), **jobs_update},
    }
    refresh: bool = bool(
        selection.get("job_refresh") or selection.get("refresh_requested")
    )
    if route == "search_jobs":
        if (
            not refresh
            and current_search_is_reusable(gated)
            and resolve_selected_jobs(gated)
        ):
            return []
        return ["scrape_jobs"]
    if route != "match_jobs":
        return []
    stages: list[str] = []
    if selection.get("job_source") == "search":
        if refresh or not (
            current_search_is_reusable(gated) and resolve_selected_jobs(gated)
        ):
            stages.append("scrape_jobs")
    elif selection.get("job_source") == "pasted" and not jobs_bucket(gated).get(
        "results"
    ):
        stages.append("extract_job")
    if refresh or not action_result_is_reusable(
        gated,
        "match_jobs",
        action_fingerprint("match_jobs", gated),
    ):
        stages.append("match_jobs")
    return stages


def persist_planned_job_state(
    state: ConversationState,
    *,
    decision: RouteDecision,
    route: RouteName,
    selection: dict[str, Any],
    jobs_update: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Persist planner-selected job state without reclassifying the request."""
    current_goal: dict[str, Any] | None = active_job_goal(state)
    role_constraints: list[str] = normalize_role_constraints(
        selection.get("role_constraints")
    )
    role_evidence: str | None = selection.get("role_evidence")
    role_source: RoleSource = selection.get("role_source") or "none"
    if not role_constraints and current_goal and not current_goal.get("invalidated"):
        role_constraints = normalize_role_constraints(
            current_goal.get("role_constraints")
        )
        role_source = "active_goal"

    selected_state: ConversationState = {
        **state,
        "routing": {
            "request": request_state_fields(selection),
            "targets": target_state_fields(selection),
        },
    }
    document: dict[str, Any] | None = unambiguous_extracted_cv(selected_state)

    if route == "search_jobs" and role_constraints:
        refresh_requested: bool = bool(
            selection.get("job_refresh") or selection.get("refresh_requested")
        )
        reuse_goal: bool = (
            refresh_requested
            and current_goal is not None
            and not current_goal.get("invalidated")
        )
        goal: dict[str, Any] = (
            current_goal
            if reuse_goal
            else build_active_job_goal(
                source=(
                    "cv_derived" if role_source == "cv_inferred" else "explicit_search"
                ),
                role_constraints=role_constraints,
                cv_id=str(document.get("id")) if document else None,
                cv_version=cv_version(document) if document else None,
                originating_turn=last_user_text(state),
            )
        )
        jobs_update["active_job_goal"] = goal
        scrape_request: dict[str, Any] = dict(jobs_update.get("scrape_request") or {})
        scrape_request["keywords"] = display_role_constraints(role_constraints)
        jobs_update["scrape_request"] = scrape_request
        selection = {
            **selection,
            "role_constraints": role_constraints,
            "role_evidence": role_evidence,
            "role_source": role_source,
            "job_source": "search",
        }
    elif route == "match_jobs":
        if current_goal is not None and not current_goal.get("invalidated"):
            jobs_update["active_job_goal"] = current_goal
        if selection.get("job_source") == "none" and jobs_bucket(state).get("results"):
            selection = {**selection, "job_source": "existing"}
        if decision.assessment_requested:
            selection = {**selection, "assessment_requested": True}

    if selection.get("job_task") == "cancel":
        if current_goal is not None:
            jobs_update["active_job_goal"] = {
                **current_goal,
                "invalidated": True,
                "invalidation_reason": "cancelled",
            }
        jobs_update["pending_match"] = None

    policy: dict[str, Any] = {
        "planned_stages": [],
        "policy_reason": "planner_selected_action",
        "active_goal_id": (jobs_update.get("active_job_goal") or {}).get("id"),
    }
    if route in {"search_jobs", "match_jobs"}:
        policy["planned_stages"] = planned_job_stages(
            state,
            route=route,
            selection=selection,
            jobs_update=jobs_update,
        )
    return selection, jobs_update, policy


def reusable_action_snapshot(
    action: AgentAction,
    update: dict[str, Any],
    state: ConversationState,
) -> dict[str, Any] | None:
    if action == "review_cv":
        review: Any = (update.get("cv") or {}).get("review")
        if not isinstance(review, dict):
            return None
        if review.get("status") == "unavailable":
            return None
        if not (review.get("feedback") or review.get("overall_score") is not None):
            return None
        return {"cv": {"review": review}}
    if action == "compare_cvs":
        comparison: Any = (update.get("cv") or {}).get("comparison")
        if not isinstance(comparison, dict):
            return None
        if not comparison.get("overview") or not comparison.get("candidates"):
            return None
        return {"cv": {"comparison": comparison}}
    if action == "extract_job":
        jobs_update: dict[str, Any] = (
            dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
        )
        results: list[dict[str, Any]] = [
            item
            for item in (jobs_update.get("results") or [])
            if isinstance(item, dict)
        ]
        latest: dict[str, Any] | None = results[-1] if results else None
        if latest is None or latest.get("validation_status") != "valid":
            return None
        return {
            "jobs": {
                "results": results,
                "active_job_keys": [
                    job_selection_key(latest, max(len(results) - 1, 0))
                ],
            }
        }
    if action == "search_jobs":
        jobs_update = (
            dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
        )
        results = [
            item
            for item in (jobs_update.get("results") or [])
            if isinstance(item, dict)
        ]
        if not results:
            return None
        keys: Any = jobs_update.get("active_job_keys")
        if not isinstance(keys, list) or not keys:
            keys = [
                job_selection_key(item, index) for index, item in enumerate(results)
            ]
        return {
            "jobs": {
                "results": results,
                "active_job_keys": keys,
                "scrape_total": jobs_update.get("scrape_total", len(results)),
                "scrape_truncated": bool(jobs_update.get("scrape_truncated")),
            }
        }
    if action == "match_jobs":
        jobs_update = (
            dict(update.get("jobs")) if isinstance(update.get("jobs"), dict) else {}
        )
        matches: list[dict[str, Any]] = [
            item
            for item in (jobs_update.get("matches") or [])
            if isinstance(item, dict)
        ]
        if not matches:
            return None
        return {"jobs": {"matches": matches}}
    return None


def job_results_for_display(state: ConversationState) -> list[dict[str, Any]]:
    jobs_state: dict[str, Any] = jobs_bucket(state)
    active_keys: Any = jobs_state.get("active_job_keys")
    selected_keys: Any = selection_bucket(state).get("selected_job_keys")
    if (isinstance(active_keys, list) and active_keys) or (
        isinstance(selected_keys, list) and selected_keys
    ):
        return resolve_selected_jobs(state)
    return [
        item for item in (jobs_state.get("results") or []) if isinstance(item, dict)
    ]


def short_list(value: Any, limit: int, item_limit: int) -> list[str]:
    if isinstance(value, str):
        value: list[Any] = [value]
    if not isinstance(value, list):
        return []
    return [
        item
        for item in (short_text(item, item_limit) for item in value[:limit])
        if item
    ]


def decode_mcp_result(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"text": short_text(raw)}

    if hasattr(raw, "artifact") and raw.artifact is not None:
        return decode_mcp_result(raw.artifact)

    if hasattr(raw, "content") and not isinstance(raw, (dict, list)):
        return decode_mcp_result(raw.content)

    if isinstance(raw, list):
        text_parts: list[str] = [
            item["text"]
            for item in raw
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if text_parts and len(text_parts) == len(raw):
            return decode_mcp_result("\n".join(text_parts))
        return [decode_mcp_result(item) for item in raw]

    if isinstance(raw, dict):
        for key in ("structuredContent", "structured_content", "artifact", "result"):
            if key in raw and raw[key] is not None:
                return decode_mcp_result(raw[key])
        if isinstance(raw.get("text"), str) and len(raw) <= 2:
            return decode_mcp_result(raw["text"])

    return raw


def extract_job_payloads(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        result: list[dict[str, Any]] = []
        for item in payload:
            result.extend(extract_job_payloads(item))
        return result

    if not isinstance(payload, dict):
        return []

    if isinstance(payload.get("job"), dict):
        return [payload["job"]]

    for key in ("jobs", "results", "items", "sites"):
        value: Any = payload.get(key)
        if isinstance(value, list):
            return extract_job_payloads(value)
        if isinstance(value, dict):
            nested: list[dict[str, Any]] = extract_job_payloads(value)
            if nested:
                return nested

    if any(key in payload for key in ("title", "job_title", "url", "job_url")):
        return [payload]

    return []


def _is_requirement_noise(line: str) -> bool:
    normalized: str = line.strip().casefold()
    if not normalized:
        return True
    if normalized == "...":
        return True
    if normalized.startswith("about "):
        return True
    return normalized in {
        "requirements",
        "job requirements",
        "qualification",
        "qualifications",
    }


def compact_job_card(job: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any]:
    def first(*keys: str) -> Any:
        for key in keys:
            if job.get(key) not in (None, "", []):
                return job[key]
        return None

    requirements: Any = first("requirements", "required_skills", "qualifications")
    if isinstance(requirements, list):
        requirements = [
            short_text(
                item.get("name") if isinstance(item, dict) else item,
                MAX_REQUIREMENT_CHARS,
            )
            for item in requirements[:MAX_REQUIREMENTS]
        ]
        requirements = [item for item in requirements if item]
    elif isinstance(requirements, str):
        requirement_lines: list[str] = [
            short_text(line, MAX_REQUIREMENT_CHARS)
            for line in requirements.splitlines()
            if str(line).strip()
        ]
        requirements = [
            line
            for line in requirement_lines[:MAX_REQUIREMENTS]
            if not _is_requirement_noise(line)
        ]
    else:
        requirements = []

    description = short_text(first("description", "summary", "content"))
    if not description and requirements:
        description = short_text(requirements[0], MAX_REQUIREMENT_CHARS)

    return {
        "title": short_text(first("title", "raw_title", "job_title")) or "Untitled job",
        "company": short_text(first("company", "company_name")) or "Unknown company",
        "location": short_text(first("location", "locations")),
        "url": short_text(first("url", "job_url", "link"), 1000),
        "salary": short_text(first("salary", "salary_range")),
        "posted_date": short_text(first("posted_date", "date_posted"), 160),
        "posted_at": short_text(first("posted_at"), 160),
        "work_type": short_text(first("work_type", "remote_type"), 80),
        "employment_type": short_text(first("employment_type", "job_type"), 80),
        "experience_level": short_text(first("experience_level", "seniority"), 80),
        "description": description,
        "requirements": requirements,
        "site": short_text(first("site") or envelope.get("site"), 80),
        "scrape_errors": short_list(envelope.get("errors"), 2, 200),
    }


def compact_scrape_response(raw: Any) -> dict[str, Any]:
    decoded: Any = decode_mcp_result(raw)
    envelope: dict[str, Any] = decoded if isinstance(decoded, dict) else {}
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, job in enumerate(extract_job_payloads(decoded)):
        card: dict[str, Any] = compact_job_card(job, envelope)
        key: str = job_selection_key({"job_card": card}, index)
        if key in seen:
            continue
        seen.add(key)
        cards.append(card)

    return {
        "total": len(cards),
        "truncated": bool(
            envelope.get("truncated")
            or envelope.get("is_truncated")
            or envelope.get("has_more")
        ),
        "cards": cards,
        "errors": short_list(envelope.get("errors"), 5, 300),
    }


def filter_scrape_args(tool: Any, request: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {
        key: value for key, value in request.items() if value not in (None, "", [], {})
    }
    schema: Any = getattr(tool, "args_schema", None)
    allowed: set[str]
    if hasattr(schema, "model_fields"):
        allowed = set(schema.model_fields)
    elif isinstance(schema, dict):
        allowed = set((schema.get("properties") or {}).keys())
    else:
        return args
    return {key: value for key, value in args.items() if key in allowed}


def message_text(message: Any) -> str:
    if isinstance(message, dict):
        content: Any = message.get("content", "")
    else:
        content = getattr(message, "content", message)
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return "".join(text_parts)
    return str(content)


def normalize_user_turn_text(content: str) -> str:
    text: str = (content or "").strip()
    if not text:
        return ""
    return text.replace(PDF_UPLOAD_MARKER, "").strip()


def message_role(message: Any) -> str:
    if isinstance(message, dict):
        explicit_role: str = str(message.get("role") or "").strip().lower()
        if explicit_role:
            return explicit_role
        msg_type: str = str(message.get("type") or "").strip().lower()
        if msg_type in {"human", "user"}:
            return "user"
        if msg_type in {"ai", "assistant"}:
            return "assistant"
        if msg_type == "tool":
            return "tool"
        return "user"
    msg_type = str(getattr(message, "type", "") or "").strip().lower()
    if msg_type == "tool":
        return "tool"
    return "user"


def _message_tool_calls(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("tool_calls")
    return getattr(message, "tool_calls", None)


def conversation_text_messages(state: ConversationState) -> list[dict[str, str]]:
    """Return model-readable turns while excluding tool protocol messages."""
    raw_messages: list[Any] = state.get("messages") or []
    if not raw_messages:
        return []

    last_user_index: int = -1
    for index, message in enumerate(raw_messages):
        if message_role(message) in {"human", "user"}:
            last_user_index = index

    start: int = last_user_index if last_user_index >= 0 else 0

    turns: list[dict[str, str]] = []
    for message in raw_messages[start:]:
        role_name: str = message_role(message)
        if role_name in {"tool"} or _message_tool_calls(message):
            continue
        content: str = message_text(message).strip()
        if not content:
            continue
        if role_name in {"human", "user"}:
            normalized: str = normalize_user_turn_text(content)
            if not normalized:
                continue
            turns.append({"role": "user", "content": normalized})
        elif role_name in {"ai", "assistant"}:
            if not should_retain_job_messages(state) and _appears_like_job_message(
                content
            ):
                continue
            turns.append({"role": "assistant", "content": content})
    return turns


def conversation_memory(state: ConversationState) -> dict[str, Any]:
    raw: Any = state.get("conversation_memory")
    if not isinstance(raw, dict):
        return {}
    try:
        return ConversationMemory.model_validate(raw).model_dump()
    except Exception:
        return {}


def conversation_memory_cursor(state: ConversationState) -> int:
    raw: Any = state.get("conversation_memory_cursor", 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def should_retain_job_messages(state: ConversationState) -> bool:
    return request_job_response(state) == "list"


def _appears_like_job_message(content: str) -> bool:
    lines: list[str] = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    numbered_item: bool = False
    for line in lines[:6]:
        first_segment: str = line.split(" ", 1)[0]
        if (
            first_segment
            and first_segment[-1] in {".", ")"}
            and first_segment[:-1].isdigit()
        ):
            numbered_item = True
            break
    has_job_fields: bool = any(
        "Location:" in line
        or "Posted:" in line
        or "Salary:" in line
        or "Description:" in line
        or "View Job" in line
        for line in lines
    )
    return bool(numbered_item and (has_job_fields or "http" in content.lower()))


def should_summarize_conversation(state: ConversationState) -> bool:
    turns: list[dict[str, str]] = conversation_text_messages(state)
    if len(turns) <= MAX_CONTEXT_MESSAGES:
        return False
    input_budget: int = context_input_budget()
    trigger_tokens: int = int(input_budget * CONTEXT_SUMMARY_TRIGGER_RATIO)
    return count_tokens_approximately(turns) >= trigger_tokens


CONVERSATION_SUMMARY_PROMPT: str = """Summarize durable conversational context from
the supplied turns into the structured memory schema. Omit raw documents and
transient result details. Do not invent facts.
"""


async def summarize_conversation_node(
    state: ConversationState,
    chat_model: ChatModel,
) -> dict[str, Any]:
    if not should_summarize_conversation(state):
        return {}

    turns: list[dict[str, str]] = conversation_text_messages(state)
    keep_from: int = max(0, len(turns) - MAX_CONTEXT_MESSAGES)
    cursor: int = min(conversation_memory_cursor(state), keep_from)
    older_turns: list[dict[str, str]] = turns[cursor:keep_from]
    if not older_turns:
        return {}

    payload: dict[str, Any] = {
        "existing_memory": conversation_memory(state),
        "new_turns": older_turns,
    }
    try:
        writer: Any = chat_model.structured(ConversationMemory)
        result: Any = await writer.ainvoke(
            [
                SystemMessage(content=CONVERSATION_SUMMARY_PROMPT),
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ]
        )
        memory: ConversationMemory = (
            result
            if isinstance(result, ConversationMemory)
            else ConversationMemory.model_validate(result)
        )
    except Exception:
        # Compaction is an optimization; a failed summary must not block a turn.
        return {}

    return {
        "conversation_memory": memory.model_dump(),
        "conversation_memory_cursor": keep_from,
    }


def last_user_text(state: ConversationState) -> str:
    for message in reversed(state.get("messages") or []):
        if message_role(message) in {"human", "user"}:
            return message_text(message)
    return ""


def router_recent_conversation(state: ConversationState) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in state.get("messages") or []:
        role_name: str = message_role(message)
        if role_name in {"tool"}:
            continue
        tool_calls: Any = (
            message.get("tool_calls")
            if isinstance(message, dict)
            else getattr(message, "tool_calls", None)
        )
        if tool_calls:
            continue
        content: str = short_text(message_text(message), MAX_ROUTER_HISTORY_CHARS)
        if not content:
            continue
        if role_name in {"human", "user"}:
            history.append({"role": "user", "content": content})
        elif role_name in {"ai", "assistant"}:
            history.append({"role": "assistant", "content": content})
    return history[-MAX_ROUTER_HISTORY_MESSAGES:]


def existing_cv_review(state: ConversationState) -> dict[str, Any] | None:
    review: Any = cv_bucket(state).get("review")
    if isinstance(review, dict):
        return review
    for doc in state_cv_documents(state):
        candidate: Any = doc.get("cv_review")
        if isinstance(candidate, dict):
            return candidate
    return None


def state_errors(state: ConversationState, extra: list[str] | None = None) -> list[str]:
    return list(state.get("errors") or []) + list(extra or [])


def compact_cv_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "matching_features": result.get("matching_features"),
        "validation_status": result.get("validation_status"),
        "validation_errors": short_list(result.get("validation_errors"), 5, 300),
        "warnings": short_list(result.get("warnings"), 8, 300),
        "confirmation_required": short_list(
            result.get("confirmation_required"),
            8,
            120,
        ),
    }


def compact_job_result(card: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    extract: dict[str, Any] = result.get("extract") or {}
    enriched_card: dict[str, Any] = dict(card)
    if not enriched_card.get("title"):
        enriched_card["title"] = (
            extract.get("normalized_title")
            or extract.get("raw_title")
            or "Pasted job description"
        )
    if not enriched_card.get("company"):
        enriched_card["company"] = extract.get("company") or ""
    if not enriched_card.get("url"):
        enriched_card["url"] = extract.get("job_url") or ""
    if not enriched_card.get("site"):
        enriched_card["site"] = extract.get("source") or ""
    return {
        "job_card": enriched_card,
        "matching_features": result.get("matching_features"),
        "validation_status": result.get("validation_status"),
        "validation_errors": short_list(result.get("validation_errors"), 5, 300),
        "warnings": short_list(result.get("warnings"), 8, 300),
    }


def sanitize_file_messages(messages: list[Any]) -> list[Any]:
    return [
        sanitized
        for message in messages
        if (sanitized := sanitize_file_message(message, stash_upload=False)) is not None
    ]


def pending_uploads_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    for message in reversed(messages):
        uploads: list[dict[str, Any]] = extract_uploads_from_message(message)
        if uploads:
            return uploads
        additional: Any = (
            message.get("additional_kwargs")
            if isinstance(message, dict)
            else getattr(message, "additional_kwargs", None)
        )
        if not isinstance(additional, dict):
            continue
        stashed_list: Any = additional.get("pending_cv_uploads")
        if isinstance(stashed_list, list) and stashed_list:
            return [item for item in stashed_list if isinstance(item, dict)]
        stashed: Any = additional.get("pending_cv_upload")
        if isinstance(stashed, dict):
            return [stashed]
    return []


def pending_upload_from_messages(messages: list[Any]) -> dict[str, Any] | None:
    uploads: list[dict[str, Any]] = pending_uploads_from_messages(messages)
    return uploads[0] if uploads else None


def clear_stashed_uploads(messages: list[Any]) -> list[Any]:
    cleared: list[Any] = []
    for message in messages:
        additional: Any = (
            message.get("additional_kwargs")
            if isinstance(message, dict)
            else getattr(message, "additional_kwargs", None)
        )
        if not isinstance(additional, dict):
            continue
        if (
            "pending_cv_upload" not in additional
            and "pending_cv_uploads" not in additional
        ):
            continue
        remaining: dict[str, Any] = {
            key: value
            for key, value in additional.items()
            if key not in {"pending_cv_upload", "pending_cv_uploads"}
        }
        if isinstance(message, dict):
            updated: dict[str, Any] = {**message, "additional_kwargs": remaining}
            cleared.append(updated)
            continue
        if hasattr(message, "model_copy"):
            cleared.append(message.model_copy(update={"additional_kwargs": remaining}))
            continue
        updated: dict[str, Any] = {
            "role": "user",
            "content": message_content_blocks(message)
            or getattr(message, "content", ""),
            "additional_kwargs": remaining,
        }
        message_id: Any = getattr(message, "id", None)
        if message_id:
            updated["id"] = message_id
        cleared.append(updated)
    return cleared


def ingest_input(state: ConversationState) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "pending_cv_upload": None,
        "pending_cv_uploads": None,
        "input_error": False,
        "cv": {
            "needs_extraction": False,
        },
        "routing": {
            "request": default_request_fields(),
            "targets": default_target_fields(),
            "plan": {
                "action": "respond",
                "reason": "awaiting_request_routing",
                "validation": "pending",
                "completed_actions": [],
            },
        },
        "execution": {
            "steps": [],
        },
        "response": None,
        "errors": [],
    }

    messages: list[Any] = list(state.get("messages") or [])
    message_updates: list[Any] = sanitize_file_messages(messages)
    cleared_uploads: list[Any] = clear_stashed_uploads(messages)
    if message_updates or cleared_uploads:
        merged_by_id: dict[str, Any] = {}
        ordered: list[Any] = []
        for message in [*message_updates, *cleared_uploads]:
            message_id: str = (
                str(message.get("id"))
                if isinstance(message, dict) and message.get("id") is not None
                else str(getattr(message, "id", "") or "")
            )
            if message_id and message_id in merged_by_id:
                continue
            if message_id:
                merged_by_id[message_id] = message
            ordered.append(message)
        updates["messages"] = ordered

    pending_uploads: Any = state.get("pending_cv_uploads")
    if not isinstance(pending_uploads, list) or not pending_uploads:
        single: Any = state.get("pending_cv_upload")
        if isinstance(single, dict):
            pending_uploads = [single]
        else:
            pending_uploads = pending_uploads_from_messages(messages)
    if not pending_uploads:
        return updates

    if any(
        isinstance(item, dict) and item.get("missing_bytes") for item in pending_uploads
    ):
        return {
            **updates,
            "input_error": True,
            "errors": ["cv_upload_payload_missing"],
        }

    existing_documents: list[dict[str, Any]] = state_cv_documents(state)
    remaining_slots: int = max(0, MAX_CV_DOCUMENTS - len(existing_documents))
    if remaining_slots == 0:
        return {
            **updates,
            "input_error": True,
            "errors": [f"cv_document_limit_reached:{MAX_CV_DOCUMENTS}"],
        }

    new_documents: list[dict[str, Any]] = []
    errors: list[str] = []
    for upload in pending_uploads[:remaining_slots]:
        try:
            new_documents.append(cv_document_from_upload(upload))
        except Exception as exc:
            filename: str = (
                str(upload.get("filename") or "cv.pdf")
                if isinstance(upload, dict)
                else "cv.pdf"
            )
            errors.append(f"cv_upload_failed:{filename}:{type(exc).__name__}")

    if not new_documents:
        return {
            **updates,
            "input_error": True,
            "errors": errors or ["cv_upload_no_readable_documents"],
        }

    documents: list[dict[str, Any]] = [*existing_documents, *new_documents]
    updates.update(
        {
            "cv": {
                "documents": documents,
                "comparison": None,
                **cv_needs_extraction_update(documents)["cv"],
            },
            "errors": errors,
        }
    )
    return updates


REQUEST_ROUTER_PROMPT: str = """Classify the latest request and resolve catalog
targets into the structured request schema. Use only supplied IDs, keys, state,
and conversation context. Preserve ambiguity instead of guessing. Do not choose
a workflow action or invent facts.
"""

WORKFLOW_PLANNER_PROMPT: str = """Select exactly one next backend action from the
structured request, resolved targets, and authoritative state facts. Select only
a missing step and do not repeat completed or reusable work. Do not invent facts.
"""


def routing_catalogs(state: ConversationState) -> dict[str, Any]:
    documents: list[dict[str, Any]] = state_cv_documents(state)
    cvs: list[dict[str, Any]] = [
        {
            "id": str(document.get("id") or ""),
            "filename": str(document.get("filename") or "cv.pdf"),
        }
        for document in documents[:MAX_CV_DOCUMENTS]
        if str(document.get("id") or "").strip()
    ]
    jobs: list[dict[str, Any]] = []
    job_keys: set[str] = set()
    for index, item in enumerate(jobs_bucket(state).get("results") or []):
        if not isinstance(item, dict):
            continue
        key: str = job_selection_key(item, index)
        job_keys.add(key)
        entry: dict[str, Any] = {"key": key, "row": index + 1}
        card: Any = item.get("job_card")
        if isinstance(card, dict):
            for field in ("title", "company"):
                if card.get(field):
                    entry[field] = card[field]
        jobs.append(entry)

    matches: list[dict[str, Any]] = []
    match_keys: set[str] = set()
    for item in jobs_bucket(state).get("matches") or []:
        if not isinstance(item, dict):
            continue
        key: str = str(item.get("job_key") or "").strip()
        if not key or key in match_keys:
            continue
        match_keys.add(key)
        entry = {"key": key, "row": len(matches) + 1}
        card = item.get("job_card")
        if isinstance(card, dict):
            for field in ("title", "company"):
                if card.get(field):
                    entry[field] = card[field]
        matches.append(entry)

    return {
        "cvs": cvs,
        "jobs": jobs,
        "matches": matches,
        "cv_ids": {item["id"] for item in cvs},
        "job_keys": job_keys,
        "match_keys": match_keys,
    }


def routing_cv_profiles(state: ConversationState) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for document in extracted_cv_documents(state)[:MAX_CV_DOCUMENTS]:
        features: Any = document.get("cv_features")
        if not isinstance(features, dict):
            continue
        summary: dict[str, Any] = cv_feature_summary(features)
        profiles.append(
            {
                "id": str(document.get("id") or ""),
                "filename": str(document.get("filename") or "cv.pdf"),
                **summary,
            }
        )
    return profiles


def request_router_context(state: ConversationState) -> dict[str, Any]:
    catalogs: dict[str, Any] = routing_catalogs(state)
    return {
        "latest_user_message": last_user_text(state)[:MAX_ROUTER_CHARS],
        "recent_conversation": router_recent_conversation(state),
        "conversation_memory": conversation_memory(state),
        "active_job_goal": active_job_goal(state),
        "cv_profiles": routing_cv_profiles(state),
        "cvs": catalogs["cvs"],
        "jobs": catalogs["jobs"],
        "matches": catalogs["matches"],
    }


def mapped_job_request(
    decision: RequestDecision,
) -> tuple[JobTask, JobResponse, JobSource, bool]:
    """Normalize the router's mapped job contract."""
    task: JobTask = decision.job.task
    response: JobResponse = decision.job.response
    source: JobSource = decision.job.source
    refresh: bool = bool(decision.job.refresh)

    if task == "extract" and source == "none":
        source = "pasted"
    if task == "search" and source == "none":
        source = "search"
    if task == "match" and source == "none":
        source = "existing"
    return task, response, source, refresh


def planner_context(state: ConversationState) -> dict[str, Any]:
    catalogs: dict[str, Any] = routing_catalogs(state)
    documents: list[dict[str, Any]] = state_cv_documents(state)
    extracted_ids: list[str] = [
        str(document.get("id") or "")
        for document in extracted_cv_documents(state)
        if str(document.get("id") or "").strip()
    ]
    targets: dict[str, Any] = selection_bucket(state)
    request: dict[str, Any] = request_values_from_state(state)
    return {
        "goal": {
            key: request.get(key)
            for key in (
                "goal",
                "goal_reason",
                "job_task",
                "job_response",
                "job_source",
                "assessment_requested",
                "score_requested",
                "role_constraints",
                "review_mode",
                "review_focus",
                "needs_cv_features",
                "is_follow_up",
                "role_source",
                "role_candidates",
            )
            if request.get(key) is not None
        },
        "targets": {
            key: targets.get(key)
            for key in (
                "cv_target_scope",
                "selected_cv_ids",
                "selected_job_keys",
                "job_target_scope",
                "unresolved_references",
                "targets_ambiguous",
            )
            if targets.get(key) is not None
        },
        "state_facts": {
            "cv_count": len(documents),
            "cv_ids": sorted(catalogs["cv_ids"]),
            "extracted_cv_ids": extracted_ids,
            "cv_needs_extraction": cvs_need_extraction(state),
            "valid_job_keys": sorted(catalogs["job_keys"]),
            "valid_match_keys": sorted(catalogs["match_keys"]),
            "job_count": len(catalogs["jobs"]),
            "match_count": len(catalogs["matches"]),
            "search_reusable": current_search_is_reusable(state),
            "match_reusable": action_result_is_reusable(
                state,
                "match_jobs",
                action_fingerprint("match_jobs", state),
            ),
            "cv_profiles": routing_cv_profiles(state),
            "active_job_goal": active_job_goal(state),
            "pending_match": pending_match_request(state),
            "completed_actions": completed_actions(state),
            "remaining_action_budget": max(
                0,
                MAX_AGENT_ACTIONS - len(completed_actions(state)),
            ),
            "refresh_requested": bool(request.get("refresh_requested")),
            "pasted_job_available": bool((request.get("job_input_text") or "").strip()),
        },
    }


def request_state_fields(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal": {
            "name": value.get("goal") or "general_question",
            "reason": value.get("goal_reason") or value.get("reason") or "",
            "confidence": value.get("decision_confidence", 1.0),
        },
        "job": {
            "task": value.get("job_task") or value.get("task") or "none",
            "response": value.get("job_response") or value.get("response") or "none",
            "source": value.get("job_source") or "none",
            "input": value.get("job_input_text"),
            "refresh": bool(value.get("job_refresh") or value.get("refresh_requested")),
            "scrape": dict(value.get("scrape_request") or {}),
        },
        "role": {
            "constraints": list(value.get("role_constraints") or []),
            "evidence": value.get("role_evidence"),
            "source": value.get("role_source") or "none",
            "candidates": list(value.get("role_candidates") or []),
        },
        "assessment": {
            "requested": bool(value.get("assessment_requested")),
            "detail_level": value.get("match_detail_level") or "summary",
        },
        "score": {
            "requested": bool(value.get("score_requested")),
            "visible": bool(value.get("show_score")),
        },
        "review": {
            "target_role": value.get("review_target_role"),
            "mode": value.get("review_mode") or "general",
            "focus": value.get("review_focus"),
            "reason": value.get("review_mode_reason"),
        },
        "cv": {
            "text_needed": bool(value.get("needs_cv_text")),
            "features_needed": bool(value.get("needs_cv_features")),
        },
        "context": {"follow_up": bool(value.get("is_follow_up"))},
    }


def target_state_fields(value: dict[str, Any]) -> dict[str, Any]:
    selected_cv_ids: list[str] = [
        str(item).strip()
        for item in (value.get("selected_cv_ids") or [])
        if str(item).strip()
    ]
    selected_job_keys: list[str] = [
        str(item).strip()
        for item in (value.get("selected_job_keys") or [])
        if str(item).strip()
    ]
    return {
        "cv": {
            "scope": value.get("cv_target_scope") or "none",
            "ids": selected_cv_ids,
        },
        "job": {
            "scope": value.get("job_target_scope") or "none",
            "keys": selected_job_keys,
        },
        "unresolved_references": list(value.get("unresolved_references") or []),
        "ambiguous": bool(value.get("targets_ambiguous")),
    }


async def request_router_node(
    state: ConversationState,
    chat_model: ChatModel,
) -> dict[str, Any]:
    latest: str = last_user_text(state)
    if not latest:
        decision = RequestDecision(
            goal="general_question",
            reason="no_user_message",
        )
    else:
        try:
            classifier: Any = chat_model.structured(RequestDecision)
            decision = await classifier.ainvoke(
                [
                    {"role": "system", "content": REQUEST_ROUTER_PROMPT},
                    {
                        "role": "user",
                        "content": "REQUEST ROUTING DATA ONLY:\n"
                        + json.dumps(
                            request_router_context(state),
                            ensure_ascii=False,
                        ),
                    },
                ]
            )
        except Exception as exc:
            decision = RequestDecision(
                goal="general_question",
                reason="request_router_failed",
            )
            return {
                "routing": {
                    "request": request_state_fields(
                        {"goal": decision.goal, "reason": decision.reason}
                    ),
                    "targets": default_target_fields(),
                },
                "errors": state_errors(
                    state,
                    [f"request_router_failed:{type(exc).__name__}"],
                ),
            }

    (
        mapped_job_task,
        mapped_job_response,
        mapped_job_source,
        mapped_job_refresh,
    ) = mapped_job_request(decision)
    mapped_job_refresh = bool(mapped_job_refresh and active_job_goal(state))
    catalogs: dict[str, Any] = routing_catalogs(state)
    selected_cv_ids: list[str] = []
    invalid_cv_ids: list[str] = []
    for item in decision.selected_cv_ids:
        value: str = str(item).strip()
        if not value:
            continue
        if value in catalogs["cv_ids"] and value not in selected_cv_ids:
            selected_cv_ids.append(value)
        elif value not in catalogs["cv_ids"]:
            invalid_cv_ids.append(value)

    selected_job_keys: list[str] = []
    invalid_job_keys: list[str] = []
    valid_job_keys: set[str] = catalogs["job_keys"] | catalogs["match_keys"]
    for item in decision.selected_job_keys or []:
        value = str(item).strip()
        if not value:
            continue
        if value in valid_job_keys and value not in selected_job_keys:
            selected_job_keys.append(value)
        elif value not in valid_job_keys:
            invalid_job_keys.append(value)

    unresolved: list[str] = [
        *decision.unresolved_references,
        *(f"unknown CV: {item}" for item in invalid_cv_ids),
        *(f"unknown job: {item}" for item in invalid_job_keys),
    ]
    targets_ambiguous: bool = bool(decision.targets_ambiguous or unresolved)
    if decision.job_target_scope == "all":
        selected_job_keys = []

    request_input: dict[str, Any] = {
        **decision.model_dump(),
        "goal_reason": decision.reason,
        "job_task": mapped_job_task,
        "job_response": mapped_job_response,
        "job_refresh": mapped_job_refresh,
        "job_source": (
            "pasted" if decision.job.task == "extract" else mapped_job_source
        ),
        "job_input_text": (latest if mapped_job_source == "pasted" else None),
        "scrape_request": decision.job.scrape.model_dump(exclude_none=True),
        "show_score": bool(decision.score_requested),
        "refresh_requested": bool(mapped_job_refresh),
        "role_candidates": [item.model_dump() for item in decision.role_candidates],
    }
    request: dict[str, Any] = request_state_fields(request_input)
    request_values_view: dict[str, Any] = request_values(request)
    role_constraints, role_evidence, role_source = normalize_job_role_fields(
        latest,
        job_task=mapped_job_task,
        role_constraints=request_values_view.get("role_constraints"),
        role_evidence=request_values_view.get("role_evidence"),
        role_source=request_values_view.get("role_source"),
        scrape_keywords=request_values_view.get("scrape_request", {}).get("keywords")
        if isinstance(request_values_view.get("scrape_request"), dict)
        else [],
    )
    request["role"] = {
        **dict(request.get("role") or {}),
        "constraints": role_constraints,
        "evidence": role_evidence,
        "source": role_source,
    }
    targets: dict[str, Any] = target_state_fields(
        {
            "cv_target_scope": decision.cv_target_scope,
            "selected_cv_ids": selected_cv_ids,
            "job_target_scope": decision.job_target_scope,
            "selected_job_keys": selected_job_keys,
            "unresolved_references": unresolved,
            "targets_ambiguous": targets_ambiguous,
        }
    )
    return {"routing": {"request": request, "targets": targets}}


async def workflow_planner_node(
    state: ConversationState,
    chat_model: ChatModel,
) -> dict[str, Any]:
    try:
        planner: Any = chat_model.structured(WorkflowPlan)
        plan: WorkflowPlan = await planner.ainvoke(
            [
                {"role": "system", "content": WORKFLOW_PLANNER_PROMPT},
                {
                    "role": "user",
                    "content": "WORKFLOW PLANNING DATA ONLY:\n"
                    + json.dumps(planner_context(state), ensure_ascii=False),
                },
            ]
        )
    except Exception as exc:
        plan = WorkflowPlan(
            action="respond",
            reason="workflow_planner_failed",
        )
        return {
            "routing": {
                "plan": {
                    "action": plan.action,
                    "reason": plan.reason,
                    "validation": "pending",
                },
            },
            "errors": state_errors(
                state,
                [f"workflow_planner_failed:{type(exc).__name__}"],
            ),
        }

    request: dict[str, Any] = request_values_from_state(state)
    if (
        plan.action == "search_jobs"
        and "search_jobs" in completed_actions(state)
        and not request.get("refresh_requested")
    ):
        if request.get("assessment_requested") and jobs_bucket(state).get("results"):
            plan = WorkflowPlan(
                action="match_jobs",
                reason="search_complete_assessment",
            )
        else:
            plan = WorkflowPlan(
                action="respond",
                reason="search_complete_present_results",
            )

    return {
        "routing": {
            "plan": {
                "action": plan.action,
                "reason": plan.reason,
                "validation": "pending",
            },
        }
    }


def legacy_decision_from_stages(state: ConversationState) -> RouteDecision:
    request: dict[str, Any] = request_values_from_state(state)
    targets: dict[str, Any] = selection_bucket(state)
    plan: dict[str, Any] = plan_bucket(state)
    router: dict[str, Any] = router_bucket(state)
    goal: str = str(request.get("goal") or "general_question")
    job_task: JobTask = request.get("job_task") or "none"
    job_response: JobResponse = request.get("job_response") or "none"
    job_refresh: bool = bool(request.get("job_refresh"))
    job_source: JobSource = request.get("job_source") or "none"
    selected_ids: list[str] = [
        str(item).strip()
        for item in (targets.get("selected_cv_ids") or [])
        if str(item).strip()
    ]
    return RouteDecision(
        route=(plan.get("action") or router.get("planned_action") or "respond"),
        reason=str(
            plan.get("reason")
            or request.get("goal_reason")
            or "workflow_action_selected"
        ),
        job_task=job_task,
        job_response=job_response,
        job_refresh=job_refresh,
        job_source=job_source,
        score_requested=bool(request.get("score_requested")),
        assessment_requested=bool(request.get("assessment_requested")),
        role_constraints=list(request.get("role_constraints") or []),
        role_evidence=request.get("role_evidence"),
        role_source=request.get("role_source") or "none",
        role_candidates=list(request.get("role_candidates") or []),
        job_target_scope=targets.get("job_target_scope") or "none",
        decision_confidence=float(
            request.get("decision_confidence")
            if request.get("decision_confidence") is not None
            else 1.0
        ),
        review_target_role=request.get("review_target_role"),
        review_mode=request.get("review_mode") or "general",
        review_focus=request.get("review_focus"),
        review_mode_reason=request.get("review_mode_reason"),
        needs_cv_text=bool(request.get("needs_cv_text")),
        needs_cv_features=bool(request.get("needs_cv_features"))
        or goal in {"review_cv", "compare_cvs", "extract_cv"}
        or job_task == "match",
        is_follow_up=bool(request.get("is_follow_up")),
        selected_cv_id=selected_ids[0] if len(selected_ids) == 1 else None,
        selected_job_keys=targets.get("selected_job_keys"),
        scrape_request=ScrapeRequest(**dict(request.get("scrape_request") or {})),
    )


def planned_action_validation_error(
    state: ConversationState,
    decision: RouteDecision,
) -> str | None:
    action: str = decision.route
    if action not in {"respond", *AGENT_ACTIONS}:
        return f"unknown_workflow_action:{action}"
    if action == "respond":
        return None
    if len(completed_actions(state)) >= MAX_AGENT_ACTIONS:
        return "action_limit_reached"

    selection: dict[str, Any] = selection_bucket(state)
    catalogs: dict[str, Any] = routing_catalogs(state)
    selected_ids: list[str] = [
        str(item).strip()
        for item in (selection.get("selected_cv_ids") or [])
        if str(item).strip()
    ]
    if any(item not in catalogs["cv_ids"] for item in selected_ids):
        return "cv_target_missing"
    selected_keys: list[str] = [
        str(item).strip()
        for item in (selection.get("selected_job_keys") or [])
        if str(item).strip()
    ]
    valid_job_keys: set[str] = catalogs["job_keys"] | catalogs["match_keys"]
    if any(item not in valid_job_keys for item in selected_keys):
        return "job_target_missing"
    if selection.get("targets_ambiguous"):
        return "target_ambiguous"
    if action == "extract_cv":
        if not state_cv_documents(state):
            return "cv_upload_required"
        if not cvs_need_extraction(state):
            return "cv_already_extracted"
    if action in {"review_cv", "compare_cvs", "match_jobs"}:
        if not state_cv_documents(state):
            return "cv_upload_required"
        extracted_ids: set[str] = {
            str(item.get("id") or "") for item in extracted_cv_documents(state)
        }
        target_ids: set[str] = set(selected_ids) if selected_ids else extracted_ids
        if not target_ids.issubset(extracted_ids):
            return "cv_extraction_required"
        if action == "review_cv" and len(target_ids) != 1:
            return "cv_review_target_count_invalid"
        if action == "compare_cvs" and len(target_ids) < 2:
            return "cv_comparison_target_count_invalid"
    if action == "extract_job" and not (selection.get("job_input_text") or "").strip():
        return "pasted_job_required"
    if action == "match_jobs":
        if selection.get("job_source") == "existing" and not catalogs["job_keys"]:
            return "existing_job_targets_missing"
        if selection.get("job_source") in {"search", "pasted"}:
            if not catalogs["job_keys"]:
                return "job_data_required_before_matching"
    if action in completed_actions(state) and not selection.get("refresh_requested"):
        if action in {"search_jobs", "match_jobs"}:
            if not action_result_is_reusable(
                state,
                action,
                action_fingerprint(action, state),
            ):
                return f"duplicate_action:{action}"
        else:
            return f"duplicate_action:{action}"
    return None


async def validate_plan_node(state: ConversationState) -> dict[str, Any]:
    decision: RouteDecision = legacy_decision_from_stages(state)
    base_error: str | None = planned_action_validation_error(state, decision)
    if base_error:
        return {
            "routing": {
                "plan": {
                    "action": "respond",
                    "reason": base_error,
                    "validation": "rejected",
                    "validation_error": base_error,
                    "planned_stages": [],
                },
            },
            "errors": state_errors(state, [base_error]),
        }

    selection: dict[str, Any] = selection_bucket(state)
    scrape_request: dict[str, Any] = decision.scrape_request.model_dump(
        exclude_none=True
    )
    if not scrape_request:
        scrape_request = dict(jobs_bucket(state).get("scrape_request") or {})
    jobs_update: dict[str, Any] = {
        "scrape_request": scrape_request,
    }
    try:
        route: RouteName = decision.route
        route_reason: str = decision.reason
        selection, jobs_update, policy = persist_planned_job_state(
            state,
            decision=decision,
            route=route,
            selection=selection,
            jobs_update=jobs_update,
        )
        request_state: dict[str, Any] = request_state_fields(selection)
        request_state["cv"] = {
            **dict(request_state.get("cv") or {}),
            "text_needed": bool(decision.needs_cv_text),
            "features_needed": bool(decision.needs_cv_features),
        }
        result: dict[str, Any] = {
            "routing": {
                "request": request_state,
                "targets": target_state_fields(selection),
                "plan": {
                    "action": route,
                    "reason": route_reason,
                    "validation": "accepted",
                    "validation_error": None,
                    "planned_stages": policy.get("planned_stages") or [],
                    "policy_reason": policy.get("policy_reason") or "",
                    "active_goal_id": policy.get("active_goal_id"),
                },
            },
            "jobs": jobs_update,
        }
        if route in {"search_jobs", "match_jobs"} and not (
            policy.get("planned_stages") or []
        ):
            result["execution"] = {
                "steps": [
                    *execution_steps(state),
                    build_skipped_execution_step(state, route),
                ]
            }
        return result
    except Exception as exc:
        reason: str = "plan_validation_failed"
        return {
            "routing": {
                "plan": {
                    "action": "respond",
                    "reason": reason,
                    "validation": "rejected",
                    "validation_error": reason,
                    "planned_stages": [],
                },
            },
            "errors": state_errors(
                state,
                [f"plan_validation_failed:{type(exc).__name__}"],
            ),
        }


def missing_cv_update(state: ConversationState) -> dict[str, Any]:
    return {
        "cv": {"needs_extraction": False},
        "errors": state_errors(
            state,
            ["cv_upload_required"],
        ),
    }


async def run_cv_subagent(state: ConversationState) -> dict[str, Any]:
    documents: list[dict[str, Any]] = state_cv_documents(state)
    if not documents:
        return missing_cv_update(state)

    updated_documents: list[dict[str, Any]] = []
    errors: list[str] = []
    for document in documents:
        cv_text: str = (document.get("cv_text") or "").strip()
        if not cv_text:
            updated_documents.append(dict(document))
            continue
        if document.get("cv_features"):
            updated_documents.append(dict(document))
            continue
        filename: str = str(document.get("filename") or "cv.pdf")
        try:
            result: dict[str, Any] = await cv_extraction_graph.ainvoke(
                {"cv_text": cv_text}
            )
            compact: dict[str, Any] = compact_cv_result(result)
            if compact.get("validation_status") != "valid":
                updated_documents.append(
                    {
                        **document,
                        "cv_result": compact,
                        "cv_features": None,
                    }
                )
                errors.append(f"cv_extraction_invalid:{filename}")
                continue
            updated_documents.append(
                {
                    **document,
                    "cv_result": compact,
                    "cv_features": compact.get("matching_features"),
                }
            )
        except Exception as exc:
            updated_documents.append(
                {
                    **document,
                    "cv_result": None,
                    "cv_features": None,
                }
            )
            errors.append(f"cv_extraction_failed:{filename}:{type(exc).__name__}")

    update: dict[str, Any] = {
        "cv": {
            "documents": updated_documents,
            **cv_needs_extraction_update(updated_documents)["cv"],
        }
    }
    if errors:
        update["errors"] = state_errors(state, errors)
    return update


async def handle_missing_cv(state: ConversationState) -> dict[str, Any]:
    return missing_cv_update(state)


async def run_cv_review(
    state: ConversationState,
    review_graph: Any,
) -> dict[str, Any]:
    documents: list[dict[str, Any]] = state_cv_documents(state)
    if not documents:
        return missing_cv_update(state)

    selection: dict[str, Any] = selection_bucket(state)
    target: dict[str, Any] | None = resolve_selected_cv(state)
    if target is None:
        return {
            "cv": {"review": None},
            "errors": state_errors(
                state,
                ["cv_extraction_required_for_review"],
            ),
        }

    try:
        result: dict[str, Any] = await review_graph.ainvoke(
            {
                "cv_text": (target.get("cv_text") or "").strip(),
                "cv_features": target.get("cv_features"),
                "target_role": selection.get("review_target_role"),
                "review_mode": selection.get("review_mode") or "general",
                "review_focus": selection.get("review_focus"),
            }
        )
        review: Any = result.get("cv_review")
        if not isinstance(review, dict):
            raise ValueError("CV review graph returned no review result")
        updated_documents: list[dict[str, Any]] = [
            {**doc, "cv_review": review}
            if doc.get("id") == target.get("id")
            else dict(doc)
            for doc in documents
        ]
        return {
            "cv": {
                "documents": updated_documents,
                "review": review,
                **cv_needs_extraction_update(updated_documents)["cv"],
            }
        }
    except Exception as exc:
        review: dict[str, Any] = {
            "status": "unavailable",
            "mode": selection.get("review_mode") or "general",
            "focus": selection.get("review_focus"),
            "target_role": selection.get("review_target_role"),
            "overall_score": None,
            "applicable_weight": 0,
            "criteria": [],
            "feedback": [],
            "deterministic_signals": {},
            "validation_errors": [f"cv_review_failed:{type(exc).__name__}"],
        }
        updated_documents: list[dict[str, Any]] = [
            {**doc, "cv_review": review}
            if doc.get("id") == target.get("id")
            else dict(doc)
            for doc in documents
        ]
        return {
            "cv": {
                "documents": updated_documents,
                "review": review,
                **cv_needs_extraction_update(updated_documents)["cv"],
            },
            "errors": state_errors(
                state,
                [f"cv_review_failed:{type(exc).__name__}"],
            ),
        }


COMPARE_CVS_PROMPT: str = """Compare multiple CV profiles from structured data.
Use only the supplied profiles. Do not invent facts.
"""


async def run_cv_comparison(
    state: ConversationState,
    chat_model: ChatModel | None = None,
) -> dict[str, Any]:
    documents: list[dict[str, Any]] = extracted_cv_documents(state)
    if len(documents) < 2:
        return {
            "cv": {"comparison": None},
            "errors": state_errors(
                state,
                ["cv_comparison_requires_two_documents"],
            ),
        }

    profiles: list[dict[str, Any]] = [
        {
            "filename": str(doc.get("filename") or "cv.pdf"),
            "features": cv_feature_summary(
                doc.get("cv_features")
                if isinstance(doc.get("cv_features"), dict)
                else None
            ),
        }
        for doc in documents[:MAX_CV_DOCUMENTS]
    ]
    comparer: Any = (chat_model or ChatModel.from_env()).structured(CvComparisonResult)
    try:
        comparison: CvComparisonResult = await comparer.ainvoke(
            [
                {"role": "system", "content": COMPARE_CVS_PROMPT},
                {
                    "role": "user",
                    "content": "CV PROFILES ONLY:\n"
                    + json.dumps(profiles, ensure_ascii=False),
                },
            ]
        )
        return {"cv": {"comparison": comparison.model_dump()}}
    except Exception as exc:
        return {
            "cv": {"comparison": None},
            "errors": state_errors(
                state,
                [f"cv_comparison_failed:{type(exc).__name__}"],
            ),
        }


def route_into_cv_subagent(state: ConversationState) -> str:
    documents: list[dict[str, Any]] = state_cv_documents(state)
    has_text: bool = any((doc.get("cv_text") or "").strip() for doc in documents)
    if not has_text:
        return "missing_cv"
    if cvs_need_extraction(state) or not any(
        doc.get("cv_features") for doc in documents
    ):
        return "extract_cv"
    route: RouteName | None = router_bucket(state).get("route")
    if route == "compare_cvs":
        return "compare_cvs"
    if route == "review_cv":
        return "review_cv"
    return "extract_cv"


def route_after_cv_extraction(state: ConversationState) -> str:
    """Continue to the requested CV action after prerequisite extraction."""
    route: RouteName | None = router_bucket(state).get("route")
    extracted_count: int = len(extracted_cv_documents(state))

    if route == "review_cv" and extracted_count >= 1:
        return "review_cv"
    if route == "compare_cvs" and extracted_count >= 2:
        return "compare_cvs"
    return "end"


def scrape_payload_from_card(
    card: dict[str, Any],
    request: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "keyword": ", ".join((request or {}).get("keywords") or []),
        "site": card.get("site"),
        "max_age_hours": (request or {}).get("max_age_hours"),
        "job": card,
        "errors": [],
    }


async def scrape_jobs_with_mcp(state: ConversationState) -> dict[str, Any]:
    existing_results: list[dict[str, Any]] = [
        item
        for item in (jobs_bucket(state).get("results") or [])
        if isinstance(item, dict)
    ]
    try:
        tool: Any = await load_scrape_jobs_tool()
        request: dict[str, Any] = dict(jobs_bucket(state).get("scrape_request") or {})
        raw: Any = await tool.ainvoke(filter_scrape_args(tool, request))
        compact: dict[str, Any] = compact_scrape_response(raw)
        job_results: list[dict[str, Any]]
        extraction_errors: list[str]
        job_results, extraction_errors = await extract_job_cards(
            compact["cards"], request
        )
        return {
            "jobs": {
                "scrape_total": compact["total"],
                "scrape_truncated": compact["truncated"],
                "results": merge_job_results(existing_results, job_results),
                "active_job_keys": [
                    job_selection_key(item, index)
                    for index, item in enumerate(job_results)
                ],
                "matches": [],
            },
            "errors": state_errors(
                state,
                compact["errors"] + extraction_errors,
            ),
        }
    except Exception as exc:
        return {
            "jobs": {
                "scrape_total": 0,
                "scrape_truncated": False,
                "active_job_keys": [],
                "matches": [],
            },
            "errors": state_errors(
                state,
                [f"job_scraping_failed:{type(exc).__name__}"],
            ),
        }


async def run_one_job_agent(
    card: dict[str, Any],
    request: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        result: dict[str, Any] = await job_extraction_graph.ainvoke(
            {"scraped_job": scrape_payload_from_card(card, request)}
        )
        return compact_job_result(card, result)
    except Exception as exc:
        return {
            "job_card": card,
            "matching_features": None,
            "validation_status": "invalid",
            "validation_errors": [f"{type(exc).__name__}: {exc}"],
            "warnings": [],
        }


def pasted_job_card(text: str) -> dict[str, Any]:
    return {
        "title": "",
        "company": "",
        "location": "",
        "url": "",
        "salary": "",
        "posted_date": "",
        "posted_at": "",
        "work_type": "",
        "employment_type": "",
        "experience_level": "",
        "description": text,
        "requirements": [],
        "site": "user_pasted",
        "scrape_errors": [],
    }


async def extract_pasted_job(state: ConversationState) -> dict[str, Any]:
    text: str = (selection_bucket(state).get("job_input_text") or "").strip()
    existing_results: list[dict[str, Any]] = [
        item
        for item in (jobs_bucket(state).get("results") or [])
        if isinstance(item, dict)
    ]
    if not text:
        return {
            "jobs": {
                "matches": [],
            },
            "errors": state_errors(
                state,
                ["pasted_job_description_missing"],
            ),
        }

    card: dict[str, Any] = pasted_job_card(text)
    result: dict[str, Any] = await run_one_job_agent(card, None)
    errors: list[str] = []
    if result.get("validation_status") != "valid":
        errors.append("pasted_job_extraction_failed")

    return {
        "jobs": {
            "scrape_total": jobs_bucket(state).get("scrape_total", 0),
            "scrape_truncated": jobs_bucket(state).get("scrape_truncated", False),
            "results": merge_job_results(existing_results, [result]),
            "active_job_keys": [job_selection_key(result, 0)],
            "matches": [],
        },
        "errors": state_errors(state, errors),
    }


async def extract_job_cards(
    cards: list[dict[str, Any]],
    request: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any]] = list(
        await asyncio.gather(*(run_one_job_agent(card, request) for card in cards))
    )
    errors: list[str] = [
        f"job_extraction_failed:{item['job_card'].get('title', 'job')}"
        for item in results
        if item.get("validation_status") != "valid"
    ]
    return results, errors


async def calculate_job_matches(state: ConversationState) -> dict[str, Any]:
    selected_cvs: list[dict[str, Any]] = resolve_selected_cvs(state)
    usable_cvs: list[dict[str, Any]] = [
        document
        for document in selected_cvs
        if isinstance(document.get("cv_result"), dict) and document.get("cv_features")
    ]
    if not usable_cvs:
        return {
            "jobs": {"matches": []},
            "errors": state_errors(
                state,
                ["cv_features_required_for_matching"],
            ),
        }

    selected_jobs: list[dict[str, Any]] = resolve_selected_jobs(state)
    if not selected_jobs:
        return {
            "jobs": {"matches": []},
            "errors": state_errors(
                state,
                ["job_targets_missing_for_matching"],
            ),
        }
    job_results: Any = jobs_bucket(state).get("results") or []
    job_key_by_id: dict[int, str] = {
        id(item): job_selection_key(item, index)
        for index, item in enumerate(job_results)
        if isinstance(item, dict)
    }
    errors: list[str] = []

    matches: list[dict[str, Any]] = []
    for document in usable_cvs:
        cv_result: Any = document.get("cv_result")
        cv_id: str = str(document.get("id") or "")
        cv_filename: str = str(document.get("filename") or "cv.pdf")
        for item in selected_jobs:
            job_title: str = str(item["job_card"].get("title") or "job")
            try:
                result: dict[str, Any] = await matching_score_graph.ainvoke(
                    {"cv_result": cv_result, "job_result": item}
                )
                score: Any = result.get("score")
                if score is not None:
                    matches.append(
                        {
                            "cv_id": cv_id,
                            "cv_filename": cv_filename,
                            "job_key": job_key_by_id.get(
                                id(item),
                                job_selection_key(item, 0),
                            ),
                            "job_card": item["job_card"],
                            "score": score,
                        }
                    )
            except Exception as exc:
                errors.append(
                    f"matching_failed:{cv_filename}:{job_title}:{type(exc).__name__}"
                )
                matches.append(
                    {
                        "cv_id": cv_id,
                        "cv_filename": cv_filename,
                        "job_key": job_key_by_id.get(
                            id(item),
                            job_selection_key(item, 0),
                        ),
                        "job_card": item["job_card"],
                        "score": {
                            "fit_verdict": "unknown",
                            "verdict_reason_code": "ASSESSMENT_UNAVAILABLE",
                            "review_reason_codes": [],
                        },
                    }
                )

    matches.sort(
        key=lambda item: (
            item["score"].get("normalized_score") is not None,
            item["score"].get("normalized_score") or -1,
        ),
        reverse=True,
    )
    return {
        "jobs": {"matches": matches, "pending_match": None},
        "errors": state_errors(state, errors),
    }


def public_presentation_intent(state: ConversationState) -> str:
    response: JobResponse = request_job_response(state)
    if response == "list":
        return "list"
    if response == "summary":
        return "summary"
    if response == "recommendation":
        return "recommendation"
    if response in {"explanation", "details"}:
        return response
    return "none"


def match_identity(item: dict[str, Any], index: int) -> str:
    key: str = str(item.get("job_key") or "").strip()
    if key:
        return key
    card: dict[str, Any] = (
        item.get("job_card") if isinstance(item.get("job_card"), dict) else {}
    )
    return str(card.get("url") or f"match:{index}").strip()


def search_role_label(state: ConversationState) -> str:
    goal: dict[str, Any] | None = active_job_goal(state)
    if goal is None:
        return ""
    return ", ".join(display_role_constraints(list(goal.get("role_constraints") or [])))


def public_job_cards(state: ConversationState) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for item in job_results_for_display(state):
        card: Any = item.get("job_card") if isinstance(item, dict) else None
        slim: dict[str, Any] = slim_job_card(
            card if isinstance(card, dict) else None,
            include_description=False,
        )
        slim.pop("company", None)
        if slim:
            jobs.append(slim)
    return jobs


def job_cards_for_current_request(state: ConversationState) -> list[dict[str, Any]]:
    response: JobResponse = request_job_response(state)
    if response != "list":
        return []
    route: Any = router_bucket(state).get("route")
    if route in {"search_jobs", "extract_job", "respond"}:
        return public_job_cards(state)
    return []


def public_assessment(
    state: ConversationState,
    *,
    show_score: bool,
) -> dict[str, Any] | None:
    response: JobResponse = request_job_response(state)
    if response not in {"summary", "recommendation", "explanation", "details"}:
        return None
    matches: list[dict[str, Any]] = [
        item
        for item in (jobs_bucket(state).get("matches") or [])
        if isinstance(item, dict)
    ]
    if response == "summary":
        if not matches and "match_jobs" not in completed_actions(state):
            return None
        detail_level: Any = selection_bucket(state).get("match_detail_level")
        if detail_level not in {"summary", "full"}:
            detail_level = "summary"
        return match_presentation.build_public_match_summary(
            matches,
            show_score=show_score,
            detail_level=detail_level,
        )

    detail_level: Any = selection_bucket(state).get("match_detail_level")
    if detail_level not in {"summary", "full"}:
        detail_level = "summary"

    if response == "recommendation":
        if not matches:
            return {"status": "unavailable"}
        return match_presentation.build_public_match_recommendation(
            matches,
            show_score=show_score,
            detail_level=detail_level,
        )

    selected_keys: Any = selection_bucket(state).get("selected_job_keys")
    wanted: list[str] = [
        str(key).strip() for key in (selected_keys or []) if str(key).strip()
    ]
    if len(wanted) != 1:
        return None
    selected = [
        item
        for index, item in enumerate(matches)
        if match_identity(item, index) == wanted[0]
    ]
    if not selected:
        return None
    return match_presentation.build_public_match_selected(
        selected,
        selected_key=wanted[0],
        show_score=show_score,
        detail_level=detail_level,
    )


def public_review_payload(state: ConversationState) -> dict[str, Any] | None:
    review: Any = cv_bucket(state).get("review")
    if not isinstance(review, dict):
        selected: dict[str, Any] | None = resolve_selected_cv(state)
        if isinstance(selected, dict):
            review = selected.get("cv_review")
    slim: dict[str, Any] | None = slim_review_result(
        review if isinstance(review, dict) else None
    )
    if slim is None or not slim.get("feedback"):
        return None
    payload: dict[str, Any] = {
        "mode": slim.get("mode"),
        "focus": slim.get("focus"),
        "target_role": slim.get("target_role"),
        "feedback": slim.get("feedback") or [],
    }
    if slim.get("overall_score") is not None:
        payload["overall_score"] = slim.get("overall_score")
        payload["score_scale"] = "0-100"
    return payload


def public_comparison_payload(state: ConversationState) -> dict[str, Any] | None:
    return slim_comparison_result(
        cv_bucket(state).get("comparison")
        if isinstance(cv_bucket(state).get("comparison"), dict)
        else None
    )


def public_extracted_job(state: ConversationState) -> dict[str, Any] | None:
    route: Any = router_bucket(state).get("route")
    if route not in {"extract_job", "respond"}:
        return None
    results: list[dict[str, Any]] = job_results_for_display(state)
    if not results:
        return None
    selected_keys: Any = selection_bucket(state).get("selected_job_keys")
    if route == "respond" and len(results) > 1 and not selected_keys:
        return None
    latest: dict[str, Any] = results[-1]
    card: dict[str, Any] = slim_job_card(
        latest.get("job_card") if isinstance(latest.get("job_card"), dict) else None
    )
    extract: Any = latest.get("extract")
    if not isinstance(extract, dict):
        return card or None
    payload: dict[str, Any] = dict(card)
    raw_content: str = short_text(extract.get("raw_content") or "", 6000)
    if raw_content:
        payload["description"] = raw_content
    responsibilities: list[str] = [
        str(value).strip()
        for value in (extract.get("responsibilities") or [])
        if str(value).strip()
    ][:12]
    if responsibilities:
        payload["responsibilities"] = responsibilities
    for source_key, public_key in (
        ("required_skills", "required_skills"),
        ("preferred_skills", "preferred_skills"),
    ):
        names: list[str] = []
        for value in extract.get(source_key) or []:
            if isinstance(value, dict):
                name: str = str(
                    value.get("name")
                    or value.get("normalized_name")
                    or value.get("raw_name")
                    or ""
                ).strip()
            else:
                name = str(value or "").strip()
            if name and name not in names:
                names.append(name)
        if names:
            payload[public_key] = names[:12]
    return payload or None


def performed_actions_payload(state: ConversationState) -> list[dict[str, Any]]:
    performed: list[dict[str, Any]] = []
    for step in execution_steps(state):
        destination: Any = step.get("to")
        if not isinstance(destination, dict):
            continue
        item: dict[str, Any] = {
            "action": step.get("action"),
            "status": step.get("status"),
            "args": dict(destination.get("args") or {}),
            "result": dict(destination.get("result") or {}),
        }
        context: Any = step.get("context")
        if isinstance(context, dict) and context.get("args_source"):
            item["args_source"] = context["args_source"]
        if step.get("error"):
            item["error"] = step["error"]
        performed.append(item)
    return performed


def presentation_payload(state: ConversationState) -> dict[str, Any]:
    router: dict[str, Any] = router_bucket(state)
    selection: dict[str, Any] = selection_bucket(state)
    request: dict[str, Any] = request_values_from_state(state)
    show_score: bool = bool(
        selection.get("show_score") or selection.get("score_requested")
    )
    jobs: list[dict[str, Any]] = job_cards_for_current_request(state)
    payload: dict[str, Any] = {
        "intent": public_presentation_intent(state),
        "show_score": show_score,
        "assessment_requested": bool(request.get("assessment_requested")),
        "input_error": bool(state.get("input_error")),
        "conversation_memory": conversation_memory(state),
    }
    performed_actions: list[dict[str, Any]] = performed_actions_payload(state)
    if performed_actions:
        payload["performed_actions"] = performed_actions
    role: str = search_role_label(state)
    if not role:
        role = ", ".join(
            display_role_constraints(list(request.get("role_constraints") or []))
        )
    if role:
        payload["role"] = role
    payload["job_list"] = jobs
    payload["job_list_count"] = len(jobs)
    missing_prerequisites: list[str] = []
    if request.get("job_task") == "match":
        if not extracted_cv_documents(state):
            missing_prerequisites.append("cv")
        if not job_results_for_display(state):
            missing_prerequisites.append("jobs")
    if jobs or missing_prerequisites:
        payload["available_job_count"] = len(job_results_for_display(state))
    if jobs and jobs_bucket(state).get("scrape_truncated"):
        payload["more_jobs_may_exist"] = True
    if missing_prerequisites:
        payload["missing_prerequisites"] = missing_prerequisites
    assessment: dict[str, Any] | None = public_assessment(state, show_score=show_score)
    if assessment is not None:
        payload["assessment"] = assessment
    review: dict[str, Any] | None = public_review_payload(state)
    if review is not None:
        payload["review"] = review
    comparison: dict[str, Any] | None = public_comparison_payload(state)
    if comparison is not None:
        payload["comparison"] = comparison
    extracted: dict[str, Any] | None = public_extracted_job(state)
    if extracted is not None:
        payload["extracted_job"] = extracted
    profiles: list[dict[str, Any]] = []
    for document in state_cv_documents(state):
        summary: dict[str, Any] = cv_feature_summary(
            document.get("cv_features")
            if isinstance(document.get("cv_features"), dict)
            else None
        )
        if not summary:
            continue
        profiles.append(
            {
                "filename": str(document.get("filename") or "cv.pdf"),
                **summary,
            }
        )
    if profiles:
        payload["cv_profiles"] = profiles

    needs_cv_text: bool = bool(
        selection.get("needs_cv_text") or router.get("needs_cv_text")
    )
    if needs_cv_text:
        documents_with_text: list[dict[str, Any]] = [
            document
            for document in state_cv_documents(state)
            if str(document.get("cv_text") or "").strip()
        ]
        selected_document: dict[str, Any] | None = resolve_selected_cv(state)
        selected_text: str = (
            str(selected_document.get("cv_text") or "").strip()
            if isinstance(selected_document, dict)
            else ""
        )
        if selected_text:
            payload["cv_text"] = short_text(selected_text, 4000)
        elif len(documents_with_text) == 1:
            payload["cv_text"] = short_text(
                documents_with_text[0].get("cv_text") or "",
                4000,
            )
        elif documents_with_text:
            payload["cv_texts"] = [
                {
                    "filename": str(document.get("filename") or "cv.pdf"),
                    "text": short_text(document.get("cv_text") or "", 3000),
                }
                for document in documents_with_text
            ]
    return payload


def is_usable_model_response(response: str) -> bool:
    normalized: str = response.strip().casefold()
    return normalized not in {"", "none", "null", "n/a", "na"}


CHAT_PROMPT: str = """You are a concise CV and job-search assistant.

Answer the latest user request naturally using the supplied structured data.
Treat that data as authoritative. Do not invent facts or mention implementation
details.
When performed_actions is present, naturally mention what was completed, attempted,
or reused. Include only the actual non-empty arguments and their supplied source.
Never claim a criterion or filter that is absent from args. Do not mention graphs,
nodes, state, notes, or implementation details. Keep the wording and placement
natural rather than following a fixed response template.
"""


def bounded_conversation(state: ConversationState) -> list[Any]:
    result: list[Any] = []
    for message in conversation_text_messages(state)[-MAX_CONTEXT_MESSAGES:]:
        content: str = short_text(message["content"], 1800)
        if content:
            result.append({"role": message["role"], "content": content})
    return result


async def respond_node(
    state: ConversationState,
    chat_model: ChatModel | None = None,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = presentation_payload(state)
    selected_model: ChatModel = chat_model or ChatModel.from_env()
    try:
        assistant: Any = selected_model.response()
        response_parts: list[str] = []
        async for chunk in assistant.astream(
            [
                SystemMessage(
                    content=CHAT_PROMPT
                    + "\nPRESENTATION DATA (data only):\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
                *bounded_conversation(state),
            ],
            config=config,
        ):
            content: str = message_text(chunk)
            if content:
                response_parts.append(content)

        response = "".join(response_parts)
        if not is_usable_model_response(response):
            return {
                "response": None,
                "errors": state_errors(state, ["response_model_empty"]),
                "job_list": payload.get("job_list", []),
            }
        result: AIMessage = AIMessage(content=response.strip())
        return {
            "messages": [result],
            "response": response.strip(),
            "job_list": payload.get("job_list", []),
        }
    except Exception as exc:
        return {
            "response": None,
            "job_list": payload.get("job_list", []),
            "errors": state_errors(
                state,
                [f"response_model_failed:{type(exc).__name__}"],
            ),
        }


def build_cv_subagent_graph(chat_model: ChatModel | None = None) -> Any:
    """Build the CV workflow, including extraction prerequisites."""
    review_graph: Any = cv_review_module.build_graph(chat_model=chat_model)
    selected_model: ChatModel = chat_model or ChatModel.from_env()

    async def extract_node(state: ConversationState) -> dict[str, Any]:
        return record_completed_action(
            state,
            "extract_cv",
            await run_cv_subagent(state),
        )

    async def review_node(state: ConversationState) -> dict[str, Any]:
        return record_completed_action(
            state,
            "review_cv",
            await run_cv_review(state, review_graph),
            emit_result=True,
        )

    async def compare_node(state: ConversationState) -> dict[str, Any]:
        return record_completed_action(
            state,
            "compare_cvs",
            await run_cv_comparison(state, selected_model),
            emit_result=True,
        )

    builder: StateGraph = StateGraph(ConversationState)
    builder.add_node("extract_cv", extract_node)
    builder.add_node("review_cv", review_node)
    builder.add_node("compare_cvs", compare_node)
    builder.add_node("missing_cv", handle_missing_cv)
    builder.set_conditional_entry_point(
        route_into_cv_subagent,
        {
            "extract_cv": "extract_cv",
            "review_cv": "review_cv",
            "compare_cvs": "compare_cvs",
            "missing_cv": "missing_cv",
        },
    )
    builder.add_conditional_edges(
        "extract_cv",
        route_after_cv_extraction,
        {
            "review_cv": "review_cv",
            "compare_cvs": "compare_cvs",
            "end": END,
        },
    )
    builder.add_edge("review_cv", END)
    builder.add_edge("compare_cvs", END)
    builder.add_edge("missing_cv", END)
    return builder.compile(name="cv_subagent")


def route_into_job_subagent(state: ConversationState) -> str:
    router: dict[str, Any] = router_bucket(state)
    selection: dict[str, Any] = selection_bucket(state)
    route: RouteName | None = router.get("route")
    planned_stages: list[str] = plan_bucket(state).get("planned_stages") or []
    if not planned_stages:
        return "end"
    if route == "extract_job":
        return "extract_pasted_job" if "extract_job" in planned_stages else "end"
    if route == "search_jobs":
        return "scrape_jobs" if "scrape_jobs" in planned_stages else "end"
    if route == "match_jobs":
        if "scrape_jobs" in planned_stages:
            if (
                not selection.get("refresh_requested")
                and current_search_is_reusable(state)
                and resolve_selected_jobs(state)
            ):
                planned_stages = [
                    item for item in planned_stages if item != "scrape_jobs"
                ]
                if "match_jobs" not in planned_stages:
                    return "end"
                if action_result_is_reusable(
                    state,
                    "match_jobs",
                    action_fingerprint("match_jobs", state),
                ):
                    return "end"
            return "scrape_jobs"
        if "match_jobs" in planned_stages:
            return "match_jobs"
        if (
            selection.get("job_source") == "pasted"
            and "extract_job" in planned_stages
            and not jobs_bucket(state).get("results")
        ):
            return "extract_pasted_job"
    return "end"


def route_after_search_or_extract(state: ConversationState) -> str:
    router: dict[str, Any] = router_bucket(state)
    route: RouteName | None = router.get("route")
    planned_stages: list[str] = plan_bucket(state).get("planned_stages") or []
    if route != "match_jobs" or "match_jobs" not in planned_stages:
        return "end"
    jobs_state: dict[str, Any] = jobs_bucket(state)
    active_keys: Any = jobs_state.get("active_job_keys")
    if isinstance(active_keys, list) and not active_keys:
        return "end"
    if not jobs_state.get("results"):
        return "end"
    return "match_jobs"


def build_job_subagent_graph() -> Any:
    """Build the job workflow, including search/extract prerequisites for matching."""

    async def search_node(state: ConversationState) -> dict[str, Any]:
        return record_completed_action(
            state,
            "search_jobs",
            await scrape_jobs_with_mcp(state),
            emit_result=True,
        )

    async def extract_node(state: ConversationState) -> dict[str, Any]:
        return record_completed_action(
            state,
            "extract_job",
            await extract_pasted_job(state),
            emit_result=True,
        )

    async def match_node(state: ConversationState) -> dict[str, Any]:
        return record_completed_action(
            state,
            "match_jobs",
            await calculate_job_matches(state),
            emit_result=True,
        )

    builder: StateGraph = StateGraph(ConversationState)
    builder.add_node("scrape_jobs", search_node)
    builder.add_node("extract_pasted_job", extract_node)
    builder.add_node("match_jobs", match_node)

    builder.set_conditional_entry_point(
        route_into_job_subagent,
        {
            "extract_pasted_job": "extract_pasted_job",
            "scrape_jobs": "scrape_jobs",
            "match_jobs": "match_jobs",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "scrape_jobs",
        route_after_search_or_extract,
        {
            "match_jobs": "match_jobs",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "extract_pasted_job",
        route_after_search_or_extract,
        {
            "match_jobs": "match_jobs",
            "end": END,
        },
    )
    builder.add_edge("match_jobs", END)
    return builder.compile(name="job_subagent")


def route_after_router(state: ConversationState) -> str:
    actions: list[str] = completed_actions(state)
    if len(actions) >= MAX_AGENT_ACTIONS:
        return "respond"
    router: dict[str, Any] = router_bucket(state)
    selection: dict[str, Any] = selection_bucket(state)
    route: RouteName = router.get("route") or "respond"
    needs_extraction: bool = cvs_need_extraction(state)
    if needs_extraction and intent_requires_cv_features(state):
        if route in {"review_cv", "compare_cvs"} and "extract_cv" not in actions:
            return route
        return "respond" if "extract_cv" in actions else "extract_cv"
    if route == "extract_cv" and not needs_extraction:
        return "respond"
    if route in actions:
        return "respond"
    if route == "extract_cv":
        return "extract_cv"
    if route == "review_cv":
        return "review_cv"
    if route == "compare_cvs":
        if len(extracted_cv_documents(state)) < 2:
            return "respond"
        return "compare_cvs"
    if route == "extract_job":
        return "extract_job"
    if route == "search_jobs":
        return "search_jobs"
    if route == "match_jobs":
        if jobs_bucket(state).get("results"):
            return "match_jobs"
        if selection.get("job_source") in {"pasted", "search"}:
            return "match_jobs"
        return "respond"
    return "respond"


def route_after_plan_validation(state: ConversationState) -> str:
    route: Any = router_bucket(state).get("route") or "respond"
    plan: dict[str, Any] = plan_bucket(state)
    if route in {"search_jobs", "match_jobs"} and not (
        plan.get("planned_stages") or []
    ):
        return "respond"
    if route in AGENT_ACTIONS or route == "respond":
        return str(route)
    return "respond"


def route_after_agent_action(state: ConversationState) -> str:
    return (
        "respond"
        if len(completed_actions(state)) >= MAX_AGENT_ACTIONS
        else "workflow_planner"
    )


def route_after_cv_subagent(state: ConversationState) -> str:
    documents: list[dict[str, Any]] = state_cv_documents(state)
    has_text: bool = any((doc.get("cv_text") or "").strip() for doc in documents)
    if not has_text:
        return "respond"
    return route_after_agent_action(state)


def route_after_job_subagent(state: ConversationState) -> str:
    router: dict[str, Any] = router_bucket(state)
    selection: dict[str, Any] = selection_bucket(state)
    route: Any = router.get("route")
    plan: dict[str, Any] = plan_bucket(state)
    staged: list[str] = plan.get("planned_stages") or []
    response_type: JobResponse = request_job_response(state)
    if route == "match_jobs":
        if "match_jobs" not in staged or "match_jobs" in completed_actions(state):
            return "respond"
    if route == "search_jobs":
        if response_type != "list":
            return "respond"
        if "search_jobs" in completed_actions(state) and not selection.get(
            "assessment_requested"
        ):
            return "respond"
    return route_after_agent_action(state)


def build_graph(
    *,
    checkpointer: Any | None = None,
    chat_model: ChatModel | None = None,
) -> Any:
    selected_model: ChatModel = chat_model or ChatModel.from_env()

    async def request_router(state: ConversationState) -> dict[str, Any]:
        return await request_router_node(state, selected_model)

    async def workflow_planner(state: ConversationState) -> dict[str, Any]:
        return await workflow_planner_node(state, selected_model)

    async def response_node(
        state: ConversationState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        return await respond_node(state, selected_model, config=config)

    async def summarize_node(state: ConversationState) -> dict[str, Any]:
        return await summarize_conversation_node(state, selected_model)

    builder: StateGraph = StateGraph(ConversationState, input_schema=StudioInput)
    builder.add_node("ingest_input", ingest_input)
    builder.add_node("summarize_conversation", summarize_node)
    builder.add_node("request_router", request_router)
    builder.add_node("workflow_planner", workflow_planner)
    builder.add_node("validate_plan", validate_plan_node)
    builder.add_node("cv_subagent", build_cv_subagent_graph(selected_model))
    builder.add_node("job_subagent", build_job_subagent_graph())
    builder.add_node("respond", response_node)

    builder.add_edge(START, "ingest_input")
    builder.add_edge("ingest_input", "summarize_conversation")
    builder.add_edge("summarize_conversation", "request_router")
    builder.add_edge("request_router", "workflow_planner")
    builder.add_edge("workflow_planner", "validate_plan")
    builder.add_conditional_edges(
        "validate_plan",
        route_after_plan_validation,
        {
            "extract_cv": "cv_subagent",
            "review_cv": "cv_subagent",
            "compare_cvs": "cv_subagent",
            "extract_job": "job_subagent",
            "search_jobs": "job_subagent",
            "match_jobs": "job_subagent",
            "respond": "respond",
        },
    )
    builder.add_conditional_edges(
        "cv_subagent",
        route_after_cv_subagent,
        {
            "workflow_planner": "workflow_planner",
            "respond": "respond",
        },
    )
    builder.add_conditional_edges(
        "job_subagent",
        route_after_job_subagent,
        {
            "workflow_planner": "workflow_planner",
            "respond": "respond",
        },
    )
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer)


graph: Any = build_graph()
