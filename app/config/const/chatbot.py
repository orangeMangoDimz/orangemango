"""Constants for the conversational CV and job-search graph."""

from __future__ import annotations

from pathlib import Path

from app.config.const.chat import MAX_CV_FILE_BYTES

# [Filesystem layout]
# const -> config -> app -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
STUDIO_ROOT: Path = PROJECT_ROOT / "studio"
CV_GRAPH_PATH: Path = STUDIO_ROOT / "cv-extraction" / "graph.py"
JOB_GRAPH_PATH: Path = STUDIO_ROOT / "job-extraction" / "graph.py"
MATCHING_SCORE_GRAPH_PATH: Path = STUDIO_ROOT / "matching-score" / "graph.py"
CV_REVIEW_GRAPH_PATH: Path = STUDIO_ROOT / "cv-review" / "graph.py"

# [Child graph module names]
# These must match the original module names so the app graph stays a drop-in
# replacement for the studio graph inside a single interpreter.
MODULE_CV_EXTRACTION: str = "orangemango_chatbot_cv_extraction"
MODULE_JOB_EXTRACTION: str = "orangemango_chatbot_job_extraction"
MODULE_MATCHING_SCORE: str = "orangemango_chatbot_matching_score"
MODULE_CV_REVIEW: str = "orangemango_chatbot_cv_review"

# [Job scraper MCP]
JOB_SUBAGENT_MCP_URL: str = "http://localhost:8080/mcp"
JOB_SUBAGENT_MCP_NAME: str = "job_scraper"
JOB_SUBAGENT_MCP_TRANSPORT: str = "http"
SCRAPE_JOBS_TOOL_NAME: str = "scrape_jobs"

# [Payload limits]
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
MIN_CONTEXT_INPUT_BUDGET: int = 1024

# [Context window environment variables]
ENV_CONTEXT_WINDOW_TOKENS: str = "OPENAI_CONTEXT_WINDOW_TOKENS"
ENV_CONTEXT_OUTPUT_RESERVE_TOKENS: str = "OPENAI_CONTEXT_OUTPUT_RESERVE_TOKENS"
ENV_CONTEXT_PROMPT_RESERVE_TOKENS: str = "OPENAI_CONTEXT_PROMPT_RESERVE_TOKENS"

# [Workflow limits]
MAX_AGENT_ACTIONS: int = 4
MAX_CV_DOCUMENTS: int = 5
MAX_ACTION_RESULT_CHARS: int = 6000
PDF_UPLOAD_MARKER: str = "[PDF CV uploaded separately]"

# [Main graph nodes]
NODE_PARENT_PLANNER: str = "parent_planner"
NODE_CV_SUBAGENT: str = "cv_subagent"
NODE_JOB_SUBAGENT: str = "job_subagent"
NODE_RESPOND: str = "respond"

# [CV subagent nodes]
NODE_EXTRACT_CV: str = "extract_cv"
NODE_REVIEW_CV: str = "review_cv"
NODE_COMPARE_CVS: str = "compare_cvs"
NODE_CV_PLANNER: str = "cv_planner"

# [Job subagent nodes]
NODE_SCRAPE_JOBS: str = "scrape_jobs"
NODE_EXTRACT_PASTED_JOB: str = "extract_job"
NODE_MATCH_JOBS: str = "match_jobs"
NODE_JOB_PLANNER: str = "job_planner"

# [Route and action names]
ROUTE_RESPOND: str = "respond"
ROUTE_EXTRACT_CV: str = "extract_cv"
ROUTE_REVIEW_CV: str = "review_cv"
ROUTE_COMPARE_CVS: str = "compare_cvs"
ROUTE_EXTRACT_JOB: str = "extract_job"
ROUTE_SEARCH_JOBS: str = "search_jobs"
ROUTE_MATCH_JOBS: str = "match_jobs"

# [Conditional edge sentinel]
BRANCH_END: str = "end"

CV_FEATURE_INTENTS: frozenset[str] = frozenset(
    {
        ROUTE_EXTRACT_CV,
        ROUTE_REVIEW_CV,
        ROUTE_COMPARE_CVS,
        ROUTE_MATCH_JOBS,
    }
)

# [Runtime request projection keys]
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
