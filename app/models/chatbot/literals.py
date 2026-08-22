"""Literal type contracts for the chatbot graph.

String members stay inline: ``Literal`` is a type-level construct and cannot
reference module constants.
"""

from __future__ import annotations

from typing import Literal

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
ExecutionStatus = Literal["completed", "failed", "skipped"]
DetailLevel = Literal["summary", "full"]
