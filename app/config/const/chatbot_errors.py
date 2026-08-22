"""Error, status, and reason codes emitted by the chatbot graph.

These are structured machine codes surfaced to the language model as data. They
are never user-facing copy; the model decides the final wording.
"""

from __future__ import annotations

# [Prefixed error codes]
# Each is concatenated with a filename, action, exception name, or limit.
ERROR_CV_UPLOAD_FAILED: str = "cv_upload_failed:"
ERROR_CV_DOCUMENT_LIMIT_REACHED: str = "cv_document_limit_reached:"
ERROR_CV_EXTRACTION_FAILED: str = "cv_extraction_failed:"
ERROR_CV_EXTRACTION_INVALID: str = "cv_extraction_invalid:"
ERROR_CV_REVIEW_FAILED_PREFIX: str = "cv_review_failed:"
ERROR_CV_COMPARISON_FAILED_PREFIX: str = "cv_comparison_failed:"
ERROR_JOB_EXTRACTION_FAILED: str = "job_extraction_failed:"
ERROR_JOB_SCRAPING_FAILED: str = "job_scraping_failed:"
ERROR_MATCHING_FAILED: str = "matching_failed:"
ERROR_REQUEST_ROUTER_FAILED_PREFIX: str = "request_router_failed:"
ERROR_WORKFLOW_PLANNER_FAILED_PREFIX: str = "workflow_planner_failed:"
ERROR_PLAN_VALIDATION_FAILED_PREFIX: str = "plan_validation_failed:"
ERROR_RESPONSE_MODEL_FAILED: str = "response_model_failed:"
ERROR_DUPLICATE_ACTION: str = "duplicate_action:"
ERROR_UNKNOWN_WORKFLOW_ACTION: str = "unknown_workflow_action:"

# [CV errors]
ERROR_CV_UPLOAD_REQUIRED: str = "cv_upload_required"
ERROR_CV_UPLOAD_PAYLOAD_MISSING: str = "cv_upload_payload_missing"
ERROR_CV_UPLOAD_NO_READABLE_DOCUMENTS: str = "cv_upload_no_readable_documents"
ERROR_CV_ALREADY_EXTRACTED: str = "cv_already_extracted"
ERROR_CV_EXTRACTION_REQUIRED: str = "cv_extraction_required"
ERROR_CV_EXTRACTION_REQUIRED_FOR_REVIEW: str = "cv_extraction_required_for_review"
ERROR_CV_TARGET_MISSING: str = "cv_target_missing"
ERROR_CV_REVIEW_FAILED: str = "cv_review_failed"
ERROR_CV_REVIEW_TARGET_COUNT_INVALID: str = "cv_review_target_count_invalid"
ERROR_CV_COMPARISON_FAILED: str = "cv_comparison_failed"
ERROR_CV_COMPARISON_TARGET_COUNT_INVALID: str = "cv_comparison_target_count_invalid"
ERROR_CV_COMPARISON_REQUIRES_TWO_DOCUMENTS: str = "cv_comparison_requires_two_documents"
ERROR_CV_FEATURES_REQUIRED_FOR_MATCHING: str = "cv_features_required_for_matching"

# [Job errors]
ERROR_JOB_TARGET_MISSING: str = "job_target_missing"
ERROR_JOB_TARGETS_MISSING_FOR_MATCHING: str = "job_targets_missing_for_matching"
ERROR_EXISTING_JOB_TARGETS_MISSING: str = "existing_job_targets_missing"
ERROR_JOB_DATA_REQUIRED_BEFORE_MATCHING: str = "job_data_required_before_matching"
ERROR_PASTED_JOB_REQUIRED: str = "pasted_job_required"
ERROR_PASTED_JOB_DESCRIPTION_MISSING: str = "pasted_job_description_missing"
ERROR_PASTED_JOB_EXTRACTION_FAILED: str = "pasted_job_extraction_failed"

# [Plan errors]
ERROR_ACTION_LIMIT_REACHED: str = "action_limit_reached"
ERROR_TARGET_AMBIGUOUS: str = "target_ambiguous"
ERROR_PLAN_VALIDATION_FAILED: str = "plan_validation_failed"

# [Response errors]
ERROR_RESPONSE_MODEL_EMPTY: str = "response_model_empty"

# [Router and planner reasons]
REASON_NO_USER_MESSAGE: str = "no_user_message"
REASON_REQUEST_ROUTER_FAILED: str = "request_router_failed"
REASON_WORKFLOW_PLANNER_FAILED: str = "workflow_planner_failed"
REASON_AWAITING_REQUEST_ROUTING: str = "awaiting_request_routing"
REASON_WORKFLOW_ACTION_SELECTED: str = "workflow_action_selected"
REASON_PLANNER_SELECTED_ACTION: str = "planner_selected_action"
REASON_PLAN_VALIDATED: str = "plan_validated"
REASON_SEARCH_COMPLETE_ASSESSMENT: str = "search_complete_assessment"
REASON_SEARCH_COMPLETE_PRESENT_RESULTS: str = "search_complete_present_results"

# [Execution trace reasons]
REASON_CV_FEATURES_READY: str = "cv_features_ready"
REASON_SEARCH_RESULTS_READY: str = "search_results_ready"
REASON_JOB_EXTRACTION_READY: str = "job_extraction_ready"
REASON_ACTION_COMPLETED: str = "action_completed"
REASON_ACTION_FAILED: str = "action_failed"
REASON_REUSED_EXISTING_RESULT: str = "reused_existing_result"

# [Plan validation statuses]
VALIDATION_PENDING: str = "pending"
VALIDATION_ACCEPTED: str = "accepted"
VALIDATION_REJECTED: str = "rejected"

# [Job goal sources and invalidation]
GOAL_SOURCE_EXPLICIT_SEARCH: str = "explicit_search"
GOAL_SOURCE_CV_DERIVED: str = "cv_derived"
GOAL_INVALIDATION_CANCELLED: str = "cancelled"

# [Job card placeholders]
JOB_CARD_UNTITLED: str = "Untitled job"
JOB_CARD_UNKNOWN_COMPANY: str = "Unknown company"
JOB_CARD_PASTED_TITLE: str = "Pasted job description"
JOB_CARD_PASTED_SITE: str = "user_pasted"

# [Result statuses]
STATUS_UNAVAILABLE: str = "unavailable"
STATUS_UNFAVORABLE: str = "unfavorable"
REVIEW_SCORE_SCALE: str = "0-100"
