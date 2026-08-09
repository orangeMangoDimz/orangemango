# [Chat Endpoint]
CHAT_ENDPOINT_DESCRIPTION = (
    "Accepts a validated message and starts a LangGraph response run for the "
    "specified thread. Generated events are available from GET /events."
)

# [API Authentication]
# Bandit exception: this is an environment variable name, not a credential.
API_AUTH_TOKEN_ENV = "API_AUTH_TOKEN"  # nosec B105
API_AUTH_NOT_CONFIGURED = "API authentication is not configured"
# Bandit exception: this is an error message, not a credential.
INVALID_API_AUTH_TOKEN = "Invalid or missing bearer token"  # nosec B105

# [Chat Metadata]
CHAT_ENDPOINT_SUMMARY = "Start a chat response"
CHAT_API_TAG = "Chat"
CHAT_SERVICE_NOT_CONFIGURED = "Chat service is not configured"

# [Chat Errors]
GENERIC_STREAM_ERROR = "Unable to generate a response."
THREAD_ID_MUST_NOT_BE_BLANK = "thread_id must not be blank"
CHAT_THREAD_NOT_FOUND = "Chat thread was not found"
CHAT_THREAD_BUSY = "A message is already being processed for this thread"
CHAT_STREAM_ERROR = "Unable to complete the response."

# [Chat Statuses]
MESSAGE_ACCEPTED = "accepted"
MESSAGE_STATUS_FAILED = "failed"
MESSAGE_STATUS_COMPLETED = "completed"

# [Events Endpoint]
EVENT_ENDPOINT_DESCRIPTION = (
    "Streams LangGraph state and assistant response events for a chat thread."
)
EVENT_ENDPOINT_SUMMARY = "Stream chat events"
EVENT_API_TAG = "Events"

# [Health Endpoint]
HEALTH_ENDPOINT_DESCRIPTION = (
    "Returns the current health status of the Orangemango API."
)
HEALTH_ENDPOINT_SUMMARY = "Check API health"
HEALTH_API_TAG = "Health"
HEALTH_STATUS_OK = "ok"

# [OpenAPI Tags]
OPENAPI_TAGS = [
    {
        "name": HEALTH_API_TAG,
        "description": "Service health and availability.",
    },
    {
        "name": CHAT_API_TAG,
        "description": "Start chat response runs.",
    },
    {
        "name": EVENT_API_TAG,
        "description": "Stream chat state and response events.",
    },
]

# [Validation and Rate Limits]
INVALID_RATE_LIMIT_REQUESTS = "max_requests must be greater than zero"
INVALID_RATE_LIMIT_WINDOW = "window_seconds must be greater than zero"
MESSAGE_MUST_NOT_BE_BLANK = "message must not be blank"
STREAM_DONE = "[DONE]"
TOO_MANY_CHAT_REQUESTS = "Too many chat requests"
